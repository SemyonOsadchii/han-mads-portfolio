from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from pathlib import Path
from time import perf_counter
from typing import Any

import mlflow
import pandas as pd
import ray
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from ray_trainable import trainable as ray_trainable
from tuning_common import (
    build_dataloaders,
    build_model,
    collect_efficiency_metrics,
    detect_device,
    fit_model,
    seed_everything,
    set_local_torch_cache,
)

ROOT = Path(__file__).resolve().parent
RAY_RESULTS_DIR = ROOT / "ray_results"

DEFAULT_RESULTS_FILE = ROOT / "results.csv"
DEFAULT_BEST_CONFIG_FILE = ROOT / "best_config.json"
DEFAULT_FINAL_CONFIG_FILE = ROOT / "final_trainer_config.json"
DEFAULT_MLFLOW_DB_FILE = ROOT / "mlflow_chapter4.db"
DEFAULT_MLFLOW_EXPERIMENT_NAME = "chapter-4-ray"

Preset = dict[str, Any]
SweepSpec = dict[str, Any]


PRESETS: dict[str, Preset] = {
    "balanced": {
        "mode": "balanced",
        "description": "Compact ResNet sweep on the ants and bees dataset.",
        "family_samples": {"resnet18": 4},
        "max_epochs": 4,
        "image_size": 160,
        "class_limit": None,
        "transformer_epochs": 2,
        "include_transformer_baseline": True,
        "scheduler": "asha",
        "grace_period": 1,
        "reduction_factor": 2,
        "search_space": "balanced",
        "cpu_per_trial": 2,
        "gpu_per_trial": 0,
    },
    "extensive": {
        "mode": "extensive",
        "description": "Broader search over ResNet18 transfer learning and compact transformers.",
        "family_samples": {"resnet18": 12, "transformer": 10},
        "max_epochs": 8,
        "image_size": 192,
        "class_limit": None,
        "transformer_epochs": 4,
        "include_transformer_baseline": False,
        "scheduler": "asha",
        "grace_period": 2,
        "reduction_factor": 2,
        "search_space": "extensive",
        "cpu_per_trial": 2,
        "gpu_per_trial": 0,
    },
    "focused": {
        "mode": "focused",
        "description": "Narrow follow-up sweep around the strongest ResNet region.",
        "family_samples": {"resnet18": 12},
        "max_epochs": 8,
        "image_size": 192,
        "class_limit": None,
        "transformer_epochs": 4,
        "include_transformer_baseline": True,
        "scheduler": "none",
        "grace_period": 8,
        "reduction_factor": 2,
        "search_space": "focused",
        "cpu_per_trial": 2,
        "gpu_per_trial": 0,
    },
    "overnight": {
        "mode": "overnight",
        "description": "Long overnight sweep over transfer learning and compact transformers.",
        "family_samples": {"resnet18": 32, "transformer": 26},
        "max_epochs": 20,
        "image_size": 224,
        "class_limit": None,
        "transformer_epochs": 8,
        "include_transformer_baseline": False,
        "scheduler": "asha",
        "grace_period": 5,
        "reduction_factor": 2,
        "search_space": "overnight",
        "cpu_per_trial": 2,
        "gpu_per_trial": 0,
    },
}

FINAL_TRAINER_DEFAULTS: dict[str, Any] = {
    "model_type": "resnet18",
    "image_size": 192,
    "epochs": 10,
    "class_limit": None,
    "seed": 42,
    "num_workers": 0,
    "batch_size": 16,
    "lr": 1e-3,
    "dropout": 0.15,
    "head_width": 384,
    "head_layers": 1,
    "head_batch_norm": 0,
    "activation": "relu",
    "embed_dim": 96,
    "depth": 3,
    "num_heads": 4,
    "patch_size": 16,
    "optimizer": "adamw",
    "unfreeze_blocks": 0,
    "weight_decay": 1e-4,
    "label_smoothing": 0.0,
}


def load_reference_config(best_config_file: Path) -> dict[str, Any]:
    if not best_config_file.exists():
        return {}
    try:
        payload = json.loads(best_config_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    config = payload.get("config", {})
    return config if isinstance(config, dict) else {}


def configure_mlflow(
    mlflow_db_file: Path = DEFAULT_MLFLOW_DB_FILE,
    experiment_name: str = DEFAULT_MLFLOW_EXPERIMENT_NAME,
) -> None:
    tracking_uri = f"sqlite:///{mlflow_db_file.resolve().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def build_final_trainer_config(best_config_file: Path) -> dict[str, Any]:
    reference = load_reference_config(best_config_file)
    final_config = dict(FINAL_TRAINER_DEFAULTS)
    final_config.update(
        {
            "model_type": str(reference.get("model_type", final_config["model_type"])),
            "batch_size": int(reference.get("batch_size", final_config["batch_size"])),
            "lr": float(reference.get("lr", final_config["lr"])),
            "dropout": float(reference.get("dropout", final_config["dropout"])),
            "head_width": int(reference.get("head_width", final_config["head_width"])),
            "head_layers": int(reference.get("head_layers", final_config["head_layers"])),
            "head_batch_norm": int(reference.get("head_batch_norm", final_config["head_batch_norm"])),
            "activation": str(reference.get("activation", final_config["activation"])),
            "embed_dim": int(reference.get("embed_dim", final_config["embed_dim"])),
            "depth": int(reference.get("depth", final_config["depth"])),
            "num_heads": int(reference.get("num_heads", final_config["num_heads"])),
            "patch_size": int(reference.get("patch_size", final_config["patch_size"])),
            "optimizer": str(reference.get("optimizer", final_config["optimizer"])),
            "unfreeze_blocks": int(reference.get("unfreeze_blocks", final_config["unfreeze_blocks"])),
            "weight_decay": float(reference.get("weight_decay", final_config["weight_decay"])),
            "label_smoothing": float(reference.get("label_smoothing", final_config["label_smoothing"])),
        }
    )
    return final_config


def build_network_name(record: dict[str, Any]) -> str:
    if record["model_type"] == "transformer":
        if record.get("run_type") == "baseline":
            return "tiny_transformer_baseline"
        return (
            "vision_transformer"
            f"_patch{record['patch_size']}"
            f"_embed{record['embed_dim']}"
            f"_depth{record['depth']}"
            f"_heads{record['num_heads']}"
            f"_head{record['head_width']}"
            f"_layers{record['head_layers']}"
            f"_{record['activation']}"
            f"_bn{record['head_batch_norm']}"
        )
    return (
        "resnet18_transfer"
        f"_head{record['head_width']}"
        f"_layers{record['head_layers']}"
        f"_{record['activation']}"
        f"_bn{record['head_batch_norm']}"
        f"_unfreeze{record['unfreeze_blocks']}"
    )


def build_network_display_name(record: dict[str, Any]) -> str:
    if record["model_type"] == "transformer":
        if record.get("run_type") == "baseline":
            return "Tiny transformer baseline"
        return (
            "Transformer"
            f" | patch={record['patch_size']}"
            f" | embed={record['embed_dim']}"
            f" | depth={record['depth']}"
            f" | heads={record['num_heads']}"
            f" | head={record['head_width']}"
            f" | layers={record['head_layers']}"
            f" | activation={record['activation']}"
            f" | bn={record['head_batch_norm']}"
        )
    return (
        "ResNet18 transfer"
        f" | head={record['head_width']}"
        f" | layers={record['head_layers']}"
        f" | activation={record['activation']}"
        f" | bn={record['head_batch_norm']}"
        f" | unfreeze={record['unfreeze_blocks']}"
    )


def get_num_samples(preset: Preset, model_type: str | None = None) -> int:
    family_samples = preset["family_samples"]
    if model_type is None:
        return int(sum(int(value) for value in family_samples.values()))
    return int(family_samples[model_type])


def unique_preserve_order(values: list[Any]) -> list[Any]:
    unique_values: list[Any] = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)
    return unique_values


def scaled_lr_choices(center: float) -> list[float]:
    raw_values = [center * scale for scale in (0.5, 0.75, 1.0, 1.25, 1.5)]
    clipped = [round(value, 6) for value in raw_values if 1e-5 <= value <= 3e-3]
    return unique_preserve_order(clipped)


def nearby_dropout_choices(center: float) -> list[float]:
    raw_values = [center - 0.1, center - 0.05, center, center + 0.05, center + 0.1]
    clipped = [round(min(0.4, max(0.0, value)), 2) for value in raw_values]
    return unique_preserve_order(clipped)


def nearby_head_width_choices(center: int) -> list[int]:
    allowed = [128, 256, 384, 512]
    return [value for value in allowed if abs(value - center) <= 128] or [256, 384, 512]


def search_options_for_mode(preset: Preset, model_type: str, best_config_file: Path) -> dict[str, Any]:
    if preset["search_space"] == "balanced":
        return {
            "model_type": model_type,
            "lr": [1e-4, 3e-4, 1e-3],
            "dropout": [0.15, 0.25, 0.35],
            "head_width": [128, 256, 384],
            "head_layers": 1,
            "head_batch_norm": 0,
            "activation": "relu",
            "embed_dim": 96 if model_type == "resnet18" else [64, 96],
            "depth": 3 if model_type == "resnet18" else [2, 3],
            "num_heads": 4 if model_type == "resnet18" else [4, 8],
            "patch_size": 16 if model_type == "resnet18" else [16, 32],
            "optimizer": ["adamw", "sgd"],
            "batch_size": [8, 16],
            "unfreeze_blocks": [0, 1] if model_type == "resnet18" else 0,
            "weight_decay": 1e-4,
            "label_smoothing": 0.0,
        }

    if preset["search_space"] == "extensive":
        if model_type == "resnet18":
            return {
                "model_type": "resnet18",
                "lr": [1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 1.5e-3, 2e-3],
                "dropout": [0.0, 0.1, 0.15, 0.2, 0.25, 0.35],
                "head_width": [128, 256, 384, 512],
                "head_layers": [1, 2],
                "head_batch_norm": [0, 1],
                "activation": ["relu", "gelu"],
                "embed_dim": 96,
                "depth": 3,
                "num_heads": 4,
                "patch_size": 16,
                "optimizer": ["adamw", "sgd"],
                "batch_size": [8, 16, 24],
                "unfreeze_blocks": [0, 1, 2],
                "weight_decay": [1e-5, 5e-5, 1e-4, 5e-4],
                "label_smoothing": [0.0, 0.05, 0.1],
            }
        return {
            "model_type": "transformer",
            "lr": [1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3],
            "dropout": [0.0, 0.1, 0.15, 0.2, 0.25],
            "head_width": [128, 256, 384],
            "head_layers": [1, 2],
            "head_batch_norm": 0,
            "activation": ["relu", "gelu"],
            "embed_dim": [64, 96, 128, 192],
            "depth": [2, 3, 4],
            "num_heads": [4, 8],
            "patch_size": [16, 32],
            "optimizer": ["adamw"],
            "batch_size": [8, 16],
            "unfreeze_blocks": 0,
            "weight_decay": [1e-5, 5e-5, 1e-4, 5e-4],
            "label_smoothing": [0.0, 0.05],
        }

    if preset["search_space"] == "overnight":
        if model_type == "resnet18":
            return {
                "model_type": "resnet18",
                "lr": [5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 1.5e-3, 2e-3],
                "dropout": [0.0, 0.1, 0.15, 0.2, 0.25, 0.35, 0.45],
                "head_width": [128, 256, 384, 512, 768],
                "head_layers": [1, 2, 3],
                "head_batch_norm": [0, 1],
                "activation": ["relu", "gelu", "silu"],
                "embed_dim": 96,
                "depth": 3,
                "num_heads": 4,
                "patch_size": 16,
                "optimizer": ["adamw", "sgd"],
                "batch_size": [8, 16, 24, 32],
                "unfreeze_blocks": [0, 1, 2, 3, 4],
                "weight_decay": [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                "label_smoothing": [0.0, 0.05, 0.1, 0.15],
            }
        return {
            "model_type": "transformer",
            "lr": [5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3],
            "dropout": [0.0, 0.1, 0.15, 0.2, 0.25, 0.35],
            "head_width": [128, 256, 384, 512],
            "head_layers": [1, 2, 3],
            "head_batch_norm": [0, 1],
            "activation": ["relu", "gelu", "silu"],
            "embed_dim": [64, 96, 128, 192, 256],
            "depth": [2, 3, 4, 6],
            "num_heads": [4, 8],
            "patch_size": [8, 16, 32],
            "optimizer": ["adamw"],
            "batch_size": [8, 16, 24],
            "unfreeze_blocks": 0,
            "weight_decay": [1e-6, 1e-5, 5e-5, 1e-4, 5e-4],
            "label_smoothing": [0.0, 0.05, 0.1],
        }

    reference = load_reference_config(best_config_file)
    reference_lr = float(reference.get("lr", 1e-3))
    reference_dropout = float(reference.get("dropout", 0.15))
    reference_width = int(reference.get("head_width", 384))
    reference_batch_size = int(reference.get("batch_size", 16))
    reference_unfreeze = int(reference.get("unfreeze_blocks", 0))
    reference_optimizer = str(reference.get("optimizer", "adamw"))

    return {
        "model_type": model_type,
        "lr": scaled_lr_choices(reference_lr),
        "dropout": nearby_dropout_choices(reference_dropout),
        "head_width": nearby_head_width_choices(reference_width),
        "head_layers": [1, 2],
        "head_batch_norm": [0, 1],
        "activation": ["relu", "gelu"],
        "embed_dim": 96 if model_type == "resnet18" else [64, 96],
        "depth": 3 if model_type == "resnet18" else [2, 3],
        "num_heads": 4 if model_type == "resnet18" else [4, 8],
        "patch_size": 16 if model_type == "resnet18" else [16, 32],
        "optimizer": unique_preserve_order([reference_optimizer, "adamw"]),
        "batch_size": unique_preserve_order([reference_batch_size, 16, 24]),
        "unfreeze_blocks": unique_preserve_order([reference_unfreeze, 0, 1]) if model_type == "resnet18" else 0,
        "weight_decay": [1e-5, 5e-5, 1e-4, 5e-4],
        "label_smoothing": [0.0, 0.05],
    }


def build_search_config(preset: Preset, model_type: str, best_config_file: Path) -> dict[str, Any]:
    options = search_options_for_mode(preset, model_type, best_config_file)
    config: dict[str, Any] = {
        "image_size": preset["image_size"],
        "epochs": preset["max_epochs"],
        "class_limit": preset["class_limit"],
        "seed": 42,
        "num_workers": 0,
    }
    for key, value in options.items():
        config[key] = tune.choice(value) if isinstance(value, list) else value
    return config


def estimate_total_combinations(options: dict[str, Any]) -> int:
    combination_counts = [len(value) for value in options.values() if isinstance(value, list)]
    return math.prod(combination_counts) if combination_counts else 1


def build_scheduler(preset: Preset) -> ASHAScheduler | None:
    if preset["scheduler"] == "none":
        return None
    if preset["scheduler"] == "asha":
        return ASHAScheduler(
            max_t=preset["max_epochs"],
            grace_period=preset["grace_period"],
            reduction_factor=preset["reduction_factor"],
        )
    raise ValueError(f"Unknown scheduler: {preset['scheduler']}")


def load_trial_history(result_json_file: Path) -> list[dict[str, Any]]:
    if not result_json_file.exists():
        return []
    history: list[dict[str, Any]] = []
    for line in result_json_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if "val_accuracy" in payload and "val_loss" in payload and "epoch" in payload:
            history.append(payload)
    return history


def summarize_trial_history(result_json_file: Path) -> dict[str, dict[str, Any]]:
    history = load_trial_history(result_json_file)
    if not history:
        empty_metrics = {"val_accuracy": None, "val_loss": None, "epoch": None}
        return {"best": empty_metrics, "final": empty_metrics}

    best_metrics = max(history, key=lambda item: float(item.get("val_accuracy", -1.0)))
    final_metrics = history[-1]
    return {
        "best": {
            "val_accuracy": best_metrics.get("val_accuracy"),
            "val_loss": best_metrics.get("val_loss"),
            "epoch": best_metrics.get("epoch"),
        },
        "final": {
            "val_accuracy": final_metrics.get("val_accuracy"),
            "val_loss": final_metrics.get("val_loss"),
            "epoch": final_metrics.get("epoch"),
        },
    }


def first_defined(*values: Any) -> Any:
    for value in values:
        if value is not None and not pd.isna(value):
            return value
    return None


def config_from_record(record: dict[str, Any] | pd.Series) -> dict[str, Any]:
    return {
        "model_type": str(record["model_type"]),
        "image_size": int(record["image_size"]),
        "dropout": float(record["dropout"]),
        "head_width": int(record["head_width"]),
        "head_layers": int(record["head_layers"]),
        "head_batch_norm": int(record["head_batch_norm"]),
        "activation": str(record["activation"]),
        "embed_dim": int(record["embed_dim"]),
        "depth": int(record["depth"]),
        "num_heads": int(record["num_heads"]),
        "patch_size": int(record["patch_size"]),
        "unfreeze_blocks": int(record["unfreeze_blocks"]),
    }


def efficiency_key(config: dict[str, Any]) -> tuple[Any, ...]:
    return (
        config["model_type"],
        config["image_size"],
        config["head_width"],
        config["head_layers"],
        config["head_batch_norm"],
        config["activation"],
        config["embed_dim"],
        config["depth"],
        config["num_heads"],
        config["patch_size"],
    )


def local_config_from_record(record: dict[str, Any] | pd.Series) -> dict[str, Any]:
    config = config_from_record(record)
    config.update(
        {
            "batch_size": int(record["batch_size"]),
            "lr": float(record["lr"]),
            "optimizer": str(record["optimizer"]),
            "weight_decay": float(record["weight_decay"]),
            "label_smoothing": float(record["label_smoothing"]),
        }
    )
    return config


def add_efficiency_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    set_local_torch_cache()
    enriched = results_df.copy()
    valid_loader_cache: dict[int, tuple[Any, int]] = {}
    efficiency_cache: dict[tuple[Any, ...], dict[str, float | int]] = {}

    for index, row in enriched.iterrows():
        config = config_from_record(row)
        image_size = int(config["image_size"])
        if image_size not in valid_loader_cache:
            _, valid_loader, num_classes = build_dataloaders(
                image_size=image_size,
                batch_size=8,
                class_limit=None,
                num_workers=0,
            )
            valid_loader_cache[image_size] = (valid_loader, num_classes)

        key = efficiency_key(config)
        if key not in efficiency_cache:
            valid_loader, num_classes = valid_loader_cache[image_size]
            model = build_model(num_classes=num_classes, config=config)
            efficiency_cache[key] = collect_efficiency_metrics(model, valid_loader)

        for metric_name, metric_value in efficiency_cache[key].items():
            enriched.at[index, metric_name] = metric_value

    return enriched


def run_transformer_baseline(preset: Preset) -> dict[str, Any]:
    seed_everything(42)
    device = detect_device()
    baseline_config = {
        "model_type": "transformer",
        "image_size": preset["image_size"],
        "dropout": 0.15,
        "head_width": 128,
        "head_layers": 1,
        "head_batch_norm": 0,
        "activation": "relu",
        "embed_dim": 64,
        "depth": 2,
        "num_heads": 4,
        "patch_size": 16,
        "unfreeze_blocks": 0,
    }
    train_loader, valid_loader, num_classes = build_dataloaders(
        image_size=preset["image_size"],
        batch_size=8,
        class_limit=preset["class_limit"],
        num_workers=0,
    )
    model = build_model(num_classes=num_classes, config=baseline_config)
    metrics = fit_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        epochs=preset["transformer_epochs"],
        optimizer_name="adamw",
        lr=3e-4,
        weight_decay=1e-4,
        label_smoothing=0.0,
    )
    return {
        "run_type": "baseline",
        "model_type": "transformer",
        "mode": preset["mode"],
        "trial_id": None,
        "val_accuracy": metrics["best"]["val_accuracy"],
        "val_loss": metrics["best"]["val_loss"],
        "epoch": metrics["best"]["epoch"],
        "final_val_accuracy": metrics["final"]["val_accuracy"],
        "final_val_loss": metrics["final"]["val_loss"],
        "final_epoch": metrics["final"]["epoch"],
        "batch_size": 8,
        "lr": 3e-4,
        "dropout": 0.15,
        "head_width": 128,
        "head_layers": 1,
        "head_batch_norm": 0,
        "activation": "relu",
        "embed_dim": 64,
        "depth": 2,
        "num_heads": 4,
        "patch_size": 16,
        "optimizer": "adamw",
        "unfreeze_blocks": 0,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "image_size": preset["image_size"],
        "notes": "Small comparison run with a patch-embedding transformer.",
        "param_count": metrics["efficiency"]["param_count"],
        "model_size_kb": metrics["efficiency"]["model_size_kb"],
        "inference_ms_per_batch": metrics["efficiency"]["inference_ms_per_batch"],
        "inference_ms_per_image": metrics["efficiency"]["inference_ms_per_image"],
        "best_val_accuracy": metrics["best"]["val_accuracy"],
        "best_val_loss": metrics["best"]["val_loss"],
        "best_epoch": metrics["best"]["epoch"],
    }


def build_record_from_trial(preset: Preset, trial: Any) -> dict[str, Any]:
    result = trial.last_result or {}
    history_metrics = summarize_trial_history(Path(trial.path) / "result.json")
    best_metrics = history_metrics["best"]
    final_metrics = history_metrics["final"]
    record = {
        "run_type": "sweep",
        "model_type": trial.config.get("model_type", result.get("model_type", "resnet18")),
        "mode": preset["mode"],
        "trial_id": trial.trial_id,
        "val_accuracy": first_defined(best_metrics["val_accuracy"], result.get("val_accuracy")),
        "val_loss": first_defined(best_metrics["val_loss"], result.get("val_loss")),
        "epoch": first_defined(best_metrics["epoch"], result.get("epoch")),
        "final_val_accuracy": first_defined(final_metrics["val_accuracy"], result.get("val_accuracy")),
        "final_val_loss": first_defined(final_metrics["val_loss"], result.get("val_loss")),
        "final_epoch": first_defined(final_metrics["epoch"], result.get("epoch")),
        "batch_size": trial.config.get("batch_size"),
        "lr": trial.config.get("lr"),
        "dropout": trial.config.get("dropout"),
        "head_width": trial.config.get("head_width"),
        "head_layers": trial.config.get("head_layers"),
        "head_batch_norm": trial.config.get("head_batch_norm"),
        "activation": trial.config.get("activation"),
        "embed_dim": trial.config.get("embed_dim"),
        "depth": trial.config.get("depth"),
        "num_heads": trial.config.get("num_heads"),
        "patch_size": trial.config.get("patch_size"),
        "optimizer": trial.config.get("optimizer"),
        "unfreeze_blocks": trial.config.get("unfreeze_blocks"),
        "weight_decay": trial.config.get("weight_decay"),
        "label_smoothing": trial.config.get("label_smoothing"),
        "image_size": trial.config.get("image_size"),
        "notes": preset["description"],
        "param_count": None,
        "model_size_kb": None,
        "inference_ms_per_batch": None,
        "inference_ms_per_image": None,
        "best_val_accuracy": first_defined(best_metrics["val_accuracy"], result.get("val_accuracy")),
        "best_val_loss": first_defined(best_metrics["val_loss"], result.get("val_loss")),
        "best_epoch": first_defined(best_metrics["epoch"], result.get("epoch")),
    }
    record["network_name"] = build_network_name(record)
    record["network_display_name"] = build_network_display_name(record)
    return record


def join_unique_values(series: pd.Series) -> str:
    values: list[str] = []
    for value in series.dropna().tolist():
        string_value = str(value)
        if string_value not in values:
            values.append(string_value)
    return ", ".join(values)


def build_sweep_specs(preset: Preset, best_config_file: Path) -> list[SweepSpec]:
    sweeps: list[SweepSpec] = []
    for model_type, num_samples in preset["family_samples"].items():
        search_options = search_options_for_mode(preset, model_type, best_config_file)
        sweeps.append(
            {
                "model_type": model_type,
                "num_samples": int(num_samples),
                "search_options": search_options,
                "search_space": build_search_config(preset, model_type, best_config_file),
            }
        )
    return sweeps


def build_network_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results_df.groupby(
            ["network_name", "network_display_name", "model_type"],
            dropna=False,
        )
        .agg(
            runs_logged=("network_name", "size"),
            best_val_accuracy=("val_accuracy", "max"),
            mean_val_accuracy=("val_accuracy", "mean"),
            best_val_loss=("val_loss", "min"),
            fastest_inference_ms_per_batch=("inference_ms_per_batch", "min"),
            fastest_inference_ms_per_image=("inference_ms_per_image", "min"),
            min_param_count=("param_count", "min"),
            min_model_size_kb=("model_size_kb", "min"),
            optimizers=("optimizer", join_unique_values),
            batch_sizes=("batch_size", join_unique_values),
            notes=("notes", "first"),
        )
        .reset_index()
        .sort_values("best_val_accuracy", ascending=False)
    )
    return summary


def log_results_to_mlflow(
    preset: Preset,
    results_df: pd.DataFrame,
    network_summary_df: pd.DataFrame,
    results_file: Path,
    best_config_file: Path,
    sweep_specs: list[SweepSpec],
    mlflow_db_file: Path = DEFAULT_MLFLOW_DB_FILE,
) -> None:
    configure_mlflow(mlflow_db_file)
    run_name = f"ray_{preset['mode']}_summary"
    includes_transformer_search = any(spec["model_type"] == "transformer" for spec in sweep_specs)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "chapter": "4",
                "framework": "ray",
                "mode": preset["mode"],
            }
        )
        mlflow.log_params(
            {
                "mode": preset["mode"],
                "search_space": preset["search_space"],
                "scheduler": preset["scheduler"],
                "num_samples": get_num_samples(preset),
                "max_epochs": preset["max_epochs"],
                "image_size": preset["image_size"],
                "include_transformer_baseline": int(preset["include_transformer_baseline"]),
                "includes_transformer_search": int(includes_transformer_search),
            }
        )
        metrics_to_log = {
            "num_logged_rows": int(len(results_df)),
            "num_sweep_trials": int((results_df["run_type"] == "sweep").sum()),
            "num_unique_networks": int(network_summary_df["network_name"].nunique()),
            "num_transformer_rows": int((results_df["model_type"] == "transformer").sum()),
            "num_resnet_rows": int((results_df["model_type"] == "resnet18").sum()),
            "best_val_accuracy": float(results_df["val_accuracy"].max()),
            "best_val_loss": float(results_df["val_loss"].min()),
        }
        if "inference_ms_per_batch" in results_df.columns and results_df["inference_ms_per_batch"].notna().any():
            metrics_to_log["fastest_inference_ms_per_batch"] = float(results_df["inference_ms_per_batch"].min())
        mlflow.log_metrics(metrics_to_log)
        mlflow.log_table(
            results_df.sort_values("val_accuracy", ascending=False),
            "tables/trials_full.json",
        )
        mlflow.log_table(network_summary_df, "tables/network_summary.json")
        mlflow.log_text(
            network_summary_df.to_csv(index=False),
            "tables/network_summary.csv",
        )
        mlflow.log_artifact(str(results_file))
        mlflow.log_artifact(str(best_config_file))
    print(f"Logged MLflow summary to experiment '{DEFAULT_MLFLOW_EXPERIMENT_NAME}'.")


def ray_runtime_env() -> dict[str, dict[str, str]]:
    python_path = str(ROOT)
    existing_python_path = os.environ.get("PYTHONPATH")
    if existing_python_path:
        python_path = os.pathsep.join([python_path, existing_python_path])
    return {"env_vars": {"PYTHONPATH": python_path}}


def resources_for_preset(preset: Preset) -> dict[str, float | int]:
    resources: dict[str, float | int] = {"cpu": int(preset["cpu_per_trial"])}
    gpu_per_trial = float(preset.get("gpu_per_trial", 0) or 0)
    if gpu_per_trial > 0:
        resources["gpu"] = gpu_per_trial
    return resources


def smoke_preset(source: Preset) -> Preset:
    preset = dict(source)
    preset.update(
        {
            "mode": f"{source['mode']}-smoke",
            "description": f"Smoke check for {source['mode']} preset.",
            "family_samples": {"resnet18": 1},
            "max_epochs": 1,
            "image_size": min(int(source["image_size"]), 96),
            "class_limit": 2,
            "transformer_epochs": 1,
            "include_transformer_baseline": False,
            "scheduler": "none",
            "grace_period": 1,
            "search_space": "balanced",
        }
    )
    return preset


def sample_option(value: Any, rng: random.Random) -> Any:
    if isinstance(value, list):
        return rng.choice(value)
    return value


def build_local_config(
    preset: Preset,
    model_type: str,
    best_config_file: Path,
    sample_index: int,
) -> dict[str, Any]:
    rng = random.Random(42 + sample_index)
    options = search_options_for_mode(preset, model_type, best_config_file)
    config = {
        "image_size": preset["image_size"],
        "epochs": preset["max_epochs"],
        "class_limit": preset["class_limit"],
        "seed": 42 + sample_index,
        "num_workers": 0,
    }
    config.update({key: sample_option(value, rng) for key, value in options.items()})
    return config


def build_local_record(
    preset: Preset,
    config: dict[str, Any],
    metrics: dict[str, Any],
    trial_id: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    record = {
        "run_type": "local",
        "model_type": config["model_type"],
        "mode": preset["mode"],
        "trial_id": trial_id,
        "val_accuracy": metrics["best"]["val_accuracy"],
        "val_loss": metrics["best"]["val_loss"],
        "epoch": metrics["best"]["epoch"],
        "final_val_accuracy": metrics["final"]["val_accuracy"],
        "final_val_loss": metrics["final"]["val_loss"],
        "final_epoch": metrics["final"]["epoch"],
        "batch_size": config["batch_size"],
        "lr": config["lr"],
        "dropout": config["dropout"],
        "head_width": config["head_width"],
        "head_layers": config["head_layers"],
        "head_batch_norm": config["head_batch_norm"],
        "activation": config["activation"],
        "embed_dim": config["embed_dim"],
        "depth": config["depth"],
        "num_heads": config["num_heads"],
        "patch_size": config["patch_size"],
        "optimizer": config["optimizer"],
        "unfreeze_blocks": config["unfreeze_blocks"],
        "weight_decay": config["weight_decay"],
        "label_smoothing": config["label_smoothing"],
        "image_size": config["image_size"],
        "notes": f"{preset['description']} Local sequential CUDA-compatible run.",
        "elapsed_seconds": elapsed_seconds,
        "param_count": metrics["efficiency"]["param_count"],
        "model_size_kb": metrics["efficiency"]["model_size_kb"],
        "inference_ms_per_batch": metrics["efficiency"]["inference_ms_per_batch"],
        "inference_ms_per_image": metrics["efficiency"]["inference_ms_per_image"],
        "best_val_accuracy": metrics["best"]["val_accuracy"],
        "best_val_loss": metrics["best"]["val_loss"],
        "best_epoch": metrics["best"]["epoch"],
    }
    record["network_name"] = build_network_name(record)
    record["network_display_name"] = build_network_display_name(record)
    return record


def release_trial_memory(device: Any) -> None:
    gc.collect()
    if getattr(device, "type", None) == "cuda":
        torch.cuda.empty_cache()


def normalize_existing_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in record.items():
        normalized[key] = None if pd.isna(value) else value
    return normalized


def load_existing_local_records(
    preset: Preset,
    results_file: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not results_file.exists():
        return [], {}

    existing_df = pd.read_csv(results_file)
    if "trial_id" not in existing_df.columns or "run_type" not in existing_df.columns:
        return [], {}

    mode = str(preset["mode"])
    existing_df = existing_df[
        (existing_df["mode"].astype(str) == mode)
        & (existing_df["run_type"].astype(str) == "local")
        & existing_df["trial_id"].notna()
    ]
    if existing_df.empty:
        return [], {}

    records: list[dict[str, Any]] = []
    config_by_trial_id: dict[str, dict[str, Any]] = {}
    for raw_record in existing_df.to_dict("records"):
        record = normalize_existing_record(raw_record)
        trial_id = str(record["trial_id"])
        records.append(record)
        config_by_trial_id[trial_id] = local_config_from_record(record)
    return records, config_by_trial_id


def persist_result_files(
    preset: Preset,
    records: list[dict[str, Any]],
    results_file: Path,
    best_config_file: Path,
    sweep_specs: list[SweepSpec],
    config_by_trial_id: dict[str, dict[str, Any]],
    baseline: dict[str, Any] | None,
) -> pd.DataFrame:
    results_file.parent.mkdir(parents=True, exist_ok=True)
    best_config_file.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(records).sort_values("val_accuracy", ascending=False).reset_index(drop=True)
    efficiency_columns = ["param_count", "model_size_kb", "inference_ms_per_batch", "inference_ms_per_image"]
    non_local_rows = results_df["run_type"].astype(str) != "local"
    if results_df.loc[non_local_rows, efficiency_columns].isna().any().any():
        results_df = add_efficiency_metrics(results_df)
    best_row = results_df.iloc[0]
    best_payload = {
        "mode": preset["mode"],
        "preset": preset,
        "sweeps": [
            {
                "model_type": sweep["model_type"],
                "num_samples": sweep["num_samples"],
                "search_options": sweep["search_options"],
                "estimated_total_configurations": estimate_total_combinations(sweep["search_options"]),
            }
            for sweep in sweep_specs
        ],
        "best_trial_id": best_row["trial_id"],
        "val_accuracy": float(best_row["val_accuracy"]),
        "val_loss": float(best_row["val_loss"]),
        "config": config_by_trial_id.get(str(best_row["trial_id"]), {}),
        "transformer_baseline": baseline,
    }
    results_df.to_csv(results_file, index=False)
    best_config_file.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    return results_df


def write_results(
    preset: Preset,
    records: list[dict[str, Any]],
    results_file: Path,
    best_config_file: Path,
    sweep_specs: list[SweepSpec],
    config_by_trial_id: dict[str, dict[str, Any]],
    baseline: dict[str, Any] | None,
    mlflow_db_file: Path,
) -> pd.DataFrame:
    results_df = persist_result_files(
        preset=preset,
        records=records,
        results_file=results_file,
        best_config_file=best_config_file,
        sweep_specs=sweep_specs,
        config_by_trial_id=config_by_trial_id,
        baseline=baseline,
    )
    network_summary_df = build_network_summary(results_df)
    log_results_to_mlflow(
        preset=preset,
        results_df=results_df,
        network_summary_df=network_summary_df,
        results_file=results_file,
        best_config_file=best_config_file,
        sweep_specs=sweep_specs,
        mlflow_db_file=mlflow_db_file,
    )
    return results_df


def run_local_search(
    preset: Preset,
    results_file: Path,
    best_config_file: Path,
    mlflow_db_file: Path = DEFAULT_MLFLOW_DB_FILE,
) -> pd.DataFrame:
    set_local_torch_cache()
    seed_everything(42)
    device = detect_device()
    sweep_specs = build_sweep_specs(preset, best_config_file)
    records, config_by_trial_id = load_existing_local_records(preset, results_file)
    completed_trial_ids = set(config_by_trial_id)
    baseline: dict[str, Any] | None = None
    if completed_trial_ids:
        print(f"Resuming local search with {len(completed_trial_ids)} completed trial(s).", flush=True)

    sample_offset = 0
    for sweep in sweep_specs:
        model_type = str(sweep["model_type"])
        for sample_index in range(int(sweep["num_samples"])):
            trial_id = f"local_{model_type}_{sample_index:03d}"
            if trial_id in completed_trial_ids:
                continue
            config = build_local_config(preset, model_type, best_config_file, sample_offset + sample_index)
            train_loader = None
            valid_loader = None
            model = None
            try:
                train_loader, valid_loader, num_classes = build_dataloaders(
                    image_size=int(config["image_size"]),
                    batch_size=int(config["batch_size"]),
                    class_limit=config["class_limit"],
                    num_workers=int(config.get("num_workers", 0)),
                )
                model = build_model(num_classes=num_classes, config=config)
                started = perf_counter()
                metrics = fit_model(
                    model=model,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    device=device,
                    epochs=int(config["epochs"]),
                    optimizer_name=str(config["optimizer"]),
                    lr=float(config["lr"]),
                    weight_decay=float(config.get("weight_decay", 1e-4)),
                    label_smoothing=float(config.get("label_smoothing", 0.0)),
                )
                elapsed_seconds = perf_counter() - started
                config_by_trial_id[trial_id] = config
                records.append(build_local_record(preset, config, metrics, trial_id, elapsed_seconds))
                persist_result_files(
                    preset=preset,
                    records=records,
                    results_file=results_file,
                    best_config_file=best_config_file,
                    sweep_specs=sweep_specs,
                    config_by_trial_id=config_by_trial_id,
                    baseline=baseline,
                )
                print(
                    "Local trial:",
                    {
                        "trial_id": trial_id,
                        "model_type": model_type,
                        "device": str(device),
                        "elapsed_seconds": round(elapsed_seconds, 1),
                        "val_accuracy": metrics["best"]["val_accuracy"],
                    },
                    flush=True,
                )
            finally:
                del model
                del train_loader
                del valid_loader
                release_trial_memory(device)
        sample_offset += int(sweep["num_samples"])

    if preset["include_transformer_baseline"]:
        baseline = run_transformer_baseline(preset)
        baseline["network_name"] = build_network_name(baseline)
        baseline["network_display_name"] = build_network_display_name(baseline)
        records.append(baseline)

    return write_results(
        preset=preset,
        records=records,
        results_file=results_file,
        best_config_file=best_config_file,
        sweep_specs=sweep_specs,
        config_by_trial_id=config_by_trial_id,
        baseline=baseline,
        mlflow_db_file=mlflow_db_file,
    )


def run_search(
    preset: Preset,
    results_file: Path,
    best_config_file: Path,
    ray_results_dir: Path = RAY_RESULTS_DIR,
    mlflow_db_file: Path = DEFAULT_MLFLOW_DB_FILE,
    max_concurrent_trials: int | None = None,
    ray_num_cpus: int | None = None,
) -> pd.DataFrame:
    set_local_torch_cache()
    seed_everything(42)
    ray_results_dir = ray_results_dir.resolve()

    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=True,
            num_cpus=ray_num_cpus,
            runtime_env=ray_runtime_env(),
        )

    reporter = tune.CLIReporter(
        metric_columns=["val_loss", "val_accuracy", "training_iteration"],
        parameter_columns=[
            "lr",
            "dropout",
            "head_width",
            "head_layers",
            "optimizer",
            "batch_size",
            "unfreeze_blocks",
        ],
    )

    sweep_specs = build_sweep_specs(preset, best_config_file)
    all_trials: list[Any] = []
    for sweep in sweep_specs:
        scheduler = build_scheduler(preset)
        tune_run_kwargs: dict[str, Any] = {
            "run_or_experiment": ray_trainable,
            "config": sweep["search_space"],
            "num_samples": sweep["num_samples"],
            "resources_per_trial": resources_for_preset(preset),
            "metric": "val_accuracy",
            "mode": "max",
            "progress_reporter": reporter,
            "storage_path": str(ray_results_dir),
            "name": f"ray_{preset['mode']}_{sweep['model_type']}",
            "trial_dirname_creator": lambda trial: f"trial_{trial.trial_id}",
            "verbose": 1,
        }
        if max_concurrent_trials is not None:
            tune_run_kwargs["max_concurrent_trials"] = max_concurrent_trials
        if scheduler is not None:
            tune_run_kwargs["scheduler"] = scheduler
        analysis = tune.run(**tune_run_kwargs)
        all_trials.extend(analysis.trials)

    records = [build_record_from_trial(preset, trial) for trial in all_trials]
    baseline: dict[str, Any] | None = None
    if preset["include_transformer_baseline"]:
        baseline = run_transformer_baseline(preset)
        baseline["network_name"] = build_network_name(baseline)
        baseline["network_display_name"] = build_network_display_name(baseline)
        records.append(baseline)

    best_trial = None
    results_file.parent.mkdir(parents=True, exist_ok=True)
    best_config_file.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(records).sort_values("val_accuracy", ascending=False).reset_index(drop=True)
    results_df = add_efficiency_metrics(results_df)
    best_row = results_df.iloc[0]
    best_trial = next(
        (trial for trial in all_trials if trial.trial_id == best_row["trial_id"]),
        None,
    )
    best_payload = {
        "mode": preset["mode"],
        "preset": preset,
        "sweeps": [
            {
                "model_type": sweep["model_type"],
                "num_samples": sweep["num_samples"],
                "search_options": sweep["search_options"],
                "estimated_total_configurations": estimate_total_combinations(sweep["search_options"]),
            }
            for sweep in sweep_specs
        ],
        "best_trial_id": best_row["trial_id"],
        "val_accuracy": float(best_row["val_accuracy"]),
        "val_loss": float(best_row["val_loss"]),
        "config": best_trial.config if best_trial is not None else {},
        "transformer_baseline": baseline,
    }
    results_df.to_csv(results_file, index=False)
    best_config_file.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    network_summary_df = build_network_summary(results_df)
    log_results_to_mlflow(
        preset=preset,
        results_df=results_df,
        network_summary_df=network_summary_df,
        results_file=results_file,
        best_config_file=best_config_file,
        sweep_specs=sweep_specs,
        mlflow_db_file=mlflow_db_file,
    )

    return results_df


def print_plan(preset: Preset, best_config_file: Path) -> None:
    payload = {
        "preset": preset,
        "planned_ray_samples": get_num_samples(preset),
        "resources_per_trial": resources_for_preset(preset),
        "sweeps": [
            {
                "model_type": sweep["model_type"],
                "num_samples": sweep["num_samples"],
                "search_options": sweep["search_options"],
                "estimated_total_configurations": estimate_total_combinations(sweep["search_options"]),
            }
            for sweep in build_sweep_specs(preset, best_config_file)
        ],
    }
    print(json.dumps(payload, indent=2))


def print_final_trainer_config(best_config_file: Path) -> None:
    payload = {
        "final_trainer": build_final_trainer_config(best_config_file),
        "source_best_config": str(best_config_file),
    }
    print(json.dumps(payload, indent=2))


def write_final_trainer_config(best_config_file: Path, output_file: Path) -> None:
    final_config = build_final_trainer_config(best_config_file)
    output_file.write_text(json.dumps(final_config, indent=2), encoding="utf-8")
    print(f"Saved final trainer config to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ray hypertuning assignment for chapter 4.")
    parser.add_argument("--mode", choices=sorted(PRESETS), default="extensive")
    parser.add_argument("--backend", choices=("ray", "local"), default="ray")
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--best-config-file", type=Path, default=DEFAULT_BEST_CONFIG_FILE)
    parser.add_argument("--final-config-file", type=Path, default=DEFAULT_FINAL_CONFIG_FILE)
    parser.add_argument("--mlflow-db-file", type=Path, default=DEFAULT_MLFLOW_DB_FILE)
    parser.add_argument("--ray-results-dir", type=Path, default=RAY_RESULTS_DIR)
    parser.add_argument("--cpu-per-trial", type=int)
    parser.add_argument("--ray-num-cpus", type=int)
    parser.add_argument("--gpu-per-trial", type=float)
    parser.add_argument("--max-concurrent-trials", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a single-trial, single-epoch smoke check through Ray and MLflow.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected preset and search space without launching Ray.",
    )
    parser.add_argument(
        "--show-final-config",
        action="store_true",
        help="Print one simple final training configuration based on best_config.json.",
    )
    parser.add_argument(
        "--write-final-config",
        action="store_true",
        help="Write the final training configuration to a json file and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = smoke_preset(PRESETS[args.mode]) if args.smoke else dict(PRESETS[args.mode])
    if args.cpu_per_trial is not None:
        preset["cpu_per_trial"] = args.cpu_per_trial
    if args.gpu_per_trial is not None:
        preset["gpu_per_trial"] = args.gpu_per_trial

    if args.show_final_config:
        print_final_trainer_config(args.best_config_file)
        return

    if args.write_final_config:
        write_final_trainer_config(args.best_config_file, args.final_config_file)
        return

    if args.dry_run:
        print_plan(preset, args.best_config_file)
        return

    if args.backend == "local":
        results_df = run_local_search(
            preset,
            args.results_file,
            args.best_config_file,
            mlflow_db_file=args.mlflow_db_file,
        )
    else:
        results_df = run_search(
            preset,
            args.results_file,
            args.best_config_file,
            ray_results_dir=args.ray_results_dir,
            mlflow_db_file=args.mlflow_db_file,
            max_concurrent_trials=args.max_concurrent_trials,
            ray_num_cpus=args.ray_num_cpus,
        )
    best_row = results_df.sort_values("val_accuracy", ascending=False).iloc[0]
    print(
        "Best run:",
        {
            "model_type": best_row["model_type"],
            "val_accuracy": best_row["val_accuracy"],
            "val_loss": best_row["val_loss"],
            "mode": best_row["mode"],
        },
    )


if __name__ == "__main__":
    main()
