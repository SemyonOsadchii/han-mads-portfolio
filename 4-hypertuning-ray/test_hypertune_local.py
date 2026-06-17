from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch


MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))
SPEC = importlib.util.spec_from_file_location("hypertune_under_test", MODULE_DIR / "hypertune.py")
hypertune = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(hypertune)

TUNING_SPEC = importlib.util.spec_from_file_location("tuning_common_under_test", MODULE_DIR / "tuning_common.py")
tuning_common = importlib.util.module_from_spec(TUNING_SPEC)
assert TUNING_SPEC.loader is not None
TUNING_SPEC.loader.exec_module(tuning_common)


def local_config(seed: int) -> dict[str, object]:
    return {
        "model_type": "resnet18",
        "image_size": 96,
        "epochs": 1,
        "class_limit": 2,
        "seed": seed,
        "num_workers": 0,
        "batch_size": 8,
        "lr": 1e-3,
        "dropout": 0.15,
        "head_width": 128,
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


def metrics() -> dict[str, object]:
    return {
        "best": {"val_accuracy": 0.75, "val_loss": 0.5, "epoch": 1},
        "final": {"val_accuracy": 0.75, "val_loss": 0.5, "epoch": 1},
        "efficiency": {
            "param_count": 10,
            "model_size_kb": 20.0,
            "inference_ms_per_batch": 1.0,
            "inference_ms_per_image": 0.1,
        },
    }


class LocalSearchPersistenceTest(unittest.TestCase):
    def test_completed_trial_is_saved_before_later_memory_error(self) -> None:
        preset = dict(hypertune.PRESETS["balanced"])
        preset.update(
            {
                "mode": "test",
                "description": "Test local persistence.",
                "family_samples": {"resnet18": 2},
                "max_epochs": 1,
                "include_transformer_baseline": False,
            }
        )
        calls = {"fit": 0}

        def fake_fit_model(**_: object) -> dict[str, object]:
            calls["fit"] += 1
            if calls["fit"] == 2:
                raise MemoryError("simulated dataloader failure")
            return metrics()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_file = tmp / "results.csv"
            best_config_file = tmp / "best_config.json"

            with (
                patch.object(hypertune, "set_local_torch_cache"),
                patch.object(hypertune, "seed_everything"),
                patch.object(hypertune, "detect_device", return_value=torch.device("cpu")),
                patch.object(hypertune, "build_local_config", side_effect=lambda *args: local_config(args[-1])),
                patch.object(hypertune, "build_dataloaders", return_value=("train", "valid", 2)) as build_dataloaders,
                patch.object(hypertune, "build_model", return_value=object()),
                patch.object(hypertune, "fit_model", side_effect=fake_fit_model),
            ):
                with self.assertRaises(MemoryError):
                    hypertune.run_local_search(
                        preset=preset,
                        results_file=results_file,
                        best_config_file=best_config_file,
                        mlflow_db_file=tmp / "mlflow.db",
                    )

            self.assertTrue(results_file.exists())
            self.assertTrue(best_config_file.exists())
            results = pd.read_csv(results_file)
            self.assertEqual(["local_resnet18_000"], results["trial_id"].tolist())
            self.assertEqual(2, build_dataloaders.call_count)

    def test_existing_local_trials_are_skipped_on_resume(self) -> None:
        preset = dict(hypertune.PRESETS["balanced"])
        preset.update(
            {
                "mode": "test",
                "description": "Test local resume.",
                "family_samples": {"resnet18": 2},
                "max_epochs": 1,
                "include_transformer_baseline": False,
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_file = tmp / "results.csv"
            best_config_file = tmp / "best_config.json"
            existing = hypertune.build_local_record(
                preset=preset,
                config=local_config(42),
                metrics=metrics(),
                trial_id="local_resnet18_000",
                elapsed_seconds=1.0,
            )
            pd.DataFrame([existing]).to_csv(results_file, index=False)

            with (
                patch.object(hypertune, "set_local_torch_cache"),
                patch.object(hypertune, "seed_everything"),
                patch.object(hypertune, "detect_device", return_value=torch.device("cpu")),
                patch.object(hypertune, "build_local_config", side_effect=lambda *args: local_config(args[-1])),
                patch.object(hypertune, "build_dataloaders", return_value=("train", "valid", 2)),
                patch.object(hypertune, "build_model", return_value=object()),
                patch.object(hypertune, "fit_model", return_value=metrics()) as fit_model,
                patch.object(hypertune, "log_results_to_mlflow"),
            ):
                hypertune.run_local_search(
                    preset=preset,
                    results_file=results_file,
                    best_config_file=best_config_file,
                    mlflow_db_file=tmp / "mlflow.db",
                )

            results = pd.read_csv(results_file).sort_values("trial_id")
            self.assertEqual(
                ["local_resnet18_000", "local_resnet18_001"],
                results["trial_id"].tolist(),
            )
            self.assertEqual(1, fit_model.call_count)
            existing_row = results[results["trial_id"] == "local_resnet18_000"].iloc[0]
            self.assertEqual(1.0, float(existing_row["elapsed_seconds"]))

    def test_efficiency_metrics_fall_back_when_inference_timing_runs_out_of_memory(self) -> None:
        class FailingModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

            def forward(self, sample: torch.Tensor) -> torch.Tensor:
                raise RuntimeError("DefaultCPUAllocator: not enough memory")

        valid_loader = [(torch.zeros(2, 3, 8, 8), torch.zeros(2, dtype=torch.long))]

        result = tuning_common.collect_efficiency_metrics(FailingModel(), valid_loader)

        self.assertEqual(1, result["param_count"])
        self.assertGreater(result["model_size_kb"], 0)
        self.assertTrue(pd.isna(result["inference_ms_per_batch"]))
        self.assertTrue(pd.isna(result["inference_ms_per_image"]))


if __name__ == "__main__":
    unittest.main()
