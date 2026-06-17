from __future__ import annotations

from typing import Any

from ray import tune

from tuning_common import (
    build_model,
    build_dataloaders,
    detect_device,
    fit_model,
    seed_everything,
    set_local_torch_cache,
)


def trainable(config: dict[str, Any]) -> None:
    set_local_torch_cache()
    seed_everything(int(config.get("seed", 42)))

    device = detect_device()
    class_limit = config.get("class_limit")
    num_workers = int(config.get("num_workers", 0))
    train_loader, valid_loader, num_classes = build_dataloaders(
        image_size=int(config["image_size"]),
        batch_size=int(config["batch_size"]),
        class_limit=int(class_limit) if class_limit is not None else None,
        num_workers=num_workers,
    )

    model = build_model(num_classes=num_classes, config=config)
    fit_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        epochs=int(config["epochs"]),
        optimizer_name=str(config["optimizer"]),
        lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 1e-4)),
        label_smoothing=float(config.get("label_smoothing", 0.0)),
        report=tune.report,
    )
