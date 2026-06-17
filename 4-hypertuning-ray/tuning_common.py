from __future__ import annotations

import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch
from filelock import FileLock

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
HYMENOPTERA_ROOT = DATA_ROOT / "hymenoptera_data"
TORCH_HOME = ROOT / ".torch-cache"

ANTS_BEES_URL = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_local_torch_cache() -> None:
    os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))
    TORCH_HOME.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_hymenoptera_data() -> Path:
    from torchvision.datasets.utils import download_and_extract_archive

    HYMENOPTERA_ROOT.parent.mkdir(parents=True, exist_ok=True)
    lock_path = DATA_ROOT / ".download.lock"
    with FileLock(str(lock_path)):
        train_dir = HYMENOPTERA_ROOT / "train"
        val_dir = HYMENOPTERA_ROOT / "val"
        if train_dir.exists() and val_dir.exists():
            return HYMENOPTERA_ROOT
        download_and_extract_archive(
            url=ANTS_BEES_URL,
            download_root=str(DATA_ROOT),
            filename="hymenoptera_data.zip",
            remove_finished=True,
        )
    if not (train_dir.exists() and val_dir.exists()):
        raise FileNotFoundError(
            "The ants/bees dataset could not be prepared in the local data folder."
        )
    return HYMENOPTERA_ROOT


def build_transforms(image_size: int) -> tuple[Any, Any]:
    from torchvision import transforms

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, valid_transform


def class_balanced_subset(dataset: Any, class_limit: int) -> torch.utils.data.Subset[Any]:
    indices_by_class: dict[int, list[int]] = defaultdict(list)
    for index, (_, label) in enumerate(dataset.samples):
        indices_by_class[label].append(index)

    selected: list[int] = []
    for label in sorted(indices_by_class):
        selected.extend(indices_by_class[label][:class_limit])
    return torch.utils.data.Subset(dataset, selected)


def build_dataloaders(
    image_size: int,
    batch_size: int,
    class_limit: int | None,
    num_workers: int = 0,
) -> tuple[Any, Any, int]:
    from torchvision import datasets

    data_root = ensure_hymenoptera_data()
    train_transform, valid_transform = build_transforms(image_size)
    train_dataset = datasets.ImageFolder(data_root / "train", transform=train_transform)
    valid_dataset = datasets.ImageFolder(data_root / "val", transform=valid_transform)

    if class_limit is not None:
        train_dataset = class_balanced_subset(train_dataset, class_limit)
        valid_dataset = class_balanced_subset(valid_dataset, max(1, class_limit // 2))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    base_dataset = train_dataset.dataset if isinstance(train_dataset, torch.utils.data.Subset) else train_dataset
    return train_loader, valid_loader, len(base_dataset.classes)


def make_activation(name: str) -> torch.nn.Module:
    if name == "relu":
        return torch.nn.ReLU(inplace=True)
    if name == "gelu":
        return torch.nn.GELU()
    if name == "silu":
        return torch.nn.SiLU(inplace=True)
    raise ValueError(f"Unknown activation: {name}")


def build_mlp_head(
    in_features: int,
    num_classes: int,
    head_width: int,
    dropout: float,
    head_layers: int,
    activation: str,
    use_batch_norm: bool,
) -> torch.nn.Sequential:
    if head_layers < 1:
        raise ValueError("head_layers must be at least 1.")

    layers: list[torch.nn.Module] = []
    current_features = in_features
    for _ in range(head_layers):
        layers.append(torch.nn.Linear(current_features, head_width))
        if use_batch_norm:
            layers.append(torch.nn.BatchNorm1d(head_width))
        layers.append(make_activation(activation))
        layers.append(torch.nn.Dropout(dropout))
        current_features = head_width

    layers.append(torch.nn.Linear(current_features, num_classes))
    return torch.nn.Sequential(*layers)


class ResNet18Transfer(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        head_width: int,
        dropout: float,
        unfreeze_blocks: int,
        head_layers: int = 1,
        activation: str = "relu",
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = build_mlp_head(
            in_features=in_features,
            num_classes=num_classes,
            head_width=head_width,
            dropout=dropout,
            head_layers=head_layers,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for parameter in self.backbone.fc.parameters():
            parameter.requires_grad = True

        if unfreeze_blocks >= 1:
            for parameter in self.backbone.layer4.parameters():
                parameter.requires_grad = True
        if unfreeze_blocks >= 2:
            for parameter in self.backbone.layer3.parameters():
                parameter.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class TinyTransformerClassifier(torch.nn.Module):
    def __init__(
        self,
        image_size: int,
        num_classes: int,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        head_width: int = 128,
        patch_size: int = 16,
        head_layers: int = 1,
        activation: str = "relu",
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size for the transformer baseline.")
        patches_per_side = image_size // patch_size
        num_patches = patches_per_side * patches_per_side

        self.patch_embed = torch.nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = torch.nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = build_mlp_head(
            in_features=embed_dim,
            num_classes=num_classes,
            head_width=head_width,
            dropout=dropout,
            head_layers=head_layers,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.positional_embedding[:, : x.size(1)]
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)


def build_resnet_model(num_classes: int, config: dict[str, Any]) -> torch.nn.Module:
    return ResNet18Transfer(
        num_classes=num_classes,
        head_width=int(config["head_width"]),
        dropout=float(config["dropout"]),
        unfreeze_blocks=int(config["unfreeze_blocks"]),
        head_layers=int(config.get("head_layers", 1)),
        activation=str(config.get("activation", "relu")),
        use_batch_norm=bool(int(config.get("head_batch_norm", 0))),
    )


def build_transformer_model(num_classes: int, config: dict[str, Any]) -> torch.nn.Module:
    return TinyTransformerClassifier(
        image_size=int(config["image_size"]),
        num_classes=num_classes,
        embed_dim=int(config.get("embed_dim", 64)),
        depth=int(config.get("depth", 2)),
        num_heads=int(config.get("num_heads", 4)),
        dropout=float(config["dropout"]),
        head_width=int(config["head_width"]),
        patch_size=int(config.get("patch_size", 16)),
        head_layers=int(config.get("head_layers", 1)),
        activation=str(config.get("activation", "relu")),
        use_batch_norm=bool(int(config.get("head_batch_norm", 0))),
    )


def build_model(num_classes: int, config: dict[str, Any]) -> torch.nn.Module:
    model_type = str(config.get("model_type", "resnet18"))
    if model_type == "resnet18":
        return build_resnet_model(num_classes=num_classes, config=config)
    if model_type == "transformer":
        return build_transformer_model(num_classes=num_classes, config=config)
    raise ValueError(f"Unknown model_type: {model_type}")


def make_optimizer(
    name: str,
    parameters: Any,
    lr: float,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def run_epoch(
    model: torch.nn.Module,
    loader: Any,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_items += batch_size

    average_loss = total_loss / max(total_items, 1)
    accuracy = total_correct / max(total_items, 1)
    return average_loss, accuracy


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def model_size_kb(model: torch.nn.Module) -> float:
    fd, filename = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(model.state_dict(), filename)
        size = Path(filename).stat().st_size / 1024
    finally:
        path = Path(filename)
        if path.exists():
            path.unlink()
    return size


def inference_time_ms(
    model: torch.nn.Module,
    sample: torch.Tensor,
    repeats: int = 30,
    warmup: int = 10,
) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample)
        if sample.device.type == "cuda":
            torch.cuda.synchronize(sample.device)
        start = perf_counter()
        for _ in range(repeats):
            _ = model(sample)
        if sample.device.type == "cuda":
            torch.cuda.synchronize(sample.device)
        end = perf_counter()
    return ((end - start) / repeats) * 1000


def is_memory_pressure_error(error: BaseException) -> bool:
    if isinstance(error, MemoryError):
        return True
    if isinstance(error, RuntimeError):
        message = str(error).lower()
        return "not enough memory" in message or "defaultcpuallocator" in message
    return False


def collect_efficiency_metrics(
    model: torch.nn.Module,
    valid_loader: Any,
    measurement_batch_size: int = 1,
) -> dict[str, float | int]:
    sample_inputs, _ = next(iter(valid_loader))
    sample_inputs = sample_inputs[:measurement_batch_size].to("cpu")
    model = model.to("cpu")
    param_count = parameter_count(model)
    size_kb = model_size_kb(model)
    try:
        inference_ms_per_batch = inference_time_ms(model, sample_inputs, repeats=5, warmup=1)
        inference_ms_per_image = inference_ms_per_batch / max(sample_inputs.size(0), 1)
    except (MemoryError, RuntimeError) as error:
        if not is_memory_pressure_error(error):
            raise
        inference_ms_per_batch = float("nan")
        inference_ms_per_image = float("nan")
    return {
        "param_count": param_count,
        "model_size_kb": size_kb,
        "inference_ms_per_batch": inference_ms_per_batch,
        "inference_ms_per_image": inference_ms_per_image,
    }


def fit_model(
    model: torch.nn.Module,
    train_loader: Any,
    valid_loader: Any,
    device: torch.device,
    epochs: int,
    optimizer_name: str,
    lr: float,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.0,
    report: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = make_optimizer(optimizer_name, model.parameters(), lr, weight_decay)
    model.to(device)

    best_state: dict[str, Any] | None = None
    best_metrics = {"val_accuracy": -1.0, "val_loss": float("inf"), "epoch": 0}
    final_metrics: dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_accuracy = run_epoch(model, valid_loader, criterion, None, device)

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": valid_loss,
            "val_accuracy": valid_accuracy,
        }
        final_metrics = metrics
        if valid_accuracy > best_metrics["val_accuracy"]:
            best_metrics = metrics.copy()
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if report is not None:
            report(metrics)

    if best_state is not None:
        model.load_state_dict(best_state)

    efficiency_metrics = collect_efficiency_metrics(model, valid_loader)

    return {
        "best": best_metrics,
        "final": final_metrics,
        "model_state": best_state,
        "efficiency": efficiency_metrics,
    }
