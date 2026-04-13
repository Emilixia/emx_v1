"""
train_flower_classifier.py
--------------------------
Fine-tune an EfficientNet-B0 classifier on the Oxford 102 Flowers dataset and
export the trained model to ONNX.

Usage
-----
    python training/train_flower_classifier.py [--epochs 30] [--batch 32] \
        [--data-dir ./training/data/flowers102] \
        [--output app/models/flower_id.onnx]

Dataset
-------
The Oxford 102 Flowers dataset is available from:
    https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

Directory structure expected inside --data-dir::
    flowers102/
        train/
            <class_id>/  ...jpg files...
        val/
            <class_id>/  ...jpg files...
        test/
            <class_id>/  ...jpg files...

You can reorganise the original dataset into this layout using
``training/data/prepare_dataset.py``.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B0_Weights

NUM_CLASSES = 102
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── Data transforms ────────────────────────────────────────────────────────────


def get_transforms() -> dict[str, transforms.Compose]:
    return {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        ),
    }


# ── Model ──────────────────────────────────────────────────────────────────────


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Load EfficientNet-B0 with pretrained ImageNet weights, replace head."""
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# ── Training loop ──────────────────────────────────────────────────────────────


def train(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}  {'─' * 40}")
        for phase in ("train", "val"):
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            running_correct = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                preds = outputs.argmax(dim=1)
                running_loss += loss.item() * inputs.size(0)
                running_correct += (preds == labels).sum().item()

            n = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / n
            epoch_acc = running_correct / n
            print(f"  {phase:5s}  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())

        scheduler.step()

    print(f"\nBest val accuracy: {best_acc:.4f}")
    model.load_state_dict(best_weights)
    return model


# ── ONNX export ────────────────────────────────────────────────────────────────


def export_onnx(model: nn.Module, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Model exported to: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train flower classifier")
    parser.add_argument("--data-dir", default="training/data/flowers102")
    parser.add_argument("--output", default="app/models/flower_id.onnx")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    tfms = get_transforms()

    datasets_ = {
        split: datasets.ImageFolder(str(data_dir / split), transform=tfms[split])
        for split in ("train", "val")
    }
    dataloaders = {
        split: DataLoader(
            ds,
            batch_size=args.batch,
            shuffle=(split == "train"),
            num_workers=4,
            pin_memory=True,
        )
        for split, ds in datasets_.items()
    }

    model = build_model().to(device)
    model = train(model, dataloaders, device, args.epochs, args.lr)
    export_onnx(model.cpu(), Path(args.output))


if __name__ == "__main__":
    main()
