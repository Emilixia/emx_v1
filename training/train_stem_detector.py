"""
train_stem_detector.py
----------------------
Fine-tune a YOLOv8n model on a stem-detection dataset and save the weights.

Usage
-----
    python training/train_stem_detector.py [--data stem_data.yaml] \
        [--epochs 100] [--batch 16] [--imgsz 640] \
        [--output app/models/stem_detector.pt]

Dataset
-------
Prepare a YOLO-format dataset with a single class ``stem``.
The YAML file should look like::

    path: training/data/stems
    train: images/train
    val:   images/val

    nc: 1
    names: [stem]

Image annotation can be done with Roboflow, Label Studio, or CVAT.
A minimal dataset of ~500–1 000 labelled flower images is recommended for
reasonable performance.

See: https://docs.ultralytics.com/datasets/detect/
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 stem detector")
    parser.add_argument(
        "--data",
        default="training/data/stems/stem_data.yaml",
        help="Path to YOLO dataset YAML file",
    )
    parser.add_argument("--base-model", default="yolov8n.pt", help="Base YOLO weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", default="app/models/stem_detector.pt")
    parser.add_argument("--device", default="", help="'' = auto, '0' = GPU 0, 'cpu'")
    parser.add_argument(
        "--project", default="training/runs/stem_detector", help="Training run folder"
    )
    parser.add_argument("--name", default="train", help="Run name sub-folder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required. Install with: pip install ultralytics"
        ) from exc

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}\n"
            "Create your dataset first — see training/data/prepare_dataset.py"
        )

    model = YOLO(args.base_model)

    print(f"Training YOLOv8 stem detector for {args.epochs} epochs…")
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device or None,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
    )

    # Copy best weights to app/models/
    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"Warning: best weights not found at {best_pt}")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_pt, output_path)
    print(f"\nBest weights copied to: {output_path}")

    # Optional: also export to ONNX for faster CPU inference
    print("Exporting to ONNX…")
    model_best = YOLO(str(output_path))
    onnx_path = output_path.with_suffix(".onnx")
    model_best.export(format="onnx", imgsz=args.imgsz, dynamic=True, simplify=True)
    print(f"ONNX model saved to: {onnx_path}")


if __name__ == "__main__":
    main()
