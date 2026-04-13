"""
prepare_dataset.py
------------------
Utilities for downloading and reorganising the Oxford 102 Flowers dataset
into the ImageFolder layout required by train_flower_classifier.py, and for
creating a skeleton YOLO stem-detection dataset.

Oxford 102 Flowers
------------------
Download the following files from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/:
    - 102flowers.tgz   (image archive)
    - imagelabels.mat  (per-image class labels, 1-indexed)
    - setid.mat        (train / val / test split indices)

Place them in the directory pointed to by --raw-dir (default: training/data/raw).

Usage
-----
    # Reorganise Oxford 102 Flowers:
    python training/data/prepare_dataset.py flowers102 \
        --raw-dir training/data/raw \
        --out-dir training/data/flowers102

    # Create skeleton YOLO stem dataset:
    python training/data/prepare_dataset.py stems \
        --out-dir training/data/stems
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


# ── Oxford 102 Flowers ─────────────────────────────────────────────────────────


def reorganise_flowers102(raw_dir: Path, out_dir: Path) -> None:
    """Convert Oxford 102 raw files into an ImageFolder directory tree."""
    try:
        import scipy.io as sio
    except ImportError as exc:
        raise ImportError("scipy is needed: pip install scipy") from exc

    images_dir = raw_dir / "jpg"
    labels_mat = raw_dir / "imagelabels.mat"
    setid_mat = raw_dir / "setid.mat"

    for p in (images_dir, labels_mat, setid_mat):
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    labels = sio.loadmat(str(labels_mat))["labels"][0]  # shape (8189,)
    setid = sio.loadmat(str(setid_mat))
    splits = {
        "train": set(setid["trnid"][0].tolist()),
        "val": set(setid["valid"][0].tolist()),
        "test": set(setid["tstid"][0].tolist()),
    }

    image_files = sorted(images_dir.glob("*.jpg"))
    if not image_files:
        raise FileNotFoundError(f"No JPG images found in {images_dir}")

    total = 0
    for img_path in image_files:
        # Oxford filenames: image_00001.jpg  (1-indexed)
        idx = int(img_path.stem.split("_")[1])
        class_id = int(labels[idx - 1])  # 1-indexed
        for split, id_set in splits.items():
            if idx in id_set:
                dest = out_dir / split / f"{class_id:03d}"
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy(img_path, dest / img_path.name)
                total += 1
                break

    print(f"Reorganised {total} images into {out_dir}")


# ── YOLO stem dataset skeleton ─────────────────────────────────────────────────


_STEM_YAML_TEMPLATE = """\
# Stem detection dataset
# Annotate images with bounding boxes around each individual stem/stalk.
# Tools: Roboflow (https://roboflow.com), Label Studio, or CVAT.

path: {abs_path}
train: images/train
val:   images/val

nc: 1
names: [stem]
"""


def create_stem_skeleton(out_dir: Path) -> None:
    """Create empty YOLO-format directory tree and YAML config for stem data."""
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    yaml_path = out_dir / "stem_data.yaml"
    if not yaml_path.exists():
        yaml_path.write_text(_STEM_YAML_TEMPLATE.format(abs_path=out_dir.resolve()))

    readme = out_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Stem Detection Dataset\n\n"
            "Place training images under `images/train/` and "
            "corresponding YOLO-format `.txt` label files under `labels/train/`.\n\n"
            "Each `.txt` label file contains one line per stem:\n"
            "```\n0 <cx> <cy> <w> <h>\n```\n"
            "where coordinates are normalised (0–1) relative to image size.\n\n"
            "Repeat for `images/val/` and `labels/val/`.\n"
        )

    print(f"Stem dataset skeleton created at: {out_dir}")
    print(f"YAML config: {yaml_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare training datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_flowers = subparsers.add_parser("flowers102", help="Reorganise Oxford 102 dataset")
    p_flowers.add_argument("--raw-dir", default="training/data/raw")
    p_flowers.add_argument("--out-dir", default="training/data/flowers102")

    p_stems = subparsers.add_parser("stems", help="Create YOLO stem dataset skeleton")
    p_stems.add_argument("--out-dir", default="training/data/stems")

    args = parser.parse_args()

    if args.command == "flowers102":
        reorganise_flowers102(Path(args.raw_dir), Path(args.out_dir))
    elif args.command == "stems":
        create_stem_skeleton(Path(args.out_dir))


if __name__ == "__main__":
    main()
