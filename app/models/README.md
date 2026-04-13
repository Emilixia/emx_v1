# app/models/

This directory holds the trained model weights used by the application at runtime.

## Required files

| File | Description |
|------|-------------|
| `flower_id.onnx` | EfficientNet-B0 classifier fine-tuned on Oxford 102 Flowers (102 classes) |
| `stem_detector.pt` | YOLOv8n detector fine-tuned to detect individual flower stems |

These files are **not** committed to version control due to their size.

---

## How to obtain the models

### Option A — Train from scratch

1. **Flower classifier**

   ```bash
   # 1. Download Oxford 102 Flowers raw data to training/data/raw/
   #    https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
   #      - 102flowers.tgz
   #      - imagelabels.mat
   #      - setid.mat

   # 2. Reorganise into ImageFolder layout
   python training/data/prepare_dataset.py flowers102

   # 3. Fine-tune and export to ONNX
   python training/train_flower_classifier.py --epochs 30
   # Output: app/models/flower_id.onnx
   ```

2. **Stem detector**

   ```bash
   # 1. Annotate flower images with stem bounding boxes
   #    (Roboflow / Label Studio recommended)
   #    Place data under training/data/stems/ — see prepare_dataset.py stems

   # 2. Create skeleton
   python training/data/prepare_dataset.py stems

   # 3. Fine-tune YOLOv8n
   python training/train_stem_detector.py --data training/data/stems/stem_data.yaml
   # Output: app/models/stem_detector.pt  +  stem_detector.onnx
   ```

### Option B — Use pre-trained checkpoints

If pre-trained weights are available (e.g. shared by the team), place them
directly in this directory with the exact filenames above.

---

## GPU vs CPU

- `flower_id.onnx` is run via **ONNX Runtime** (`CPUExecutionProvider` by default).
  For GPU inference install `onnxruntime-gpu` instead of `onnxruntime`.
- `stem_detector.pt` is run via **Ultralytics YOLO**, which auto-detects CUDA.
  Pass `device="0"` (or `"cuda"`) to `StemCounter` to use a GPU.
