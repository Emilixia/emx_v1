# EMX Flower Analyser

A Windows desktop application that uses AI to **identify flower species** and **count stems** in images.

---

## Features

| Feature | Details |
|---------|---------|
| 🌸 Flower identification | EfficientNet-B0 (ONNX) — 102 Oxford Flowers classes |
| 🌿 Stem counting | YOLOv8n detection with bounding-box overlay |
| 🖼 Drag-and-drop UI | PyQt6 native Windows interface |
| 🗂 Batch processing | Analyse a whole folder at once |
| 📊 CSV export | Save results for downstream analysis |
| ⚡ Parallel inference | Classifier + detector run simultaneously |
| 📦 Standalone `.exe` | No Python install needed (PyInstaller) |

---

## Repository Structure

```
emx_v1/
├── app/
│   ├── main.py                      # Entry point
│   ├── ui/
│   │   ├── main_window.py           # Main PyQt6 window
│   │   └── result_panel.py          # Results sidebar
│   ├── core/
│   │   ├── flower_classifier.py     # ONNX EfficientNet-B0 wrapper
│   │   └── stem_counter.py          # YOLOv8 stem detector wrapper
│   ├── models/
│   │   ├── flower_id.onnx           # (not in repo — see training)
│   │   ├── stem_detector.pt         # (not in repo — see training)
│   │   └── README.md                # How to obtain model weights
│   └── utils/
│       ├── image_loader.py          # Preprocessing utilities
│       └── report_generator.py      # CSV / PDF export
├── training/
│   ├── train_flower_classifier.py   # EfficientNet fine-tuning → ONNX
│   ├── train_stem_detector.py       # YOLOv8 fine-tuning
│   └── data/
│       └── prepare_dataset.py       # Dataset download / layout helpers
├── tests/
│   ├── test_classifier.py
│   └── test_stem_counter.py
├── requirements.txt
├── build.spec                       # PyInstaller spec
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Obtain model weights

See [`app/models/README.md`](app/models/README.md) for instructions on:
- Training the flower classifier from the Oxford 102 dataset
- Training the stem detector on your annotated data

### 3. Run the app

```bash
python -m app.main
# or
python app/main.py
```

### 4. Build a standalone Windows `.exe`

```bash
pyinstaller build.spec
# Output: dist/EMX_FlowerAnalyser/EMX_FlowerAnalyser.exe
```

---

## Training

### Flower classifier (EfficientNet-B0 → ONNX)

```bash
# Prepare Oxford 102 Flowers dataset
python training/data/prepare_dataset.py flowers102 \
    --raw-dir training/data/raw \
    --out-dir training/data/flowers102

# Fine-tune (GPU recommended)
python training/train_flower_classifier.py \
    --epochs 30 --batch 32 --device cuda
# → app/models/flower_id.onnx
```

### Stem detector (YOLOv8n)

```bash
# Create dataset skeleton
python training/data/prepare_dataset.py stems

# Annotate images (Roboflow / Label Studio), then train:
python training/train_stem_detector.py \
    --data training/data/stems/stem_data.yaml \
    --epochs 100 --batch 16
# → app/models/stem_detector.pt
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

The test suite uses mock ONNX sessions and a fake Ultralytics YOLO, so no real
model files or GPU are required.

---

## GPU Support

| Component | How to enable GPU |
|-----------|-------------------|
| Flower classifier (ONNX) | `pip install onnxruntime-gpu` — pass `providers=["CUDAExecutionProvider", "CPUExecutionProvider"]` to `FlowerClassifier` |
| Stem detector (YOLO) | Pass `device="0"` (GPU index) or `device="cuda"` to `StemCounter` |

---

## Supported Image Formats

`.jpg` · `.jpeg` · `.png` · `.bmp` · `.tiff` · `.tif` · `.webp`