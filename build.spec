# -*- mode: python ; coding: utf-8 -*-
"""
build.spec
----------
PyInstaller spec file for the EMX Flower Analyser.

Build on Windows
----------------
    pip install pyinstaller
    pyinstaller build.spec

Output: dist/EMX_FlowerAnalyser/EMX_FlowerAnalyser.exe  (one-folder)
         dist/EMX_FlowerAnalyser.exe                      (one-file, slower start)

Notes
-----
- Model files (flower_id.onnx, stem_detector.pt) must be present in app/models/
  before building.
- onnxruntime and ultralytics hidden imports are listed explicitly so PyInstaller
  does not miss them.
"""

import sys
from pathlib import Path

block_cipher = None

# ── Collect model files ────────────────────────────────────────────────────────
models_dir = Path("app/models")
datas = []
for model_file in models_dir.glob("*"):
    if model_file.is_file() and model_file.suffix in (".onnx", ".pt", ".yaml"):
        datas.append((str(model_file), "app/models"))

# ── Analysis ───────────────────────────────────────────────────────────────────
a = Analysis(
    ["app/main.py"],
    pathex=[str(Path(".").resolve())],
    binaries=[],
    datas=datas,
    hiddenimports=[
        # ONNX Runtime
        "onnxruntime",
        "onnxruntime.capi",
        "onnxruntime.capi.onnxruntime_pybind11_state",
        # Ultralytics / YOLOv8
        "ultralytics",
        "ultralytics.nn.tasks",
        "ultralytics.utils",
        "ultralytics.utils.ops",
        "ultralytics.models.yolo.detect",
        # OpenCV
        "cv2",
        # PyQt6
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        # PIL / Pillow
        "PIL",
        "PIL.Image",
        "PIL.ImageOps",
        # reportlab
        "reportlab",
        "reportlab.platypus",
        "reportlab.lib.pagesizes",
        # pandas
        "pandas",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython", "jupyter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ── One-folder build (recommended for faster start-up) ────────────────────────
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EMX_FlowerAnalyser",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,       # no console window
    icon=None,           # set to "app/resources/icon.ico" if available
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="EMX_FlowerAnalyser",
)
