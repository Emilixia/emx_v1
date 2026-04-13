"""
main_window.py
--------------
Main PyQt6 application window for the Flower Analysis app.

Features
--------
- Drag-and-drop OR file-browser image loading
- Live preview with stem-detection overlays
- Sidebar results panel (flower ID + stem count)
- Batch processing (folder → CSV export)
- Parallel inference (classifier + detector run in background threads)
"""

from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import (
    Q_ARG,
    QMetaObject,
    QSize,
    Qt,
    QThread,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QAction,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QIcon,
    QImage,
    QKeySequence,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app.ui.result_panel import ResultPanel
from app.utils.image_loader import collect_images, load_pil
from app.utils.report_generator import AnalysisResult, export_csv

# Lazy imports so the window opens even when models are missing
_classifier: object = None
_stem_counter: object = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        from app.core.flower_classifier import FlowerClassifier
        _classifier = FlowerClassifier()
    return _classifier


def _get_stem_counter():
    global _stem_counter
    if _stem_counter is None:
        from app.core.stem_counter import StemCounter
        _stem_counter = StemCounter()
    return _stem_counter


# ── Worker thread ─────────────────────────────────────────────────────────────


class _AnalysisWorker(QThread):
    """Runs flower-ID and stem-count in background threads, then emits results."""

    finished = pyqtSignal(list, int, object)  # predictions, stem_count, annotated_bgr
    error = pyqtSignal(str)

    def __init__(self, image_path: Path, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.image_path = image_path

    def run(self) -> None:
        try:
            pil_image = load_pil(self.image_path)
            clf = _get_classifier()
            ctr = _get_stem_counter()

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                future_clf = ex.submit(clf.predict, pil_image, 5)
                future_ctr = ex.submit(ctr.count, pil_image)
                predictions = future_clf.result()
                stem_result = future_ctr.result()
            annotated = ctr.draw_detections(pil_image, stem_result)

            self.finished.emit(predictions, stem_result.count, annotated)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Image preview widget ───────────────────────────────────────────────────────


class _ImagePreview(QLabel):
    """Displays an image scaled to fit, centred."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 400)
        self.setStyleSheet(
            "background: #1a1a2e; border-radius: 8px; color: #888;"
        )
        self.setText("Drop an image here\nor click  Open Image")
        self.setWordWrap(True)
        self._pixmap_original: Optional[QPixmap] = None

    def set_pixmap_from_bgr(self, bgr_array) -> None:
        """Display a NumPy BGR array."""
        import cv2
        rgb = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._pixmap_original = QPixmap.fromImage(qimg)
        self._refresh()

    def set_pixmap_from_path(self, path: Path) -> None:
        self._pixmap_original = QPixmap(str(path))
        self._refresh()

    def resizeEvent(self, event) -> None:
        self._refresh()
        super().resizeEvent(event)

    def _refresh(self) -> None:
        if self._pixmap_original and not self._pixmap_original.isNull():
            scaled = self._pixmap_original.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)


# ── Drop-zone overlay ──────────────────────────────────────────────────────────


class _DropZone(QWidget):
    """Transparent overlay that accepts drag-and-drop image files."""

    image_dropped = pyqtSignal(Path)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_file():
                self.image_dropped.emit(path)
                break


# ── Main Window ────────────────────────────────────────────────────────────────


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EMX – Flower Analyser")
        self.setMinimumSize(950, 650)
        self.resize(1100, 720)
        self.setAcceptDrops(True)

        self._current_image_path: Optional[Path] = None
        self._worker: Optional[_AnalysisWorker] = None
        self._batch_results: List[AnalysisResult] = []

        self._build_ui()
        self._build_menu()
        self._build_toolbar()

    # ── Build UI ───────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ── Left: image preview + drop zone ───────────────────────────────────
        preview_frame = QWidget()
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(6)

        self._preview = _ImagePreview()
        self._drop_zone = _DropZone(self._preview)
        self._drop_zone.setGeometry(self._preview.rect())
        self._drop_zone.image_dropped.connect(self._load_image)

        preview_layout.addWidget(self._preview)

        btn_row = QHBoxLayout()
        self._open_btn = QPushButton("📂  Open Image")
        self._open_btn.clicked.connect(self._browse_image)
        self._open_btn.setFixedHeight(36)

        self._batch_btn = QPushButton("🗂  Batch Folder…")
        self._batch_btn.clicked.connect(self._batch_process)
        self._batch_btn.setFixedHeight(36)

        btn_row.addWidget(self._open_btn)
        btn_row.addWidget(self._batch_btn)
        preview_layout.addLayout(btn_row)

        main_layout.addWidget(preview_frame, stretch=3)

        # ── Right: results panel ───────────────────────────────────────────────
        self._result_panel = ResultPanel()
        main_layout.addWidget(self._result_panel, stretch=1)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready — open or drop an image to analyse.")

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        open_act = QAction("&Open Image…", self)
        open_act.setShortcut(QKeySequence("Ctrl+O"))
        open_act.triggered.connect(self._browse_image)
        file_menu.addAction(open_act)

        batch_act = QAction("&Batch Process Folder…", self)
        batch_act.setShortcut(QKeySequence("Ctrl+Shift+O"))
        batch_act.triggered.connect(self._batch_process)
        file_menu.addAction(batch_act)

        file_menu.addSeparator()

        export_csv_act = QAction("Export Results to &CSV…", self)
        export_csv_act.setShortcut(QKeySequence("Ctrl+S"))
        export_csv_act.triggered.connect(self._export_csv)
        file_menu.addAction(export_csv_act)

        file_menu.addSeparator()

        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence("Ctrl+Q"))
        quit_act.triggered.connect(QApplication.quit)
        file_menu.addAction(quit_act)

        help_menu = menubar.addMenu("&Help")
        about_act = QAction("&About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(22, 22))
        self.addToolBar(toolbar)

        open_act = QAction("Open", self)
        open_act.triggered.connect(self._browse_image)
        toolbar.addAction(open_act)

        batch_act = QAction("Batch", self)
        batch_act.triggered.connect(self._batch_process)
        toolbar.addAction(batch_act)

        toolbar.addSeparator()

        export_act = QAction("Export CSV", self)
        export_act.triggered.connect(self._export_csv)
        toolbar.addAction(export_act)

    # ── Image loading ──────────────────────────────────────────────────────────

    def _browse_image(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            str(Path.home()),
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.webp)",
        )
        if path_str:
            self._load_image(Path(path_str))

    def _load_image(self, path: Path) -> None:
        self._current_image_path = path
        self._preview.set_pixmap_from_path(path)
        self._result_panel.set_loading()
        self._status_bar.showMessage(f"Analysing: {path.name} …")
        self._run_analysis(path)

    # ── Inference ──────────────────────────────────────────────────────────────

    def _run_analysis(self, path: Path) -> None:
        if self._worker and self._worker.isRunning():
            return

        self._worker = _AnalysisWorker(path, parent=self)
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.start()

    def _on_analysis_done(
        self,
        predictions: list,
        stem_count: int,
        annotated_bgr,
    ) -> None:
        self._preview.set_pixmap_from_bgr(annotated_bgr)
        self._result_panel.update_results(predictions, stem_count)

        if self._current_image_path and predictions:
            species, conf = predictions[0]
            result = AnalysisResult(
                image_path=self._current_image_path,
                flower_species=species,
                confidence=conf,
                stem_count=stem_count,
                top_predictions=predictions,
            )
            # Avoid duplicates in batch results
            existing = {str(r.image_path) for r in self._batch_results}
            if str(self._current_image_path) not in existing:
                self._batch_results.append(result)

        self._status_bar.showMessage(
            f"Done — {predictions[0][0].title() if predictions else 'unknown'}"
            f" | {stem_count} stem(s)"
        )

    def _on_analysis_error(self, message: str) -> None:
        self._result_panel.clear()
        self._status_bar.showMessage(f"Error: {message}")
        QMessageBox.critical(self, "Analysis Error", message)

    # ── Batch processing ───────────────────────────────────────────────────────

    def _batch_process(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select Image Folder", str(Path.home())
        )
        if not folder:
            return

        images = collect_images(folder)
        if not images:
            QMessageBox.information(self, "Batch", "No supported images found in that folder.")
            return

        progress = QProgressDialog(
            "Processing images…", "Cancel", 0, len(images), self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        batch_results: List[AnalysisResult] = []

        for i, img_path in enumerate(images):
            if progress.wasCanceled():
                break
            progress.setValue(i)
            progress.setLabelText(f"Processing {img_path.name} ({i+1}/{len(images)})")
            QApplication.processEvents()

            try:
                pil_image = load_pil(img_path)
                clf = _get_classifier()
                ctr = _get_stem_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                    future_clf = ex.submit(clf.predict, pil_image, 5)
                    future_ctr = ex.submit(ctr.count, pil_image)
                    predictions = future_clf.result()
                    stem_result = future_ctr.result()
                if predictions:
                    species, conf = predictions[0]
                    batch_results.append(
                        AnalysisResult(
                            image_path=img_path,
                            flower_species=species,
                            confidence=conf,
                            stem_count=stem_result.count,
                            top_predictions=predictions,
                        )
                    )
            except Exception as exc:
                self._status_bar.showMessage(f"Skipped {img_path.name}: {exc}")

        progress.setValue(len(images))
        self._batch_results.extend(batch_results)

        if batch_results:
            QMessageBox.information(
                self,
                "Batch Complete",
                f"Processed {len(batch_results)} image(s).\n"
                "Use File → Export Results to CSV to save.",
            )

    # ── Export ─────────────────────────────────────────────────────────────────

    def _export_csv(self) -> None:
        if not self._batch_results:
            QMessageBox.information(
                self, "Export", "No results to export yet. Analyse some images first."
            )
            return

        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Results",
            str(Path.home() / "flower_analysis.csv"),
            "CSV Files (*.csv)",
        )
        if not path_str:
            return

        try:
            out = export_csv(self._batch_results, path_str)
            QMessageBox.information(self, "Exported", f"Results saved to:\n{out}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    # ── About ──────────────────────────────────────────────────────────────────

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About EMX Flower Analyser",
            "<h3>EMX Flower Analyser v1.0</h3>"
            "<p>AI-powered flower identification and stem counting.</p>"
            "<ul>"
            "<li><b>Flower ID:</b> EfficientNet-B0 (ONNX) — Oxford 102 Flowers</li>"
            "<li><b>Stem count:</b> YOLOv8n fine-tuned on stem imagery</li>"
            "</ul>"
            "<p>Drop an image or open a folder for batch analysis.</p>",
        )

    # ── Drag-and-drop at window level ─────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_file():
                self._load_image(path)
                break
