"""
result_panel.py
---------------
PyQt6 widget that displays flower-identification and stem-count results.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# ── Helper widgets ─────────────────────────────────────────────────────────────


class _SectionHeader(QLabel):
    def __init__(self, text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        font = QFont()
        font.setBold(True)
        font.setPointSize(10)
        self.setFont(font)
        self.setStyleSheet("color: #2e7d32; margin-top: 6px; margin-bottom: 2px;")


class _Divider(QFrame):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.setStyleSheet("color: #c8e6c9;")


class _ConfidenceRow(QWidget):
    """One row: species name  ░░░░░░░░░░  XX.X%"""

    def __init__(
        self,
        species: str,
        confidence: float,
        highlight: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)

        name_label = QLabel(species.title())
        name_label.setMinimumWidth(180)
        name_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        if highlight:
            name_label.setStyleSheet("font-weight: bold; color: #1b5e20;")

        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(int(confidence * 100))
        bar.setFixedWidth(110)
        bar.setFixedHeight(14)
        bar.setTextVisible(False)
        bar.setStyleSheet(
            "QProgressBar { border: 1px solid #a5d6a7; border-radius: 4px; "
            "background: #f1f8e9; }"
            "QProgressBar::chunk { background: #43a047; border-radius: 3px; }"
        )

        pct_label = QLabel(f"{confidence:.1%}")
        pct_label.setFixedWidth(48)
        pct_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if highlight:
            pct_label.setStyleSheet("font-weight: bold;")

        layout.addWidget(name_label)
        layout.addWidget(bar)
        layout.addWidget(pct_label)


# ── Main widget ────────────────────────────────────────────────────────────────


class ResultPanel(QWidget):
    """Sidebar panel showing flower ID results and stem count.

    Call :meth:`update_results` to populate with data, or
    :meth:`set_loading` while inference is running.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(300)
        self.setMaximumWidth(380)
        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(4)

        # Title
        title = QLabel("Analysis Results")
        title_font = QFont()
        title_font.setPointSize(13)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        root.addWidget(title)
        root.addWidget(_Divider())

        # ── Flower ID section ──────────────────────────────────────────────────
        root.addWidget(_SectionHeader("🌸  Flower Identification"))

        self._top_species_label = QLabel("—")
        self._top_species_label.setWordWrap(True)
        top_font = QFont()
        top_font.setPointSize(12)
        top_font.setBold(True)
        self._top_species_label.setFont(top_font)
        self._top_species_label.setStyleSheet("color: #1b5e20;")
        root.addWidget(self._top_species_label)

        self._top_conf_label = QLabel("")
        self._top_conf_label.setStyleSheet("color: #555; margin-bottom: 4px;")
        root.addWidget(self._top_conf_label)

        # Scroll area for top-5 predictions
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedHeight(155)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        self._predictions_container = QWidget()
        self._predictions_layout = QVBoxLayout(self._predictions_container)
        self._predictions_layout.setContentsMargins(0, 0, 0, 0)
        self._predictions_layout.setSpacing(2)
        self._predictions_layout.addStretch()
        scroll.setWidget(self._predictions_container)
        root.addWidget(scroll)

        root.addWidget(_Divider())

        # ── Stem count section ─────────────────────────────────────────────────
        root.addWidget(_SectionHeader("🌿  Stem Count"))

        stem_row = QHBoxLayout()
        self._stem_count_label = QLabel("—")
        stem_font = QFont()
        stem_font.setPointSize(28)
        stem_font.setBold(True)
        self._stem_count_label.setFont(stem_font)
        self._stem_count_label.setStyleSheet("color: #2e7d32;")
        self._stem_unit_label = QLabel("stems detected")
        self._stem_unit_label.setStyleSheet("color: #555; margin-top: 10px;")
        stem_row.addWidget(self._stem_count_label)
        stem_row.addWidget(self._stem_unit_label)
        stem_row.addStretch()
        root.addLayout(stem_row)

        root.addWidget(_Divider())

        # Status / loading label
        self._status_label = QLabel("Load an image to begin.")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #777; font-style: italic;")
        root.addWidget(self._status_label)

        root.addStretch()

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_loading(self, message: str = "Analysing image…") -> None:
        """Show a loading state while inference runs."""
        self._top_species_label.setText("…")
        self._top_conf_label.setText("")
        self._stem_count_label.setText("…")
        self._status_label.setText(message)
        self._clear_predictions()

    def update_results(
        self,
        top_predictions: List[Tuple[str, float]],
        stem_count: int,
    ) -> None:
        """Populate the panel with *top_predictions* and *stem_count*.

        Parameters
        ----------
        top_predictions:
            List of ``(species_name, confidence)`` tuples, highest confidence
            first.
        stem_count:
            Total number of detected stems.
        """
        self._clear_predictions()

        if top_predictions:
            best_species, best_conf = top_predictions[0]
            self._top_species_label.setText(best_species.title())
            self._top_conf_label.setText(f"Confidence: {best_conf:.1%}")

            for i, (species, conf) in enumerate(top_predictions):
                row = _ConfidenceRow(species, conf, highlight=(i == 0))
                # Insert before the trailing stretch
                self._predictions_layout.insertWidget(
                    self._predictions_layout.count() - 1, row
                )
        else:
            self._top_species_label.setText("Unknown")
            self._top_conf_label.setText("")

        self._stem_count_label.setText(str(stem_count))
        self._status_label.setText("Analysis complete.")

    def clear(self) -> None:
        """Reset panel to its initial state."""
        self._top_species_label.setText("—")
        self._top_conf_label.setText("")
        self._stem_count_label.setText("—")
        self._status_label.setText("Load an image to begin.")
        self._clear_predictions()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _clear_predictions(self) -> None:
        layout = self._predictions_layout
        while layout.count() > 1:  # keep the trailing stretch
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
