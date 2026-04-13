"""
main.py
-------
Entry point for the EMX Flower Analyser desktop application.

Usage
-----
    python -m app.main
    # or directly:
    python app/main.py
"""

from __future__ import annotations

import sys

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

from app.ui.main_window import MainWindow


def _apply_dark_palette(app: QApplication) -> None:
    """Apply a clean dark/green-accented colour palette."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 46))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Base, QColor(24, 24, 37))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(35, 35, 55))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(46, 125, 50))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Button, QColor(46, 125, 50))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 80, 80))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(67, 160, 71))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Link, QColor(100, 221, 110))
    app.setPalette(palette)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("EMX Flower Analyser")
    app.setOrganizationName("Emilixia")
    app.setApplicationVersion("1.0.0")

    _apply_dark_palette(app)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
