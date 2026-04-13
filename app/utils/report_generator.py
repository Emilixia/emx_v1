"""
report_generator.py
-------------------
Export analysis results (flower identification + stem count) to CSV or PDF.
"""

from __future__ import annotations

import csv
import datetime
from pathlib import Path
from typing import List, Optional

# PDF generation via reportlab (optional — gracefully degrades to CSV-only if
# reportlab is not installed).
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    _REPORTLAB_AVAILABLE = True
except ImportError:
    _REPORTLAB_AVAILABLE = False


# ── Data model ─────────────────────────────────────────────────────────────────


class AnalysisResult:
    """Container for a single image's analysis output."""

    def __init__(
        self,
        image_path: str | Path,
        flower_species: str,
        confidence: float,
        stem_count: int,
        top_predictions: Optional[List[tuple[str, float]]] = None,
    ) -> None:
        self.image_path = Path(image_path)
        self.flower_species = flower_species
        self.confidence = confidence
        self.stem_count = stem_count
        # [(species, confidence), ...]
        self.top_predictions: List[tuple[str, float]] = top_predictions or []
        self.timestamp = datetime.datetime.now()

    def as_dict(self) -> dict:
        return {
            "image": self.image_path.name,
            "flower_species": self.flower_species,
            "confidence": f"{self.confidence:.4f}",
            "stem_count": self.stem_count,
            "timestamp": self.timestamp.isoformat(timespec="seconds"),
        }


# ── CSV export ─────────────────────────────────────────────────────────────────


def export_csv(results: List[AnalysisResult], output_path: str | Path) -> Path:
    """Write *results* to a CSV file at *output_path*.

    Returns the resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["image", "flower_species", "confidence", "stem_count", "timestamp"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.as_dict())

    return output_path


# ── PDF export ─────────────────────────────────────────────────────────────────


def export_pdf(results: List[AnalysisResult], output_path: str | Path) -> Path:
    """Write *results* to a PDF report at *output_path*.

    Raises
    ------
    RuntimeError
        If ``reportlab`` is not installed.
    """
    if not _REPORTLAB_AVAILABLE:
        raise RuntimeError(
            "reportlab is required for PDF export. "
            "Install it with: pip install reportlab"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story = [
        Paragraph("Flower Analysis Report", styles["Title"]),
        Paragraph(f"Generated: {generated}", styles["Normal"]),
        Spacer(1, 0.5 * cm),
    ]

    # Summary table
    header = ["Image", "Species", "Confidence", "Stems", "Timestamp"]
    table_data = [header] + [
        [
            r.image_path.name,
            r.flower_species,
            f"{r.confidence:.2%}",
            str(r.stem_count),
            r.timestamp.strftime("%Y-%m-%d %H:%M"),
        ]
        for r in results
    ]

    col_widths = [5 * cm, 5 * cm, 3 * cm, 2 * cm, 4 * cm]
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4CAF50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)

    doc.build(story)
    return output_path
