"""Reporte PDF: cada gráfico y tabla va con su explicabilidad («cómo se lee») y su
interpretación («qué dicen estos resultados»), igual que en pantalla."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from xml.sax.saxutils import escape

#: Orden de las fases CRISP-DM en el reporte.
SECTIONS = [
    "I. Comprensión del negocio",
    "II. Comprensión de los datos (carga y EDA)",
    "III. Preparación de los datos",
    "IV. Modelado (entrenamiento, hiperparámetros, validación cruzada, selección)",
    "V. Evaluación (pruebas inferenciales robustas)",
    "VI. Despliegue (predicción e IA)",
]


@dataclass(eq=False)
class ReportItem:
    key: str
    section: str
    title: str
    how: str
    interpretation: str
    png: Optional[bytes] = None
    table: Optional[pd.DataFrame] = None
    llm_text: Optional[str] = None
    order: int = 0
    extra: Dict[str, str] = field(default_factory=dict)


def _styles():
    base = getSampleStyleSheet()
    ink, muted, accent = colors.HexColor("#0f172a"), colors.HexColor("#64748b"), colors.HexColor("#1e3a8a")
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=18, leading=22, textColor=ink, alignment=0),
        "sub": ParagraphStyle("s", parent=base["Normal"], fontSize=9, leading=12, textColor=muted, spaceAfter=10),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=14, leading=18, textColor=accent, spaceBefore=6),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=11, leading=14, textColor=ink, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=9, leading=12.5, textColor=ink, spaceAfter=4),
        "how": ParagraphStyle("how", parent=base["Normal"], fontSize=8.3, leading=11.2, textColor=colors.HexColor("#334155"),
                              backColor=colors.HexColor("#f1f5f9"), borderPadding=5, spaceBefore=4, spaceAfter=6),
        "interp": ParagraphStyle("in", parent=base["Normal"], fontSize=8.8, leading=12, textColor=colors.HexColor("#14532d"),
                                 backColor=colors.HexColor("#f0fdf4"), borderPadding=5, spaceBefore=4, spaceAfter=6),
        "llm": ParagraphStyle("llm", parent=base["Normal"], fontSize=8.8, leading=12, textColor=colors.HexColor("#4c1d95"),
                              backColor=colors.HexColor("#f5f3ff"), borderPadding=5, spaceBefore=4, spaceAfter=6),
        "cell": ParagraphStyle("c", parent=base["Normal"], fontSize=6.6, leading=8, textColor=ink),
        "head": ParagraphStyle("hc", parent=base["Normal"], fontSize=6.6, leading=8, textColor=colors.white,
                               fontName="Helvetica-Bold"),
    }


def _fmt(v) -> str:
    if isinstance(v, float):
        if v != v:  # NaN
            return "—"
        return f"{v:.4g}" if abs(v) < 1e4 else f"{v:,.0f}"
    return str(v)


def _tables(df: pd.DataFrame, st, width: float, max_cols: int = 8, max_rows: int = 45) -> List[Table]:
    """Parte tablas anchas en bloques de columnas, repitiendo la primera como llave."""
    df = df.head(max_rows)
    cols = list(df.columns)
    key, rest = cols[0], cols[1:]
    chunks = [rest[i : i + max_cols - 1] for i in range(0, len(rest), max_cols - 1)] or [[]]
    out = []
    for chunk in chunks:
        use = [key] + chunk
        data = [[Paragraph(escape(str(c)), st["head"]) for c in use]]
        for _, row in df[use].iterrows():
            data.append([Paragraph(escape(_fmt(row[c])), st["cell"]) for c in use])
        first = min(width * 0.3, 4.2 * cm) if len(use) > 2 else width / 2
        other = (width - first) / max(len(use) - 1, 1)
        t = Table(data, colWidths=[first] + [other] * (len(use) - 1), repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        out.append(t)
    return out


def _image(png: bytes, width: float) -> Image:
    img = Image(io.BytesIO(png))
    ratio = img.imageHeight / img.imageWidth
    w = min(width, 16.5 * cm)
    h = w * ratio
    if h > 17 * cm:
        h, w = 17 * cm, 17 * cm / ratio
    img.drawWidth, img.drawHeight = w, h
    return img


def build_pdf(
    items: List[ReportItem],
    meta: Dict[str, str],
    executive: Optional[Dict[str, List[str] | str]] = None,
) -> bytes:
    st = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.4 * cm, bottomMargin=1.4 * cm,
                            title="Reporte CRISP-DM · SDOH Digital Twin")
    width = A4[0] - 3.2 * cm
    story: list = [
        Paragraph("Reporte analítico CRISP-DM", st["title"]),
        Paragraph(escape(
            f"SDOH Digital Twin · Motor Python/Streamlit · Emitido {datetime.now():%d/%m/%Y %H:%M}"
            + "".join(f" · {k}: {v}" for k, v in meta.items())
        ), st["sub"]),
        Paragraph(
            "Cada gráfico y tabla se acompaña de dos bloques: <b>Cómo se lee</b> (explicabilidad: qué "
            "método se usó y cómo interpretar la figura) e <b>Interpretación</b> (lectura de los "
            "resultados obtenidos con estos datos). Cuando se generó, se añade una <b>interpretación "
            "asistida por IA</b> (LangChain/LangFlow), que debe contrastarse con los números.",
            st["body"],
        ),
    ]
    if executive:
        story.append(Paragraph("Resumen ejecutivo (LangChain)", st["h1"]))
        if executive.get("panorama"):
            story.append(Paragraph(escape(str(executive["panorama"])), st["body"]))
        for label, key in (("Hallazgos", "hallazgos"), ("Riesgos y limitaciones", "limitaciones"),
                           ("Recomendaciones", "recomendaciones")):
            if executive.get(key):
                story.append(Paragraph(f"<b>{label}</b>", st["body"]))
                for x in executive[key]:
                    story.append(Paragraph("• " + escape(str(x)), st["body"]))

    by_section: Dict[str, List[ReportItem]] = {}
    for it in items:
        by_section.setdefault(it.section, []).append(it)
    n = 0
    for sec in SECTIONS + [s for s in by_section if s not in SECTIONS]:
        its = sorted(by_section.get(sec, []), key=lambda i: i.order)
        if not its:
            continue
        story += [PageBreak(), Paragraph(escape(sec), st["h1"])]
        for it in its:
            n += 1
            block = [Paragraph(f"{n}. {escape(it.title)}", st["h2"])]
            if it.png:
                block.append(_image(it.png, width))
            story.append(KeepTogether(block))
            if it.table is not None and not it.table.empty:
                for t in _tables(it.table, st, width):
                    story += [t, Spacer(1, 4)]
                if len(it.table) > 45:
                    story.append(Paragraph(f"(Se muestran 45 de {len(it.table)} filas.)", st["sub"]))
            story.append(Paragraph("<b>Cómo se lee.</b> " + escape(it.how), st["how"]))
            story.append(Paragraph("<b>Interpretación.</b> " + escape(it.interpretation), st["interp"]))
            if it.llm_text:
                story.append(Paragraph("<b>Interpretación asistida por IA.</b> "
                                       + escape(it.llm_text).replace("\n", "<br/>"), st["llm"]))
    if n == 0:
        story.append(Paragraph("Aún no hay resultados: recorra las fases de la app antes de generar el reporte.", st["body"]))
    doc.build(story)
    return buf.getvalue()


def save_and_register(pdf: bytes, user_id: Optional[int], title: str) -> str:
    """Guarda el PDF en generated_reports y lo registra para que aparezca en la app web."""
    from app.core.database import SessionLocal
    from app.models.report import Report
    from app.services.report_service import REPORT_DIR

    filename = f"CRISPDM_Streamlit_{datetime.now():%Y%m%d_%H%M%S}.pdf"
    path = os.path.join(REPORT_DIR, filename)
    with open(path, "wb") as fh:
        fh.write(pdf)
    if user_id:
        db = SessionLocal()
        try:
            db.add(Report(title=title, report_type="crispdm_streamlit", format="pdf", status="generated",
                          filename=filename, file_path=path, query_params="origen=streamlit",
                          created_by=user_id))
            db.commit()
        finally:
            db.close()
    return path
