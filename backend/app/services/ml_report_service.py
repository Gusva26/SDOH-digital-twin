"""Reportes PDF del modelo predictivo: comparación de los 4 modelos y escenarios simulados.

Los PDF se guardan en generated_reports y se registran en la tabla `reports`, así
aparecen en la página Reportes y en la Fase VI, y se descargan por
/api/reports/download/{id} como el resto de reportes.
"""

import os
from datetime import datetime
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sqlalchemy.orm import Session  # noqa: E402

from app.models.report import Report  # noqa: E402
from app.models.user import User  # noqa: E402
from app.services import ml_service  # noqa: E402
from app.services.report_service import CHART_DIR, REPORT_DIR  # noqa: E402

LEVEL_HEX = {"low": "#10b981", "moderate": "#f59e0b", "high": "#f97316", "critical": "#ef4444"}


def _area_name(metrics: Dict[str, Any]) -> str:
    """Nombre del área de estudio a partir de los condados que la componen.

    Los cuartiles se calculan sobre el área completa, así que el reporte debe
    nombrarla tal cual en vez de asumir un único condado.
    """
    area = metrics.get("study_area") or {}
    counties = area.get("counties") or []
    if not counties:
        return "el área de estudio"
    if len(counties) == 1:
        return counties[0]["name"]
    return "el área de estudio (" + ", ".join(c["name"] for c in counties) + ")"

INK = colors.HexColor("#0f172a")
MUTED = colors.HexColor("#475569")
ACCENT = colors.HexColor("#1e40af")


def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontName="Helvetica-Bold", fontSize=17,
                                leading=21, textColor=INK, alignment=0, spaceAfter=4),
        "sub": ParagraphStyle("s", parent=base["Normal"], fontSize=9, leading=12, textColor=MUTED,
                              spaceAfter=10),
        "h2": ParagraphStyle("h", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12,
                             leading=15, textColor=ACCENT, spaceBefore=10, spaceAfter=6),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=9.5, leading=13.5, textColor=INK,
                               spaceAfter=5),
        "note": ParagraphStyle("n", parent=base["Normal"], fontName="Helvetica-Oblique", fontSize=8,
                               leading=11, textColor=MUTED, spaceBefore=6),
        "cell": ParagraphStyle("c", parent=base["Normal"], fontSize=8.5, leading=11, textColor=INK),
    }


def _table(rows: List[List[Any]], widths: List[float], highlight_row: int = -1) -> Table:
    t = Table(rows, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if highlight_row > 0:
        style += [
            ("BACKGROUND", (0, highlight_row), (-1, highlight_row), colors.HexColor("#dcfce7")),
            ("FONTNAME", (0, highlight_row), (-1, highlight_row), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(style))
    return t


def _chart_path(prefix: str) -> str:
    os.makedirs(CHART_DIR, exist_ok=True)
    return os.path.join(CHART_DIR, f"{prefix}_{datetime.now().timestamp() * 1000:.0f}.png")


def _hbar(labels: List[str], values: List[float], colors_: List[str], xlabel: str, prefix: str,
          fmt: str = "{:.3f}") -> str:
    path = _chart_path(prefix)
    fig, ax = plt.subplots(figsize=(7.2, max(1.8, 0.38 * len(labels) + 0.6)))
    y = range(len(labels))
    ax.barh(list(y), values, color=colors_, height=0.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8.5, color="#334155")
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=8.5, color="#475569")
    ax.tick_params(axis="x", labelsize=8, colors="#475569")
    ax.axvline(0, color="#94a3b8", linewidth=0.8)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for i, v in enumerate(values):
        ax.text(v, i, " " + fmt.format(v), va="center", ha="left" if v >= 0 else "right",
                fontsize=7.5, color="#1e293b")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _register(db: Session, user: User, title: str, report_type: str, filename: str, path: str,
              params: str) -> Report:
    report = Report(
        title=title,
        report_type=report_type,
        format="pdf",
        status="generated",
        filename=filename,
        file_path=path,
        query_params=params,
        created_by=user.id,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def _doc(path: str) -> SimpleDocTemplate:
    return SimpleDocTemplate(path, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                             topMargin=1.4 * cm, bottomMargin=1.4 * cm)


def _fmt_date(iso: str) -> str:
    return (iso or "").replace("T", " ").replace("Z", " UTC")


# --------------------------------------------------------------------------- #
# Reporte del entrenamiento (comparación de los 4 modelos)
# --------------------------------------------------------------------------- #

def model_report(db: Session, user: User, target: str) -> Report:
    bundle = ml_service.load_bundle(target)
    if not bundle:
        raise LookupError(f"No hay modelo entrenado para {target}.")
    m = bundle["metrics"]
    st = _styles()
    target_name = m["target"]["name"]
    filename = f"ML_Modelos_{target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    path = os.path.join(REPORT_DIR, filename)

    story: List[Any] = [
        Paragraph(f"Reporte de modelos predictivos: {target_name}", st["title"]),
        Paragraph(
            f"SDOH Digital Twin · Fuente: {m['source']} · Año {m['year']} · "
            f"Entrenado: {_fmt_date(m['trained_at'])} · Emitido: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            st["sub"],
        ),
        Paragraph("1. Resumen", st["h2"]),
    ]
    best = next(r for r in m["models"] if r["selected"])
    story.append(Paragraph(
        f"Se entrenaron y compararon <b>4 clasificadores</b> para predecir el nivel de riesgo de "
        f"<b>{m['target_info']}</b> por census tract, a partir de {len(m['features'])} determinantes "
        f"sociales. El modelo seleccionado y desplegado es <b>{m['best_model_name']}</b>, con "
        f"F1-macro de {best['cv_f1_mean']:.4f} en validación cruzada ({m['cv_folds']} folds), "
        f"F1 de {best['test_f1_macro']:.4f} y ROC-AUC de {best['test_roc_auc']:.4f} en el conjunto de prueba.",
        st["body"],
    ))
    reg = m.get("regressor")
    if reg:
        story.append(Paragraph(
            f"Como complemento, una {reg['name']} estima el porcentaje exacto con R² = {reg['r2_test']:.3f} "
            f"y error absoluto medio de {reg['mae_test']:.2f} puntos porcentuales en test.",
            st["body"],
        ))

    story.append(Paragraph("2. Datos y definición de clases", st["h2"]))
    cuts = m["class_cuts"]
    ts = m["target_stats"]
    story.append(_table(
        [
            ["Elemento", "Detalle"],
            ["Muestras", f"{m['n_samples']} census tracts (train {m['n_train']} / test {m['n_test']})"],
            ["Objetivo (y)", f"{target_name}: {m['target_info']}"],
            ["Rango observado", f"{ts['min']:.1f} % a {ts['max']:.1f} % (mediana {ts['median']:.1f} %)"],
            ["Bajo", f"< {cuts['moderate']:.1f} %"],
            ["Moderado", f"{cuts['moderate']:.1f} % a {cuts['high']:.1f} %"],
            ["Alto", f"{cuts['high']:.1f} % a {cuts['critical']:.1f} %"],
            ["Crítico", f">= {cuts['critical']:.1f} %"],
            ["Entradas (X)", Paragraph(", ".join(f["name"] for f in m["features"]), st["cell"])],
        ],
        [4 * cm, 13.6 * cm],
    ))

    story.append(Paragraph("3. Comparación de los 4 modelos", st["h2"]))
    rows = [["Modelo", "F1 CV", "F1 test", "Accuracy", "ROC-AUC", "Tiempo", "Estado"]]
    for r in m["models"]:
        rows.append([
            r["name"], f"{r['cv_f1_mean']:.4f} ± {r['cv_f1_std']:.3f}", f"{r['test_f1_macro']:.4f}",
            f"{r['test_accuracy']:.4f}", f"{r['test_roc_auc']:.4f}", f"{r['train_seconds']:.1f} s",
            "Desplegado" if r["selected"] else "Candidato",
        ])
    sel = 1 + next(i for i, r in enumerate(m["models"]) if r["selected"])
    story.append(_table(rows, [4.6 * cm, 2.9 * cm, 1.9 * cm, 2 * cm, 1.9 * cm, 1.6 * cm, 2.7 * cm], sel))
    chart = _hbar(
        [r["name"] for r in m["models"]], [r["cv_f1_mean"] for r in m["models"]],
        ["#f59e0b" if r["selected"] else "#94a3b8" for r in m["models"]],
        "F1-macro (validación cruzada)", "ml_f1",
    )
    story += [Spacer(1, 6), Image(chart, width=15 * cm, height=15 * cm * 0.3)]
    story.append(Paragraph(
        f"Criterio de selección: {m['selection_metric']}. El conjunto de prueba (20 %) no se usa para "
        "elegir; solo confirma el desempeño del modelo seleccionado.",
        st["note"],
    ))

    story.append(Paragraph(f"4. Matriz de confusión: {best['name']} (test)", st["h2"]))
    names = ["Bajo", "Moderado", "Alto", "Crítico"]
    cm_rows = [["Real / Predicho"] + names] + [
        [names[i]] + [str(v) for v in row] for i, row in enumerate(best["confusion_matrix"])
    ]
    cmt = _table(cm_rows, [3.4 * cm] + [2.6 * cm] * 4)
    cmt.setStyle(TableStyle([("ALIGN", (1, 1), (-1, -1), "CENTER")] + [
        ("BACKGROUND", (i + 1, i + 1), (i + 1, i + 1), colors.HexColor("#bfdbfe")) for i in range(4)
    ]))
    story.append(cmt)

    story.append(Paragraph("5. Importancia de variables (modelo desplegado)", st["h2"]))
    imp = m["feature_importance"]
    chart = _hbar([f["name"] for f in imp], [f["importance"] for f in imp], ["#3b82f6"] * len(imp),
                  "Caída de F1 al permutar la variable", "ml_imp")
    story.append(Image(chart, width=15 * cm, height=15 * cm * (max(1.8, 0.38 * len(imp) + 0.6) / 7.2)))

    story.append(Paragraph("6. Limitaciones", st["h2"]))
    for txt in [
        "CDC PLACES publica estimaciones de área pequeña basadas en modelos; las variables están "
        "correlacionadas y las métricas pueden ser optimistas.",
        "Análisis ecológico a nivel de census tract: no permite inferencias sobre individuos.",
        "Las predicciones son asociativas, no causales; los niveles son relativos a los cuartiles "
        "del área de estudio, calculados sobre todos sus tracts.",
    ]:
        story.append(Paragraph(f"• {txt}", st["body"]))

    _doc(path).build(story)
    return _register(db, user, f"Modelos ML · {target_name}", "ml_models", filename, path,
                     f"target={target}")


# --------------------------------------------------------------------------- #
# Reporte de un escenario simulado
# --------------------------------------------------------------------------- #

def scenario_report(db: Session, user: User, target: str, features: Dict[str, Any]) -> Report:
    p = ml_service.predict(features, target)
    bundle = ml_service.load_bundle(target)
    m = bundle["metrics"]
    st = _styles()
    target_name = m["target"]["name"]
    filename = f"ML_Escenario_{target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    path = os.path.join(REPORT_DIR, filename)
    level = p["risk_level"]

    story: List[Any] = [
        Paragraph(f"Predicción de escenario: {target_name}", st["title"]),
        Paragraph(
            f"SDOH Digital Twin · Modelo: {p['model']} · Emitido: {datetime.now().strftime('%d/%m/%Y %H:%M')} "
            f"· Generado por: {user.username}",
            st["sub"],
        ),
    ]

    clf_cell = (
        f"Voto clasificador: {p['classifier_level_name'].upper()} ({p['confidence'] * 100:.0f} %)"
        if "classifier_level_name" in p
        else f"Confianza {p['confidence'] * 100:.1f} %"
    )
    badge = Table(
        [[f"Nivel {p['level_name'].upper()}",
          f"Estimación {p['estimated_value']:.1f} %" if "estimated_value" in p else "",
          clf_cell,
          f"Percentil {p['percentile']:.0f}" if "percentile" in p else ""]],
        colWidths=[4.4 * cm] * 4,
    )
    badge.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor(LEVEL_HEX[level])),
        ("TEXTCOLOR", (0, 0), (0, 0), colors.white),
        ("BACKGROUND", (1, 0), (-1, 0), colors.HexColor("#eff6ff")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story += [badge, Spacer(1, 8)]

    story.append(Paragraph("1. Interpretación", st["h2"]))
    story.append(Paragraph(p["interpretation"], st["body"]))

    story.append(Paragraph("2. Verificación cruzada del clasificador de niveles", st["h2"]))
    story.append(Paragraph(
        "El nivel publicado arriba se deriva de la estimación puntual. El clasificador se "
        "entrena por separado como verificación: su voto y sus probabilidades se muestran aquí.",
        st["body"],
    ))
    probs = p["probabilities"]
    chart = _hbar(["Bajo", "Moderado", "Alto", "Crítico"],
                  [probs[k] * 100 for k in ["low", "moderate", "high", "critical"]],
                  [LEVEL_HEX[k] for k in ["low", "moderate", "high", "critical"]],
                  "Probabilidad (%)", "ml_prob", fmt="{:.1f} %")
    story.append(Image(chart, width=14 * cm, height=14 * cm * (1.8 / 7.2)))
    rng = p["level_range"]
    if rng.get("open_min") and rng.get("open_max"):
        band = f"banda abierta (referente: {rng['min']:.1f} % a {rng['max']:.1f} %)"
    elif rng.get("open_min"):
        band = f"hasta {rng['max']:.1f} %, sin cota inferior"
    elif rng.get("open_max"):
        band = f"desde {rng['min']:.1f} %, sin cota superior"
    else:
        band = f"{rng['min']:.1f} % a {rng['max']:.1f} %"
    story.append(Paragraph(
        f"Nivel {p['level_name']}: {band} de {m['target_info']} ({p['level_meaning']}). "
        f"El nivel lo determina la estimación puntual, de modo que el valor y su banda son "
        f"siempre coherentes.",
        st["note"],
    ))

    if p.get("drivers"):
        story.append(Paragraph("3. Factores que explican la predicción", st["h2"]))
        story.append(Paragraph(
            "Efecto de cada determinante sobre el % estimado, comparado con dejarlo en la mediana del "
            "área de estudio (puntos porcentuales). Positivo = eleva el riesgo; negativo = lo reduce.",
            st["body"],
        ))
        drv = p["drivers"]
        chart = _hbar([d["name"] for d in drv], [d["effect_pp"] for d in drv],
                      ["#ef4444" if d["effect_pp"] > 0 else "#10b981" for d in drv],
                      "Efecto sobre el % estimado (pp)", "ml_drv", fmt="{:+.2f}")
        story.append(Image(chart, width=15 * cm, height=15 * cm * (max(1.8, 0.38 * len(drv) + 0.6) / 7.2)))

    story.append(KeepTogether([
        Paragraph("4. Valores del escenario", st["h2"]),
        _table(
            [["Determinante social", "Escenario", "Mediana área", "Diferencia"]]
            + [
                [d["name"], f"{d['value']:.1f} %", f"{d['median']:.1f} %", f"{d['value'] - d['median']:+.1f}"]
                for d in (p.get("drivers") or [])
            ],
            [8 * cm, 3 * cm, 3.4 * cm, 3.2 * cm],
        ),
    ]))
    reg = m.get("regressor") or {}
    story.append(Paragraph(
        f"Clasificador: {p['model']} (F1 CV {m['models'][0]['cv_f1_mean']:.4f}). Estimación puntual: "
        f"{reg.get('name', '-')} (R² {reg.get('r2_test', 0):.3f}, MAE {reg.get('mae_test', 0):.2f} pp). "
        f"Resultados asociativos, no causales; datos CDC PLACES de {_area_name(m)}.",
        st["note"],
    ))

    _doc(path).build(story)
    return _register(db, user, f"Escenario ML · {target_name} · {p['level_name']}", "ml_scenario",
                     filename, path, f"target={target}")
