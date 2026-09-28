import numpy as np
import streamlit as st

import ui
from engine import modeling
from engine.data import TARGET_INFO
from engine.style import INK, LEVEL_COLORS, LEVEL_ES, plt
from app.services.ml_service import DEFAULT_TARGET

st.title("🧹 Fase III · Preparación de los datos")
ds = ui.need("ds", "Primero cargue el dataset en «II. Carga del dataset».")

targets = ds.targets
c1, c2 = st.columns([2, 1])
target = c1.selectbox("Resultado de salud a predecir (y)", targets,
                      index=targets.index(DEFAULT_TARGET) if DEFAULT_TARGET in targets else 0,
                      format_func=lambda c: f"{ds.label(c)} — {TARGET_INFO.get(c, '')}")
test_size = c2.slider("Proporción de prueba", 0.1, 0.4, 0.2, 0.05)
features = st.multiselect("Determinantes sociales (X)", ds.features, default=ds.features, format_func=ds.label)

prep = st.session_state.prep
changed = prep is None or prep["target"] != target or prep["features"] != features or prep["test_size"] != test_size
if st.button("Aplicar preparación", type="primary", disabled=not features) or (prep is not None and not changed):
    if changed:
        ui.reset_from(ui.S_PREP)
        st.session_state.prep = modeling.prepare(ds.df, target, features, test_size)
else:
    st.info("Elija objetivo y variables y pulse «Aplicar preparación».")
    st.stop()

p = st.session_state.prep
tname = ds.label(p["target"])

ui.render("prep_pipeline", ui.S_PREP, "Pipeline de preparación", modeling.HOW_PREP,
          modeling.interpret_prep(p, tname), table=modeling.prep_table(p), order=1)

y = p["data"][p["target"]]
fig, ax = plt.subplots(figsize=(7, 3.1))
bins = np.linspace(y.min(), y.max(), 40)
edges = [-np.inf] + p["cuts"] + [np.inf]
for i, c in enumerate(modeling.CLASSES):
    part = y[(y > edges[i]) & (y <= edges[i + 1])]
    ax.hist(part, bins=bins, color=LEVEL_COLORS[c], label=LEVEL_ES[c], alpha=0.9)
for cut in p["cuts"]:
    ax.axvline(cut, color=INK, linestyle="--", linewidth=1)
ax.set_xlabel(f"{tname} (%)")
ax.set_ylabel("Tracts")
ax.legend(fontsize=7.5, frameon=False)
ax.set_title("Discretización del objetivo en niveles de riesgo (cuartiles de train)", color=INK)
ui.render(
    "prep_levels", ui.S_PREP, "Definición de los niveles de riesgo",
    "Histograma del resultado de salud coloreado por el nivel asignado. Las líneas discontinuas son los "
    "cuartiles Q1, Q2 (mediana) y Q3 calculados solo con el conjunto de entrenamiento.",
    f"Los cortes quedan en {p['cuts'][0]:.1f} %, {p['cuts'][1]:.1f} % y {p['cuts'][2]:.1f} %. El nivel "
    f"Crítico agrupa el 25 % de tracts con mayor prevalencia de {TARGET_INFO.get(p['target'], tname)}, desde "
    f"{p['cuts'][2]:.1f} % hasta {y.max():.1f} %: es un rango más amplio que el de los demás niveles por la "
    "asimetría positiva observada en el EDA. Los niveles son relativos al área de estudio, no umbrales clínicos.",
    fig=fig, order=2,
)
