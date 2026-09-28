import streamlit as st

import ui
from engine import modeling

st.title("🎛️ Fase IV · Hiperparámetros")
r = ui.need("train", "Primero entrene los modelos en «IV. Entrenamiento».")

best = r["models"][r["best"]]
st.markdown(
    f"Búsqueda en rejilla con validación cruzada {r['n_splits']}×{r['n_repeats']}. "
    f"Modelo seleccionado: **{best['name']}** con `{modeling._fmt_params(best['best_params'])}`."
)
for i, key in enumerate(r["ranking"]):
    m = r["models"][key]
    ui.render(f"hp_{key}", ui.S_MODEL, f"Hiperparámetros · {m['name']}", modeling.HOW_HYPERPARAMS,
              modeling.interpret_hyperparams(m), fig=modeling.fig_hyperparams(m), table=m["grid"], order=10 + i)
