import pandas as pd
import streamlit as st

import ui
from engine import modeling

st.title("🏋️ Fase IV · Entrenamiento")
p = ui.need("prep", "Primero aplique la preparación en «III. Preparación».")
st.markdown(
    f"Objetivo: **{ui.label(p['target'])}** · {len(p['X_train'])} tracts de entrenamiento · "
    f"{len(p['X_test'])} de prueba · {len(p['features'])} determinantes."
)
c1, c2 = st.columns(2)
k = c1.slider("Folds de validación cruzada (k)", 3, 10, 5)
reps = c2.slider("Repeticiones de la CV", 1, 5, 2)
st.caption(f"Cada configuración se evaluará con {k * reps} particiones. Más folds dan pruebas inferenciales "
           "más potentes pero tardan más.")
if st.button("Entrenar y comparar modelos", type="primary"):
    ui.reset_from(ui.S_MODEL)
    bar = st.progress(0.0, "Iniciando…")
    st.session_state.train = modeling.train_all(p, k, reps, progress=lambda f, t: bar.progress(f, t))
    bar.empty()

specs = modeling.candidates()
cand = pd.DataFrame(
    [{"Modelo": s["name"], "Rol": s["role"],
      "Hiperparámetros a buscar": "; ".join(f"{k_.replace('model__', '')} ∈ {v}" for k_, v in s["grid"].items()) or "—",
      "Configuraciones": int(pd.Series([len(v) for v in s["grid"].values()]).prod()) if s["grid"] else 1}
     for s in specs.values()]
)
ui.render(
    "train_candidates", ui.S_MODEL, "Algoritmos candidatos y espacio de búsqueda",
    "Cada fila es un algoritmo con su papel en la comparación y la rejilla de hiperparámetros que se "
    "explora. Todos comparten el mismo preprocesado (imputación por mediana + estandarización) dentro de "
    "un Pipeline de scikit-learn, de modo que la comparación es justa.",
    f"Se comparan familias complementarias: un modelo lineal interpretable, dos ensambles de árboles "
    f"(bagging y boosting) y una red neuronal, más una línea base aleatoria que fija el suelo de "
    f"desempeño. En total {int(cand['Configuraciones'].sum())} configuraciones × {k * reps} particiones.",
    table=cand, order=0,
)

r = ui.need("train", "Pulse «Entrenar y comparar modelos».")
ui.render("train_summary", ui.S_MODEL, "Resultados del entrenamiento", modeling.HOW_TRAINING,
          modeling.interpret_training(r), fig=modeling.fig_training(r), table=modeling.training_table(r), order=1)
st.caption("Continúe con «Hiperparámetros», «Validación cruzada» y «Selección del mejor modelo».")
