import streamlit as st

import ui
from engine import modeling

st.title("🔁 Fase IV · Validación cruzada")
r = ui.need("train", "Primero entrene los modelos en «IV. Entrenamiento».")

ui.render("cv_folds", ui.S_MODEL, "Distribución del F1 por fold", modeling.HOW_CV, modeling.interpret_cv(r),
          fig=modeling.fig_cv(r), table=modeling.cv_table(r), order=30)
ui.render("cv_learning", ui.S_MODEL, "Curva de aprendizaje del modelo seleccionado", modeling.HOW_LEARNING,
          modeling.interpret_learning_curve(r), fig=modeling.fig_learning_curve(r), order=31)
