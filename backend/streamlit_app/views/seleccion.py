import streamlit as st

import ui
from engine import modeling

st.title("🏆 Fase IV · Selección del mejor modelo")
r = ui.need("train", "Primero entrene los modelos en «IV. Entrenamiento».")
p = st.session_state.prep
ds = st.session_state.ds

best = r["models"][r["best"]]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Modelo seleccionado", best["name"])
c2.metric("F1-macro CV", f"{best['cv_mean']:.3f}", f"± {best['cv_std']:.3f}")
c3.metric("F1-macro prueba", f"{best['test_f1']:.3f}")
c4.metric("ROC-AUC prueba", f"{best['test_auc']:.3f}")

ui.render("sel_ranking", ui.S_MODEL, "Ranking de candidatos y criterio de selección", modeling.HOW_SELECTION,
          modeling.interpret_selection(r), table=modeling.selection_table(r), order=40)
ui.render("sel_confusion", ui.S_MODEL, "Matriz de confusión del modelo seleccionado", modeling.HOW_CONFUSION,
          modeling.interpret_confusion(r), fig=modeling.fig_confusion(r), order=41)
ui.render("sel_report", ui.S_MODEL, "Métricas por nivel de riesgo", modeling.HOW_CLASS_REPORT,
          modeling.interpret_class_report(r), table=modeling.class_report_table(r), order=42)
ui.render("sel_roc", ui.S_MODEL, "Curvas ROC por nivel", modeling.HOW_ROC,
          modeling.interpret_roc(r, p["y_test"]), fig=modeling.fig_roc(r, p["y_test"]), order=43)
imp = r["importance"].assign(Variable=lambda d: d["Código"].map(ds.label))[["Variable", "Importancia", "Desv."]]
ui.render("sel_importance", ui.S_MODEL, "Explicabilidad global: importancia por permutación", modeling.HOW_IMPORTANCE,
          modeling.interpret_importance(r, ds.names), fig=modeling.fig_importance(r, ds.names),
          table=imp.round(4), order=44)
