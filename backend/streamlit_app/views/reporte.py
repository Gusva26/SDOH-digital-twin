from datetime import datetime

import pandas as pd
import streamlit as st

import ui
from engine import llm
from engine.report import SECTIONS, build_pdf, save_and_register

st.title("📄 Fase VI · Reporte PDF")
items = list(st.session_state.report.values())
if not items:
    st.warning("Aún no hay resultados: recorra las fases de la app; cada gráfico y tabla se añade al reporte.")
    st.stop()

summary = pd.DataFrame(
    [{"Fase": s, "Gráficos": sum(1 for i in items if i.section == s and i.png),
      "Tablas": sum(1 for i in items if i.section == s and i.table is not None),
      "Con interpretación IA": sum(1 for i in items if i.section == s and i.llm_text)} for s in SECTIONS]
)
st.markdown("#### Contenido del reporte")
st.dataframe(summary, width="stretch", hide_index=True)
st.caption(
    f"{len(items)} resultados. Cada uno se incluye con su bloque «Cómo se lee» (explicabilidad) e "
    "«Interpretación» calculada; los que tengan interpretación IA la incluyen además."
)

if llm.available():
    if st.button("✨ Generar resumen ejecutivo (LangChain, salida estructurada)"):
        with st.spinner("Redactando…"):
            try:
                st.session_state.executive = llm.executive_summary(items)
            except Exception as exc:
                st.error(f"No se pudo generar: {exc}")
    ex = st.session_state.executive
    if ex:
        st.markdown("**Panorama.** " + ex["panorama"])
        for label, key in (("Hallazgos", "hallazgos"), ("Limitaciones", "limitaciones"), ("Recomendaciones", "recomendaciones")):
            st.markdown(f"**{label}**\n" + "\n".join(f"- {x}" for x in ex[key]))
else:
    st.caption("Configure OPENAI_API_KEY para añadir un resumen ejecutivo generado con LangChain.")

ds, prep, tr = st.session_state.ds, st.session_state.prep, st.session_state.train
meta = {}
if ds:
    meta["Fuente"] = ds.source
    meta["Tracts"] = str(len(ds.df))
if prep:
    meta["Objetivo"] = ui.label(prep["target"])
if tr:
    meta["Modelo seleccionado"] = tr["models"][tr["best"]]["name"]

if st.button("Generar reporte PDF", type="primary"):
    with st.spinner("Maquetando PDF…"):
        pdf = build_pdf(items, meta, st.session_state.executive)
        st.session_state.pdf = pdf
        try:
            user = st.session_state.user or {}
            path = save_and_register(pdf, user.get("id"), f"CRISP-DM Streamlit · {meta.get('Objetivo', 'análisis')}")
            st.success(f"Guardado en {path}" + (" y registrado en la página Reportes de la plataforma." if user.get("id") else "."))
        except Exception as exc:
            st.warning(f"PDF generado, pero no se pudo registrar en la plataforma: {exc}")
if st.session_state.get("pdf"):
    st.download_button("⬇️ Descargar PDF", st.session_state.pdf, file_name=f"CRISPDM_{datetime.now():%Y%m%d_%H%M}.pdf",
                       mime="application/pdf")
