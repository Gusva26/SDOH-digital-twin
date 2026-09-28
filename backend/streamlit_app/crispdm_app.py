"""Motor CRISP-DM en Python + Streamlit para el SDOH Digital Twin.

Ejecutar (desde backend/):  streamlit run streamlit_app/crispdm_app.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for path in (HERE, os.path.dirname(HERE)):  # engine/ui y el paquete `app` del backend
    if path not in sys.path:
        sys.path.insert(0, path)

import streamlit as st  # noqa: E402

import ui  # noqa: E402

st.set_page_config(page_title="SDOH · Motor CRISP-DM", page_icon="🧬", layout="wide")
ui.init_state()
ui.require_login()

V = os.path.join(HERE, "views")
nav = st.navigation(
    {
        "CRISP-DM": [
            st.Page(os.path.join(V, "negocio.py"), title="I. Comprensión del negocio", icon="🎯", default=True),
            st.Page(os.path.join(V, "carga.py"), title="II. Carga del dataset", icon="📥"),
            st.Page(os.path.join(V, "eda.py"), title="II. EDA", icon="🔍"),
            st.Page(os.path.join(V, "preparacion.py"), title="III. Preparación", icon="🧹"),
            st.Page(os.path.join(V, "entrenamiento.py"), title="IV. Entrenamiento", icon="🏋️"),
            st.Page(os.path.join(V, "hiperparametros.py"), title="IV. Hiperparámetros", icon="🎛️"),
            st.Page(os.path.join(V, "validacion.py"), title="IV. Validación cruzada", icon="🔁"),
            st.Page(os.path.join(V, "seleccion.py"), title="IV. Selección del mejor modelo", icon="🏆"),
            st.Page(os.path.join(V, "inferencia.py"), title="V. Pruebas inferenciales", icon="📐"),
            st.Page(os.path.join(V, "despliegue.py"), title="VI. Despliegue: predicción", icon="🚀"),
            st.Page(os.path.join(V, "reporte.py"), title="VI. Reporte PDF", icon="📄"),
        ],
        "Inteligencia artificial": [
            st.Page(os.path.join(V, "langchain_view.py"), title="Asistente LangChain", icon="🦜"),
            st.Page(os.path.join(V, "langflow_view.py"), title="LangFlow", icon="🌊"),
        ],
    }
)

with st.sidebar:
    user = st.session_state.user or {}
    st.caption(f"Usuario: **{user.get('username', '—')}**")
    ds = st.session_state.ds
    st.caption(f"Dataset: **{len(ds.df)} tracts**" if ds else "Dataset: sin cargar")
    tr = st.session_state.train
    if tr:
        st.caption(f"Mejor modelo: **{tr['models'][tr['best']]['name']}**")
    st.caption(f"Resultados para el reporte: **{len(st.session_state.report)}**")
    if ui.REQUIRE_LOGIN and st.button("Cerrar sesión"):
        st.session_state.clear()
        st.rerun()

nav.run()
