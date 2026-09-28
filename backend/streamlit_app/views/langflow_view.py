import uuid

import pandas as pd
import streamlit as st

from engine import langflow_client as lf
from engine import llm

st.title("🌊 Módulo LangFlow")
st.markdown(
    "Ejecuta el flujo visual **`sdoh-assistant`** diseñado en LangFlow "
    "(`langflow/sdoh_assistant.json`): *Chat Input + Message History → Prompt Template → "
    "Language Model → Chat Output*. La app inyecta en la variable `sdoh_context` del prompt los "
    "resultados del análisis de esta sesión, de modo que el flujo responde con los números reales."
)

cfg = lf.config()
h = lf.health()
c1, c2, c3 = st.columns(3)
c1.metric("Servidor LangFlow", "Conectado" if h["ok"] else "Sin conexión")
c2.metric("Flujo", cfg["flow_id"])
c3.metric("API key", cfg["api_key"])
st.caption(f"URL: {cfg['base_url']} · nodo del prompt: «{cfg['prompt_node']}»")
if not h["ok"]:
    st.warning(
        f"No se pudo contactar con LangFlow ({h['detail']}). Arránquelo en el puerto 7860, importe "
        "`langflow/sdoh_assistant.json` y configure LANGFLOW_BASE_URL / LANGFLOW_API_KEY en .env "
        "(ver langflow/README.md)."
    )

nodes = lf.flow_nodes()
if nodes:
    st.markdown("#### Componentes del flujo")
    st.dataframe(pd.DataFrame(nodes), width="stretch", hide_index=True)
    st.info(
        "**📖 Cómo se lee** — Cada fila es un componente del grafo de LangFlow y «conecta con» indica a "
        "qué nodos envía su salida. El mensaje del usuario y el historial de la sesión alimentan la "
        "plantilla de prompt, que el modelo de lenguaje completa y devuelve por Chat Output."
    )

items = list(st.session_state.report.values())
context = llm.context_digest(items)
st.session_state.setdefault("lf_session", f"streamlit-{uuid.uuid4()}")

tab_chat, tab_item = st.tabs(["Conversación con el flujo", "Interpretar un gráfico o tabla"])
with tab_chat:
    for role, content in st.session_state.chat_lf:
        st.chat_message(role).markdown(content)
    if st.button("Nueva conversación"):
        st.session_state.chat_lf = []
        st.session_state.lf_session = f"streamlit-{uuid.uuid4()}"
        st.rerun()
    q = st.chat_input("Pregunte al flujo de LangFlow…", disabled=not h["ok"])
    if q:
        st.chat_message("user").markdown(q)
        with st.chat_message("assistant"), st.spinner("Ejecutando flujo…"):
            try:
                ans = lf.run(q, context, "es", st.session_state.lf_session)
            except Exception as exc:
                ans = f"Error al ejecutar el flujo: {exc}"
            st.markdown(ans)
        st.session_state.chat_lf += [("user", q), ("assistant", ans)]

with tab_item:
    if not items:
        st.info("Aún no hay gráficos ni tablas calculados.")
        st.stop()
    sel = st.selectbox("Resultado", items, format_func=lambda i: f"{i.section.split(' ')[0]} · {i.title}", key="lf_item")
    st.success(f"**🔎 Interpretación calculada**\n\n{sel.interpretation}")
    if st.button("🌊 Interpretar con LangFlow", type="primary", disabled=not h["ok"]):
        with st.spinner("Ejecutando flujo…"):
            try:
                sel.llm_text = lf.run(
                    "Explica este resultado para un gestor de salud en 3 párrafos breves: qué muestra, "
                    "qué dicen los números y qué implica en la práctica.",
                    llm.item_digest(sel), "es",
                )
            except Exception as exc:
                st.error(f"Error al ejecutar el flujo: {exc}")
    if sel.llm_text:
        st.markdown(f"**Interpretación asistida (se incluirá en el reporte)**\n\n{sel.llm_text}")
