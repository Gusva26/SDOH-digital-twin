import streamlit as st

from engine import llm

st.title("🦜 Asistente LangChain")
st.markdown(
    "Módulo construido con **LangChain** (cadenas LCEL `prompt | llm | parser`, memoria de "
    "conversación con `MessagesPlaceholder` y salida estructurada con Pydantic). El asistente "
    "responde sobre los resultados calculados en esta sesión: el LLM no calcula, solo explica "
    "los números que el motor ya obtuvo."
)
if not llm.available():
    st.error("Falta OPENAI_API_KEY en el entorno (.env). El resto de la app funciona sin ella.")
    st.stop()

items = list(st.session_state.report.values())
st.caption(f"Contexto disponible: {len(items)} resultados (modelo {llm.AI_MODEL}).")

tab_chat, tab_item = st.tabs(["Conversación", "Interpretar un gráfico o tabla"])

with tab_chat:
    for role, content in st.session_state.chat_lc:
        st.chat_message(role).markdown(content)
    if st.button("Limpiar conversación"):
        st.session_state.chat_lc = []
        st.rerun()
    q = st.chat_input("Pregunte, p. ej.: ¿Por qué se eligió este modelo y es la diferencia significativa?")
    if q:
        st.chat_message("user").markdown(q)
        with st.chat_message("assistant"), st.spinner("Pensando…"):
            try:
                ans = llm.chat(q, st.session_state.chat_lc, items)
            except Exception as exc:
                ans = f"Error del proveedor LLM: {exc}"
            st.markdown(ans)
        st.session_state.chat_lc += [("user", q), ("assistant", ans)]

with tab_item:
    if not items:
        st.info("Aún no hay gráficos ni tablas calculados.")
        st.stop()
    sel = st.selectbox("Resultado", items, format_func=lambda i: f"{i.section.split(' ')[0]} · {i.title}")
    if sel.png:
        st.image(sel.png, width="stretch")
    if sel.table is not None:
        st.dataframe(sel.table, width="stretch", hide_index=True)
    st.success(f"**🔎 Interpretación calculada**\n\n{sel.interpretation}")
    if st.button("✨ Interpretar con LangChain", type="primary"):
        with st.spinner("Generando…"):
            sel.llm_text = llm.explain_item(sel)
    if sel.llm_text:
        st.markdown(f"**✨ Interpretación asistida (se incluirá en el reporte)**\n\n{sel.llm_text}")
