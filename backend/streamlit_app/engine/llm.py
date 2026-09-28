"""Módulo LangChain: interpretación asistida, asistente conversacional y resumen
ejecutivo estructurado sobre los resultados calculados por el motor.

El LLM nunca calcula: recibe los números ya obtenidos (tablas e interpretaciones
deterministas) y solo los redacta/contextualiza. Si no hay API key, la app sigue
funcionando con las interpretaciones deterministas.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from .report import ReportItem

AI_MODEL = os.getenv("AI_MODEL", "gpt-4o-mini")

SYSTEM = (
    "Eres un estadístico y científico de datos experto en Determinantes Sociales de la Salud "
    "(SDOH) que explica resultados a gestores sanitarios. Responde en español, con precisión "
    "técnica y lenguaje claro. Usa EXCLUSIVAMENTE los números del contexto; si algo no está en "
    "él, dilo. No inventes cifras. Recuerda que los datos son agregados por census tract (CDC "
    "PLACES): las conclusiones son ecológicas y asociativas, no causales ni individuales."
)


def available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _llm(temperature: float = 0.2):
    from langchain.chat_models import init_chat_model

    return init_chat_model(AI_MODEL, model_provider="openai", temperature=temperature)


def item_digest(it: ReportItem, max_rows: int = 25) -> str:
    parts = [f"## {it.title}", f"Método: {it.how}", f"Interpretación calculada: {it.interpretation}"]
    if it.table is not None and not it.table.empty:
        parts.append("Tabla:\n" + it.table.head(max_rows).to_csv(index=False, float_format="%.4g"))
    return "\n".join(parts)


def context_digest(items: List[ReportItem], max_chars: int = 24000) -> str:
    text = "\n\n".join(item_digest(i, 12) for i in sorted(items, key=lambda i: (i.section, i.order)))
    return text[:max_chars] if text else "(todavía no se ha calculado ningún resultado)"


def explain_item(it: ReportItem) -> str:
    """Cadena LCEL: prompt | llm | parser → interpretación ampliada de un gráfico/tabla."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM),
            (
                "human",
                "Explica este resultado del análisis CRISP-DM en 3 párrafos breves: "
                "(1) qué muestra y cómo leerlo, (2) qué dicen los números concretos, "
                "(3) implicación práctica para priorizar intervenciones y una cautela.\n\n{digest}",
            ),
        ]
    )
    chain = prompt | _llm() | StrOutputParser()
    return chain.invoke({"digest": item_digest(it)})


def chat(question: str, history: List[Tuple[str, str]], items: List[ReportItem]) -> str:
    """Asistente conversacional con memoria de la sesión y contexto de resultados."""
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM + "\n\nResultados disponibles de la sesión:\n{context}"),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    msgs = [HumanMessage(c) if role == "user" else AIMessage(c) for role, c in history[-12:]]
    chain = prompt | _llm(0.3) | StrOutputParser()
    return chain.invoke({"context": context_digest(items), "history": msgs, "question": question})


class ResumenEjecutivo(BaseModel):
    panorama: str = Field(description="Dos o tres frases con el estado general del análisis y el modelo elegido.")
    hallazgos: List[str] = Field(description="3 a 5 hallazgos; cada uno cita al menos un número del contexto.")
    limitaciones: List[str] = Field(description="2 a 4 riesgos o limitaciones metodológicas.")
    recomendaciones: List[str] = Field(description="2 a 4 recomendaciones accionables para gestores de salud.")


def executive_summary(items: List[ReportItem]) -> Optional[dict]:
    """Salida estructurada (Pydantic) para el reporte PDF."""
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM), ("human", "Redacta el resumen ejecutivo del análisis:\n\n{context}")]
    )
    chain = prompt | _llm().with_structured_output(ResumenEjecutivo)
    return chain.invoke({"context": context_digest(items)}).model_dump()
