"""Módulo LangFlow: ejecuta el flujo `sdoh-assistant` (langflow/sdoh_assistant.json)
inyectando como contexto los resultados del análisis de la sesión Streamlit."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests

from app.services.ai_service import (
    LANGFLOW_API_KEY,
    LANGFLOW_BASE_URL,
    LANGFLOW_FLOW_ID,
    LANGFLOW_PROMPT_NODE,
    LANGFLOW_TIMEOUT,
)

FLOW_FILE = os.getenv("LANGFLOW_FLOW_FILE", "/langflow/sdoh_assistant.json")


def config() -> Dict[str, str]:
    return {
        "base_url": LANGFLOW_BASE_URL,
        "flow_id": LANGFLOW_FLOW_ID,
        "prompt_node": LANGFLOW_PROMPT_NODE,
        "api_key": "configurada" if LANGFLOW_API_KEY else "no configurada",
    }


def health() -> Dict[str, Any]:
    try:
        r = requests.get(f"{LANGFLOW_BASE_URL}/health", timeout=5)
        return {"ok": r.ok, "status": r.status_code, "detail": r.text[:200]}
    except requests.RequestException as exc:
        return {"ok": False, "status": None, "detail": str(exc)[:200]}


def flow_nodes() -> Optional[List[Dict[str, str]]]:
    """Componentes del flujo versionado en el repositorio (para mostrarlo en pantalla)."""
    if not os.path.exists(FLOW_FILE):
        return None
    with open(FLOW_FILE, encoding="utf-8") as fh:
        flow = json.load(fh)
    data = flow.get("data", flow)
    nodes = []
    for n in data.get("nodes", []):
        d = n.get("data", {})
        node = d.get("node", {})
        nodes.append({
            "id": d.get("id", n.get("id", "")),
            "tipo": d.get("type", ""),
            "nombre": node.get("display_name", d.get("type", "")),
            "descripción": (node.get("description") or "")[:140],
        })
    edges = [(e.get("source", ""), e.get("target", "")) for e in data.get("edges", [])]
    return [{**n, "conecta con": ", ".join(t for s, t in edges if s == n["id"])} for n in nodes]


def run(message: str, context: str, language: str = "es", session_id: Optional[str] = None) -> str:
    payload: Dict[str, Any] = {
        "input_value": message,
        "input_type": "chat",
        "output_type": "chat",
        "tweaks": {
            LANGFLOW_PROMPT_NODE: {
                "sdoh_context": context,
                "language": "Spanish" if language == "es" else "English",
            }
        },
    }
    if session_id:
        payload["session_id"] = session_id
    headers = {"x-api-key": LANGFLOW_API_KEY} if LANGFLOW_API_KEY else {}
    resp = requests.post(f"{LANGFLOW_BASE_URL}/api/v1/run/{LANGFLOW_FLOW_ID}", json=payload,
                         headers=headers, timeout=LANGFLOW_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["outputs"][0]["outputs"][0]["results"]["message"]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Respuesta inesperada de LangFlow: {str(data)[:300]}") from exc
