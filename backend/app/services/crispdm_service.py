"""Implementación operativa del modelo de proceso CRISP-DM sobre el gemelo SP-5.

Cada una de las seis fases —comprensión del negocio, comprensión de los datos,
preparación de los datos, modelado, evaluación y despliegue— se expone como una
función que inspecciona o ejecuta el estado real del sistema, de modo que la
metodología sea auditable y no meramente declarativa.

Referencias: Chapman et al. (2000); Wirth & Hipp (2000); Schröer et al. (2021).
"""

import json
import os
import platform
import random
import statistics
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import func, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.geo import (
    CatchmentMembership,
    CensusTract,
    County,
    Hospital,
    HospitalCatchment,
)
from app.models.sdoh import (
    Alert,
    AlertRule,
    EquityIndex,
    IndicatorCatalog,
    SDOHIndicator,
)
from app.models.user import AuditLog, Permission, Role, User
from app.services import equity_service, ml_service

# --------------------------------------------------------------------------- #
# Metadatos de las fases
# --------------------------------------------------------------------------- #

LATENCY_TARGET_SECONDS = 5.0  # H3
POPULATION_TARGET = 500_000  # H3

PHASES: List[Dict[str, Any]] = [
    {
        "key": "business-understanding",
        "order": 1,
        "roman": "I",
        "name": "Comprensión del negocio",
        "name_en": "Business Understanding",
        "layer": "Transversal (gobierno)",
        "objectives": ["OE1"],
        "question": "¿Qué decisión de equidad debe soportar el gemelo y para qué actor?",
    },
    {
        "key": "data-understanding",
        "order": 2,
        "roman": "II",
        "name": "Comprensión de los datos",
        "name_en": "Data Understanding",
        "layer": "Capa de sensado físico",
        "objectives": ["OE1", "OE3"],
        "question": "¿Qué fuentes existen, con qué cobertura, granularidad y calidad?",
    },
    {
        "key": "data-preparation",
        "order": 3,
        "roman": "III",
        "name": "Preparación de los datos",
        "name_en": "Data Preparation",
        "layer": "Capa de integración digital",
        "objectives": ["OE3"],
        "question": "¿Cómo armonizar, normalizar y unir espacialmente fuentes heterogéneas?",
    },
    {
        "key": "modeling",
        "order": 4,
        "roman": "IV",
        "name": "Modelado",
        "name_en": "Modeling",
        "layer": "Capa de simulación y analítica",
        "objectives": ["OE2", "OE4"],
        "question": "¿Cómo cuantificar la vulnerabilidad y activar la respuesta?",
    },
    {
        "key": "evaluation",
        "order": 5,
        "roman": "V",
        "name": "Evaluación",
        "name_en": "Evaluation",
        "layer": "Capa de simulación y analítica",
        "objectives": ["OE2", "OE4"],
        "question": "¿Se cumplen los criterios de latencia, validez y utilidad?",
    },
    {
        "key": "deployment",
        "order": 6,
        "roman": "VI",
        "name": "Despliegue",
        "name_en": "Deployment",
        "layer": "Transversal (interfaz y gobierno)",
        "objectives": ["OE2", "OE3"],
        "question": "¿Cómo se opera, visualiza, difunde y audita de forma continua?",
    },
]

PHASE_KEYS = [p["key"] for p in PHASES]

#: Último banco de pruebas de la Fase V, persistido para que sobreviva a recargas.
EVALUATION_PATH = os.path.join(ml_service.ARTIFACT_DIR, "crispdm_evaluation.json")


def _last_evaluation() -> Optional[Dict[str, Any]]:
    if not os.path.exists(EVALUATION_PATH):
        return None
    try:
        with open(EVALUATION_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _default_year(db: Session) -> int:
    year = db.query(func.max(SDOHIndicator.year)).scalar()
    return int(year) if year else datetime.utcnow().year


def phase_overview(db: Session) -> Dict[str, Any]:
    """Estado de avance de las seis fases según los artefactos presentes en la BD."""
    year = _default_year(db)
    tracts = db.query(func.count(CensusTract.id)).scalar() or 0
    catalog = db.query(func.count(IndicatorCatalog.id)).scalar() or 0
    values = db.query(func.count(SDOHIndicator.id)).scalar() or 0
    computed = (
        db.query(func.count(EquityIndex.id))
        .filter(EquityIndex.method == equity_service.METHOD)
        .scalar()
        or 0
    )
    alerts = db.query(func.count(Alert.id)).scalar() or 0
    audits = db.query(func.count(AuditLog.id)).scalar() or 0
    ml = ml_service.status()

    evaluated = _last_evaluation() is not None
    # Cada fase: (lista, qué falta para completarla).
    checks = {
        "business-understanding": (bool(db.query(Role).first()), "Crear roles y usuarios (arranque del backend)."),
        "data-understanding": (tracts > 0 and catalog > 0, "Cargar datos: make seed-etl."),
        "data-preparation": (values > 0, "Cargar valores SDOH: make seed-etl."),
        "modeling": (
            computed > 0 and ml["trained"],
            "Ejecutar pipeline (índice compuesto) y entrenar los 4 modelos ML.",
        ),
        "evaluation": (
            evaluated and ml["trained"],
            "Ejecutar el banco de pruebas de la Fase V y entrenar los modelos ML.",
        ),
        "deployment": (
            ml["trained"] and (audits > 0 or alerts > 0),
            "Desplegar el modelo ML (entrenarlo) y operar la plataforma.",
        ),
    }
    status = {k: v[0] for k, v in checks.items()}
    phases = []
    for phase in PHASES:
        ready, todo = checks[phase["key"]]
        phases.append({**phase, "ready": ready, "todo": None if ready else todo})
    return {
        "year": year,
        "phases": phases,
        "completed": sum(1 for v in status.values() if v),
        "total": len(PHASES),
        "counts": {
            "tracts": tracts,
            "indicators": catalog,
            "values": values,
            "computed_indexes": computed,
            "alerts": alerts,
            "audit_events": audits,
            "ml_trained": ml["trained"],
            "ml_best_model": ml.get("best_model_name"),
            "evaluated": evaluated,
        },
    }


# --------------------------------------------------------------------------- #
# Fase I — Comprensión del negocio
# --------------------------------------------------------------------------- #

def business_understanding(db: Session) -> Dict[str, Any]:
    roles = []
    for role in db.query(Role).order_by(Role.id).all():
        roles.append(
            {
                "code": role.code,
                "name": role.name,
                "description": role.description,
                "is_system": bool(role.is_system),
                "permissions": sorted(p.code for p in role.permissions),
                "users": len(role.users) if hasattr(role, "users") else None,
            }
        )

    return {
        "phase": "business-understanding",
        "goal": (
            "Permitir que un gestor hospitalario identifique, dentro de su área de "
            "captación, los tractos censales con mayor vulnerabilidad social y "
            "justifique la asignación de recursos con evidencia trazable."
        ),
        "research_question": (
            "¿Cómo puede arquitecturarse la tecnología de gemelos digitales para "
            "habilitar el monitoreo en tiempo real de los SDOH y la equidad en salud "
            "dentro de áreas de captación hospitalaria metropolitanas?"
        ),
        "objectives": [
            {
                "code": "OE1",
                "text": "Formalizar el área de captación como sistema socioespacial dinámico.",
            },
            {
                "code": "OE2",
                "text": "Diseñar la arquitectura de tres capas con recomputación < 5 s.",
            },
            {
                "code": "OE3",
                "text": "Especificar protocolos de interoperabilidad para ingestión heterogénea.",
            },
            {
                "code": "OE4",
                "text": "Desarrollar métricas de equidad computables en tiempo real.",
            },
        ],
        "hypotheses": [
            {
                "code": "H1",
                "text": "Menor latencia y mayor resolución espacial que la evaluación retrospectiva.",
            },
            {
                "code": "H2",
                "text": "El modelado de accesibilidad con decaimiento mejora la validez predictiva.",
            },
            {
                "code": "H3",
                "text": f"Recomputación < {LATENCY_TARGET_SECONDS:.0f} s hasta "
                f"{POPULATION_TARGET:,} habitantes.".replace(",", " "),
            },
        ],
        "success_criteria": {
            "business": [
                "Priorización explícita de tractos críticos por área de captación.",
                "Reporte difundible a actores no técnicos (PDF, Word, Excel, CSV).",
                "Trazabilidad de cada índice hasta la fuente y el método de cálculo.",
            ],
            "data_mining": [
                f"Latencia de recomputación < {LATENCY_TARGET_SECONDS:.0f} s (H3).",
                "Índices reproducibles e idempotentes ante recomputación.",
                "Auditoría de toda operación de escritura sobre datos.",
            ],
        },
        "constraints": [
            "Solo fuentes públicas agregadas; sin datos identificables de paciente.",
            "Unidad mínima de análisis: tracto censal.",
            "Entorno fijado a Python 3.12 + PostgreSQL 16 / PostGIS 3.4.",
        ],
        "stakeholders": roles,
        "inventory": {
            "users": db.query(func.count(User.id)).scalar() or 0,
            "roles": db.query(func.count(Role.id)).scalar() or 0,
            "permissions": db.query(func.count(Permission.id)).scalar() or 0,
            "hospitals": db.query(func.count(Hospital.id)).scalar() or 0,
            "catchments": db.query(func.count(HospitalCatchment.id)).scalar() or 0,
        },
    }


# --------------------------------------------------------------------------- #
# Fase II — Comprensión de los datos
# --------------------------------------------------------------------------- #

def data_understanding(db: Session, year: Optional[int] = None) -> Dict[str, Any]:
    year = year or _default_year(db)
    tracts_total = db.query(func.count(CensusTract.id)).scalar() or 0

    sources = [
        {"source": s or "(sin declarar)", "indicators": n}
        for s, n in db.query(IndicatorCatalog.source, func.count(IndicatorCatalog.id))
        .group_by(IndicatorCatalog.source)
        .all()
    ]
    domains = [
        {"domain": d or "(sin dominio)", "indicators": n}
        for d, n in db.query(IndicatorCatalog.domain, func.count(IndicatorCatalog.id))
        .group_by(IndicatorCatalog.domain)
        .order_by(IndicatorCatalog.domain)
        .all()
    ]
    years = [
        {"year": y, "values": n}
        for y, n in db.query(SDOHIndicator.year, func.count(SDOHIndicator.id))
        .group_by(SDOHIndicator.year)
        .order_by(SDOHIndicator.year)
        .all()
    ]

    profile = []
    stats = (
        db.query(
            IndicatorCatalog.code,
            IndicatorCatalog.name,
            IndicatorCatalog.domain,
            IndicatorCatalog.source,
            IndicatorCatalog.higher_is_better,
            IndicatorCatalog.weight,
            func.count(SDOHIndicator.id),
            func.min(SDOHIndicator.value),
            func.max(SDOHIndicator.value),
            func.avg(SDOHIndicator.value),
        )
        .outerjoin(
            SDOHIndicator,
            (SDOHIndicator.catalog_id == IndicatorCatalog.id)
            & (SDOHIndicator.year == year),
        )
        .group_by(IndicatorCatalog.id)
        .order_by(IndicatorCatalog.domain, IndicatorCatalog.code)
        .all()
    )
    for code, name, domain, source, hib, weight, n, mn, mx, avg in stats:
        profile.append(
            {
                "code": code,
                "name": name,
                "domain": domain,
                "source": source,
                "direction": "protector" if hib == 1 else "riesgo",
                "weight": weight,
                "observations": n or 0,
                "completeness": round((n or 0) / tracts_total * 100, 2) if tracts_total else 0.0,
                "missing_tracts": max(tracts_total - (n or 0), 0),
                "min": round(mn, 2) if mn is not None else None,
                "max": round(mx, 2) if mx is not None else None,
                "mean": round(float(avg), 2) if avg is not None else None,
            }
        )

    total_expected = tracts_total * len(profile)
    total_observed = sum(p["observations"] for p in profile)

    # Marcador estable del conjunto demostrativo: el seed sintético crea tractos sin
    # geometría, mientras que una carga real (TIGER/Line) sí las trae.
    tracts_with_geometry = (
        db.query(func.count(CensusTract.id))
        .filter(CensusTract.geom.is_not(None))
        .scalar()
        or 0
    )
    seeded_indexes = (
        db.query(func.count(EquityIndex.id))
        .filter(EquityIndex.method == "seed_demo")
        .scalar()
        or 0
    )
    synthetic = tracts_total > 0 and (tracts_with_geometry == 0 or seeded_indexes > 0)

    population = db.query(func.coalesce(func.sum(CensusTract.total_population), 0)).scalar()

    return {
        "phase": "data-understanding",
        "year": year,
        "unit_of_analysis": "Tracto censal (census_tracts)",
        "aggregation_unit": "Área de captación (hospital_catchments · population_share)",
        "srid": 4326,
        "counts": {
            "counties": db.query(func.count(County.id)).scalar() or 0,
            "tracts": tracts_total,
            "hospitals": db.query(func.count(Hospital.id)).scalar() or 0,
            "catchments": db.query(func.count(HospitalCatchment.id)).scalar() or 0,
            "memberships": db.query(func.count(CatchmentMembership.id)).scalar() or 0,
            "indicators": len(profile),
            "values": total_observed,
            "population": int(population or 0),
            "tracts_with_geometry": tracts_with_geometry,
        },
        "sources": sources,
        "domains": domains,
        "years": years,
        "profile": profile,
        "quality": {
            "expected_observations": total_expected,
            "observed": total_observed,
            "completeness": round(total_observed / total_expected * 100, 2)
            if total_expected
            else 0.0,
            "uniqueness_constraint": "uq_tract_catalog_year (tract_id, catalog_id, year)",
            "checks": [
                "Completitud: registros sin GEOID, medida o valor son descartados en la ingesta.",
                "Consistencia: unicidad de la terna tracto-indicador-año.",
                "Plausibilidad: intervalos low_ci / high_ci y sample_size de la fuente.",
                "Direccionalidad: atributo higher_is_better del catálogo.",
            ],
        },
        "dataset_nature": "sintético-demostrativo" if synthetic else "cargado por ETL",
        "warning": (
            "El conjunto demostrativo se genera con semilla fija y sus tractos carecen "
            f"de geometría PostGIS ({tracts_with_geometry}/{tracts_total} con geom): "
            "habilita verificación funcional y pruebas de rendimiento, no inferencia "
            "epidemiológica."
        )
        if synthetic
        else None,
    }


# --------------------------------------------------------------------------- #
# Fase III — Preparación de los datos
# --------------------------------------------------------------------------- #

def data_preparation(
    db: Session, year: Optional[int] = None, weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    year = year or _default_year(db)
    catalog = db.query(IndicatorCatalog).order_by(IndicatorCatalog.code).all()

    started = time.perf_counter()
    w = equity_service.normalized_weights(catalog, weights)
    matrix = equity_service.build_normalized_matrix(db, year, catalog)
    elapsed = time.perf_counter() - started

    tract_ids = set()
    for scores in matrix.values():
        tract_ids.update(scores.keys())
    complete = [t for t in tract_ids if all(t in matrix.get(c.code, {}) for c in catalog)]

    indicators = []
    for item in catalog:
        scores = matrix.get(item.code, {})
        values = list(scores.values())
        indicators.append(
            {
                "code": item.code,
                "name": item.name,
                "domain": item.domain,
                "inverted": item.higher_is_better != 1,
                "raw_weight": item.weight,
                "normalized_weight": round(w.get(item.code, 0.0), 6),
                "tracts": len(scores),
                "min": round(min(values), 4) if values else None,
                "max": round(max(values), 4) if values else None,
                "mean": round(statistics.fmean(values), 4) if values else None,
            }
        )

    sample_ids = sorted(tract_ids)[:5]
    geoids = dict(
        db.query(CensusTract.id, CensusTract.geoid)
        .filter(CensusTract.id.in_(sample_ids))
        .all()
    ) if sample_ids else {}
    sample = [
        {
            "tract_id": t,
            "geoid": geoids.get(t),
            "scores": {
                item.code: round(matrix.get(item.code, {}).get(t), 4)
                for item in catalog
                if matrix.get(item.code, {}).get(t) is not None
            },
        }
        for t in sample_ids
    ]

    return {
        "phase": "data-preparation",
        "year": year,
        "steps": [
            {
                "step": "Selección",
                "detail": "Columnas LocationID, Measure y Data_Value del volcado de CDC PLACES.",
            },
            {
                "step": "Limpieza",
                "detail": "Descarte de registros sin GEOID, sin medida reconocida o con valor faltante.",
            },
            {
                "step": "Integración",
                "detail": "Resolución del GEOID contra census_tracts y de la medida contra indicator_catalog.",
            },
            {
                "step": "Formateo",
                "detail": "Conversión a punto flotante y asignación del año de referencia.",
            },
            {
                "step": "Carga",
                "detail": "Escritura transaccional bajo la restricción uq_tract_catalog_year.",
            },
        ],
        "transformations": [
            "Normalización mínimo-máximo a [0,1] por indicador y año.",
            "Inversión de escala (1 − x̃) de los indicadores de riesgo.",
            "Normalización de los pesos del catálogo a suma unitaria.",
            "Unión espacial tracto → captación ponderada por population_share.",
        ],
        "weights_sum": round(sum(w.values()), 6),
        "indicators": indicators,
        "coverage": {
            "tracts_with_data": len(tract_ids),
            "tracts_complete": len(complete),
            "tracts_partial": len(tract_ids) - len(complete),
            "tracts_total": db.query(func.count(CensusTract.id)).scalar() or 0,
        },
        "sample": sample,
        "elapsed_seconds": round(elapsed, 4),
    }


# --------------------------------------------------------------------------- #
# Fase IV — Modelado
# --------------------------------------------------------------------------- #

def modeling(
    db: Session, year: Optional[int] = None, weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    year = year or _default_year(db)
    catalog = db.query(IndicatorCatalog).order_by(IndicatorCatalog.code).all()
    w = equity_service.normalized_weights(catalog, weights)

    by_risk = dict(
        db.query(EquityIndex.risk_level, func.count(EquityIndex.id))
        .filter(EquityIndex.year == year)
        .group_by(EquityIndex.risk_level)
        .all()
    )
    by_method = [
        {"method": m or "(sin método)", "count": n}
        for m, n in db.query(EquityIndex.method, func.count(EquityIndex.id))
        .filter(EquityIndex.year == year)
        .group_by(EquityIndex.method)
        .all()
    ]

    return {
        "phase": "modeling",
        "year": year,
        "technique": "Índice compuesto ponderado de indicadores normalizados",
        "method_id": equity_service.METHOD,
        "rationale": (
            "Se elige frente a PCA o aprendizaje supervisado por interpretabilidad: en "
            "decisiones con impacto distributivo, la contribución de cada dominio debe "
            "poder explicarse y auditarse. Como complemento, un modelo supervisado "
            "(el mejor de 4 clasificadores) predice resultados de salud reales a partir "
            "de los determinantes sociales."
        ),
        "formula": {
            "composite": "E_t = Σ_k w_k · s_{k,t}",
            "protective": "s_{k,t} = x̃_{k,t}",
            "risk": "s_{k,t} = 1 − x̃_{k,t}",
            "vulnerability": "V_t = 100 − P_E(E_t)",
        },
        "risk_thresholds": [
            {"level": level, "min_percentile": threshold}
            for threshold, level in equity_service.RISK_THRESHOLDS
        ]
        + [{"level": "low", "min_percentile": 0.0}],
        "weights": [
            {
                "code": item.code,
                "name": item.name,
                "domain": item.domain,
                "direction": "protector" if item.higher_is_better == 1 else "riesgo",
                "raw": item.weight,
                "normalized": round(w.get(item.code, 0.0), 6),
            }
            for item in catalog
        ],
        "assumptions": [
            "Aditividad y compensabilidad entre dominios.",
            "Ponderación experta (no estimada a partir de resultados de salud).",
            "Ausencia de rezago temporal entre determinante social y resultado en salud.",
        ],
        "companion_models": [
            {
                "name": "Motor de reglas de alerta",
                "detail": "AlertRule(indicador, comparador ∈ {gt, lt, gte, lte}, umbral, severidad).",
                "rules": db.query(func.count(AlertRule.id)).scalar() or 0,
            },
            {
                "name": "Asistente conversacional con contexto",
                "detail": "LLM con contexto recuperado de la base; prohibido fabricar estadísticas.",
                "rules": None,
            },
            _ml_companion(),
            {
                "name": "Representación 3D del gemelo",
                "detail": "Color y altura de vóxel por nivel de riesgo del tracto.",
                "rules": None,
            },
        ],
        "stored_indexes": {
            "total": sum(by_risk.values()),
            "by_risk": {k or "(sin nivel)": v for k, v in by_risk.items()},
            "by_method": by_method,
        },
        "complexity": "O(n · k) sobre n tractos y k indicadores; percentil por bisección.",
    }


def _ml_companion() -> Dict[str, Any]:
    ml = ml_service.status()
    if not ml["trained"]:
        detail = "4 clasificadores (LR, RF, GB, MLP) sin entrenar todavía."
    else:
        best = ml["models"][0]
        detail = (
            f"4 clasificadores comparados; desplegado {ml['best_model_name']} "
            f"(F1 CV {best['cv_f1_mean']}) para predecir {ml['target']['name']}."
        )
    return {"name": "Modelo predictivo supervisado", "detail": detail, "rules": None}


def _ml_service_row() -> Dict[str, Any]:
    ml = ml_service.status()
    return {
        "name": "Modelo ML",
        "detail": (
            f"{ml['best_model_name']} · best_model.joblib · POST /api/ml/predict"
            if ml["trained"]
            else "Sin modelo desplegado (entrenar en Modelos ML)"
        ),
        "runtime": "scikit-learn · joblib",
        "status": "running" if ml["trained"] else "not trained",
    }


# --------------------------------------------------------------------------- #
# Fase V — Evaluación
# --------------------------------------------------------------------------- #

def _spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Correlación de rangos de Spearman sin dependencias externas."""
    if len(a) < 3 or len(a) != len(b):
        return None

    def ranks(values: Sequence[float]) -> List[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            mean_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = mean_rank
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (
        sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)
    ) ** 0.5
    return round(num / den, 4) if den else None


def evaluation_summary(db: Session, year: Optional[int] = None) -> Dict[str, Any]:
    """Lectura barata del estado de evaluación (sin ejecutar el banco de pruebas)."""
    year = year or _default_year(db)
    by_risk = dict(
        db.query(EquityIndex.risk_level, func.count(EquityIndex.id))
        .filter(
            EquityIndex.year == year,
            EquityIndex.method == equity_service.METHOD,
        )
        .group_by(EquityIndex.risk_level)
        .all()
    )
    quality = data_understanding(db, year)["quality"]
    alerts_open = (
        db.query(func.count(Alert.id)).filter(Alert.status == "open").scalar() or 0
    )
    return {
        "phase": "evaluation",
        "year": year,
        "criteria": [
            {
                "dimension": "Técnica",
                "metric": "Latencia de recomputación (mediana y p95)",
                "hypothesis": "H3",
                "target": f"< {LATENCY_TARGET_SECONDS:.0f} s",
            },
            {
                "dimension": "Calidad de datos",
                "metric": "Completitud por indicador y tracto",
                "hypothesis": "—",
                "target": "≥ 80 %",
            },
            {
                "dimension": "Validez",
                "metric": "Estabilidad del orden ante perturbación de pesos (Spearman)",
                "hypothesis": "H1, H2",
                "target": "ρ ≥ 0,90",
            },
            {
                "dimension": "Predicción (ML)",
                "metric": "F1-macro y ROC-AUC de 4 clasificadores (CV k=5 + test 20 %)",
                "hypothesis": "—",
                "target": "Mejor modelo desplegado",
            },
            {
                "dimension": "Utilidad",
                "metric": "Alertas accionables sobre tractos críticos",
                "hypothesis": "H1",
                "target": "Revisión experta",
            },
        ],
        "risk_distribution": {k or "(sin nivel)": v for k, v in by_risk.items()},
        "data_quality": quality,
        "open_alerts": alerts_open,
        "indexes_computed": sum(by_risk.values()),
        "last_run": _last_evaluation(),
        "note": (
            "La evaluación es formativa: mide viabilidad computacional y comportamiento "
            "de escala sobre el conjunto disponible, no valida epidemiológicamente los "
            "índices."
        ),
    }


def run_evaluation(
    db: Session,
    year: Optional[int] = None,
    repeats: int = 5,
    sensitivity_runs: int = 20,
    perturbation: float = 0.25,
    seed: int = 42,
) -> Dict[str, Any]:
    """Ejecuta el banco de pruebas de la Fase V sobre los datos reales de la BD."""
    year = year or _default_year(db)
    repeats = max(1, min(repeats, 30))
    sensitivity_runs = max(0, min(sensitivity_runs, 200))

    catalog = db.query(IndicatorCatalog).order_by(IndicatorCatalog.code).all()
    if not catalog:
        return {"phase": "evaluation", "year": year, "error": "Catálogo vacío"}

    # --- Latencia (H3) ---------------------------------------------------- #
    timings: List[float] = []
    composites: Dict[int, float] = {}
    matrix: Dict[str, Dict[int, float]] = {}
    weights: Dict[str, float] = {}
    for _ in range(repeats):
        t0 = time.perf_counter()
        composites, matrix, weights = equity_service.compute_composites(db, year)
        timings.append(time.perf_counter() - t0)

    if not composites:
        return {
            "phase": "evaluation",
            "year": year,
            "error": f"Sin valores SDOH para el año {year}",
        }

    timings.sort()
    p95 = timings[min(int(len(timings) * 0.95), len(timings) - 1)]
    tract_ids = list(composites.keys())
    population = (
        db.query(func.coalesce(func.sum(CensusTract.total_population), 0))
        .filter(CensusTract.id.in_(tract_ids))
        .scalar()
        or 0
    )
    median = statistics.median(timings)
    per_capita = median / population if population else 0.0

    # --- Estratificación de riesgo ---------------------------------------- #
    ordered = sorted(composites.values())
    baseline_levels = {
        t: equity_service.risk_level(
            100.0 - equity_service._percentile_of(ordered, c)
        )
        for t, c in composites.items()
    }
    distribution: Dict[str, int] = {}
    for level in baseline_levels.values():
        distribution[level] = distribution.get(level, 0) + 1

    # --- Sensibilidad de pesos (Monte Carlo) ------------------------------- #
    rng = random.Random(seed)
    correlations: List[float] = []
    stabilities: List[float] = []
    base_vector = [composites[t] for t in tract_ids]
    for _ in range(sensitivity_runs):
        perturbed = {
            code: max(w * (1 + rng.uniform(-perturbation, perturbation)), 1e-9)
            for code, w in weights.items()
        }
        total = sum(perturbed.values())
        perturbed = {k: v / total for k, v in perturbed.items()}
        alt = equity_service.composite_scores(matrix, perturbed)
        alt_vector = [alt.get(t, 0.0) for t in tract_ids]
        rho = _spearman(base_vector, alt_vector)
        if rho is not None:
            correlations.append(rho)
        alt_sorted = sorted(alt_vector)
        same = sum(
            1
            for t, v in zip(tract_ids, alt_vector)
            if equity_service.risk_level(
                100.0 - equity_service._percentile_of(alt_sorted, v)
            )
            == baseline_levels[t]
        )
        stabilities.append(same / len(tract_ids) * 100)

    quality = data_understanding(db, year)["quality"]
    projected = (
        median * (POPULATION_TARGET / population) if population else None
    )

    result = {
        "phase": "evaluation",
        "year": year,
        "executed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "scale": {
            "tracts": len(tract_ids),
            "indicators": len(catalog),
            "population": int(population),
        },
        "latency": {
            "repeats": repeats,
            "median_seconds": round(median, 4),
            "p95_seconds": round(p95, 4),
            "min_seconds": round(timings[0], 4),
            "max_seconds": round(timings[-1], 4),
            "seconds_per_100k_inhabitants": round(per_capita * 100_000, 4),
            "projected_at_target_population": round(projected, 4)
            if projected is not None
            else None,
            "target_seconds": LATENCY_TARGET_SECONDS,
            "target_population": POPULATION_TARGET,
            "meets_h3": bool(projected is not None and projected < LATENCY_TARGET_SECONDS),
        },
        "risk_distribution": distribution,
        "sensitivity": {
            "runs": sensitivity_runs,
            "perturbation": perturbation,
            "spearman_mean": round(statistics.fmean(correlations), 4)
            if correlations
            else None,
            "spearman_min": round(min(correlations), 4) if correlations else None,
            "risk_level_stability_pct": round(statistics.fmean(stabilities), 2)
            if stabilities
            else None,
        },
        "data_quality": quality,
        "process_review": [
            "El compuesto se calcula en O(n · k) con percentil por bisección: la "
            "distribución global se precomputa una sola vez por ejecución.",
            "La normalización se indexa por tract_id, no por el valor del indicador.",
            "La recomputación es idempotente (restricción uq_equity_tract_year_type).",
        ],
        "limitations": [
            "Evaluación formativa sobre el conjunto disponible; no valida "
            "epidemiológicamente los índices.",
            "CDC PLACES son estimaciones de área pequeña basadas en modelos; la "
            "validación externa requiere contrastar con registros clínicos.",
        ],
    }
    os.makedirs(os.path.dirname(EVALUATION_PATH), exist_ok=True)
    with open(EVALUATION_PATH, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    return result


# --------------------------------------------------------------------------- #
# Fase VI — Despliegue
# --------------------------------------------------------------------------- #

REPORT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "generated_reports")
)


def deployment(db: Session, limit: int = 10) -> Dict[str, Any]:
    db_version = None
    postgis_version = None
    try:
        db_version = db.execute(text("SELECT version()")).scalar()
    except SQLAlchemyError:
        db.rollback()
    try:
        postgis_version = db.execute(text("SELECT PostGIS_Version()")).scalar()
    except SQLAlchemyError:
        db.rollback()

    reports = []
    if os.path.isdir(REPORT_DIR):
        for name in sorted(os.listdir(REPORT_DIR), reverse=True):
            path = os.path.join(REPORT_DIR, name)
            if os.path.isfile(path) and not name.startswith("."):
                reports.append(
                    {
                        "file": name,
                        "format": os.path.splitext(name)[1].lstrip(".").upper(),
                        "size_kb": round(os.path.getsize(path) / 1024, 1),
                        "modified": datetime.utcfromtimestamp(
                            os.path.getmtime(path)
                        ).isoformat(timespec="seconds")
                        + "Z",
                    }
                )
    by_format: Dict[str, int] = {}
    for r in reports:
        by_format[r["format"]] = by_format.get(r["format"], 0) + 1

    audit_rows = (
        db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit).all()
    )
    audit = [
        {
            "id": a.id,
            "username": a.username,
            "action": a.action,
            "resource": a.resource,
            "detail": a.detail,
            "created_at": a.created_at.isoformat(timespec="seconds") + "Z"
            if a.created_at
            else None,
        }
        for a in audit_rows
    ]

    driver = settings.DATABASE_URL.split("://", 1)[0] if settings.DATABASE_URL else None

    return {
        "phase": "deployment",
        "services": [
            {
                "name": "API",
                "detail": f"{settings.APP_NAME} v{settings.APP_VERSION} · FastAPI",
                "runtime": f"Python {platform.python_version()}",
                "status": "running",
            },
            {
                "name": "Base de datos",
                "detail": " · ".join(
                    p for p in [(db_version or "PostgreSQL").split(",")[0],
                                f"PostGIS {postgis_version}" if postgis_version else None]
                    if p
                ),
                "runtime": driver,
                "status": "connected" if db_version else "unknown",
            },
            _ml_service_row(),
            {
                "name": "Frontend",
                "detail": "React 18 · TypeScript · Vite · Three.js / R3F",
                "runtime": "Node · Vite dev server",
                "status": "external",
            },
        ],
        "governance": {
            "roles": db.query(func.count(Role.id)).scalar() or 0,
            "permissions": db.query(func.count(Permission.id)).scalar() or 0,
            "users": db.query(func.count(User.id)).scalar() or 0,
            "audit_events": db.query(func.count(AuditLog.id)).scalar() or 0,
        },
        "reports": {
            "directory": REPORT_DIR,
            "total": len(reports),
            "by_format": by_format,
            "recent": reports[:limit],
        },
        "monitoring": {
            "alert_rules": db.query(func.count(AlertRule.id)).scalar() or 0,
            "alerts_open": db.query(func.count(Alert.id))
            .filter(Alert.status == "open")
            .scalar()
            or 0,
            "alerts_total": db.query(func.count(Alert.id)).scalar() or 0,
        },
        "recent_audit": audit,
        "feedback_loop": [
            "Las alertas señalan deriva de los indicadores y disparan revisión de umbrales.",
            "La bitácora de auditoría documenta quién alteró pesos, umbrales o datos.",
            "La recomputación programada devuelve el proyecto a la Fase I.",
        ],
        "python": sys.version.split()[0],
    }


# --------------------------------------------------------------------------- #
# Ejecución encadenada de las fases III → V
# --------------------------------------------------------------------------- #

def run_pipeline(
    db: Session,
    year: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None,
    repeats: int = 5,
    train_ml: bool = True,
) -> Dict[str, Any]:
    """Ejecuta preparación → modelado (índice + ML) → evaluación → despliegue."""
    year = year or _default_year(db)
    steps = []

    t0 = time.perf_counter()
    prep = data_preparation(db, year, weights)
    steps.append(
        {
            "phase": "data-preparation",
            "seconds": round(time.perf_counter() - t0, 4),
            "summary": f"{len(prep['indicators'])} indicadores · "
            f"{prep['coverage']['tracts_with_data']} tractos con datos",
        }
    )

    t0 = time.perf_counter()
    created = equity_service.compute_equity_indexes(db, year, weights)
    steps.append(
        {
            "phase": "modeling",
            "seconds": round(time.perf_counter() - t0, 4),
            "summary": f"{created} índices persistidos",
        }
    )

    ml_metrics = None
    if train_ml:
        t0 = time.perf_counter()
        current = ml_service.status()
        target = (current.get("target") or {}).get("code") or ml_service.DEFAULT_TARGET
        try:
            ml_metrics = ml_service.train(db, target=target)
            summary = (
                f"4 modelos entrenados · mejor {ml_metrics['best_model_name']} "
                f"(F1 CV {ml_metrics['models'][0]['cv_f1_mean']})"
            )
        except ValueError as e:
            summary = f"ML no entrenado: {e}"
        steps.append(
            {"phase": "modeling", "seconds": round(time.perf_counter() - t0, 4), "summary": summary}
        )

    t0 = time.perf_counter()
    evaluation = run_evaluation(db, year, repeats=repeats)
    steps.append(
        {
            "phase": "evaluation",
            "seconds": round(time.perf_counter() - t0, 4),
            "summary": f"latencia mediana {evaluation.get('latency', {}).get('median_seconds')} s",
        }
    )

    if ml_metrics:
        steps.append(
            {
                "phase": "deployment",
                "seconds": 0.0,
                "summary": f"{ml_metrics['best_model_name']} desplegado en best_model.joblib "
                "· POST /api/ml/predict",
            }
        )

    return {
        "year": year,
        "created_indexes": created,
        "ml_best_model": ml_metrics["best_model_name"] if ml_metrics else None,
        "steps": steps,
        "evaluation": evaluation,
        "total_seconds": round(sum(s["seconds"] for s in steps), 4),
    }
