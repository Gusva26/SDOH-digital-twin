"""Endpoints de la metodología CRISP-DM aplicada al gemelo digital SP-5."""

from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_permission
from app.core.database import get_db
from app.models.user import User
from app.services import crispdm_service, ml_evaluation
from app.services.audit_service import audit

router = APIRouter(prefix="/crispdm", tags=["CRISP-DM"])


@router.get("/phases")
def list_phases(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Las seis fases con su estado de avance según los artefactos existentes."""
    return crispdm_service.phase_overview(db)


@router.get("/business-understanding")
def business_understanding(
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("sdoh:read")),
):
    """Fase I: objetivos, hipótesis, criterios de éxito, actores y restricciones."""
    return crispdm_service.business_understanding(db)


@router.get("/data-understanding")
def data_understanding(
    year: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("sdoh:read")),
):
    """Fase II: fuentes, cobertura, perfilado y calidad de los datos."""
    return crispdm_service.data_understanding(db, year)


@router.get("/data-preparation")
def data_preparation(
    year: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("sdoh:read")),
):
    """Fase III: normalización, inversión de riesgo, pesos y cobertura resultante."""
    return crispdm_service.data_preparation(db, year)


@router.get("/modeling")
def modeling(
    year: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("sdoh:read")),
):
    """Fase IV: técnica, fórmula, pesos, umbrales de riesgo y supuestos."""
    return crispdm_service.modeling(db, year)


@router.get("/evaluation")
def evaluation(
    year: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("sdoh:read")),
):
    """Fase V: criterios de evaluación y estado actual (lectura, sin banco de pruebas)."""
    return crispdm_service.evaluation_summary(db, year)


@router.post("/evaluation/run")
def run_evaluation(
    year: Optional[int] = Query(None),
    repeats: int = Query(5, ge=1, le=30),
    sensitivity_runs: int = Query(20, ge=0, le=200),
    perturbation: float = Query(0.25, gt=0, le=1),
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("sdoh:compute")),
):
    """Fase V: ejecuta latencia (H3), sensibilidad de pesos y calidad de datos."""
    result = crispdm_service.run_evaluation(
        db,
        year=year,
        repeats=repeats,
        sensitivity_runs=sensitivity_runs,
        perturbation=perturbation,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    audit(
        db,
        user,
        "EVALUATE",
        "crispdm",
        None,
        f"year={result['year']} mediana={result['latency']['median_seconds']}s",
    )
    return result


def _with_target_index(payload: dict) -> dict:
    """Añade el índice de objetivos para que el selector de la UI se alimente solo."""
    from app.services import ml_service

    payload["available_targets"] = ml_service.trained_targets()
    payload["targets"] = [
        t for t in ml_service.TARGETS if t in payload["available_targets"]
    ]
    payload["evaluated_targets"] = sorted(
        code for code in ml_service.TARGETS if ml_evaluation.load(code) is not None
    )
    return payload


@router.get("/evaluation/ml")
def ml_evaluation_read(
    target: Optional[str] = Query(None),
    _: User = Depends(require_permission("sdoh:read")),
):
    """Fase V: lee la última evaluación ML persistida, o un resumen si no se ha ejecutado."""
    try:
        payload = ml_evaluation.load(target) or {
            "available": False,
            "target": target,
            "executed_at": None,
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _with_target_index(payload)


@router.post("/evaluation/ml/run")
def ml_evaluation_run(
    target: Optional[str] = Query(None),
    year: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("sdoh:compute")),
):
    """Fase V: ejecuta las cuatro evaluaciones ML reales y persiste el resultado.

    Selección del modelo con prueba de separabilidad, parity plot y residuales,
    hipótesis H1/H2/H3 y descomposición de varianza. Tarda alrededor de 30 s.
    """
    try:
        result = ml_evaluation.run_full(target, year)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    verdicts = {h["code"]: h["verdict"] for h in result["hypotheses"]["hypotheses"]}
    audit(
        db,
        user,
        "EVALUATE",
        "crispdm",
        None,
        f"ml target={result['target']} " + " ".join(f"{k}={v}" for k, v in verdicts.items()),
    )
    return _with_target_index(result)


@router.get("/deployment")
def deployment(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("sdoh:read")),
):
    """Fase VI: servicios, gobierno, reportes, monitoreo y bitácora de auditoría."""
    return crispdm_service.deployment(db, limit)
@router.post("/pipeline/run")
def run_pipeline(
    year: Optional[int] = Query(None),
    repeats: int = Query(5, ge=1, le=30),
    train_ml: bool = Query(True),
    weights: Optional[dict] = Body(None),
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("sdoh:compute")),
):
    """Ejecuta encadenadas las fases III → IV (índice + ML) → V → VI."""
    result = crispdm_service.run_pipeline(
        db, year=year, weights=weights, repeats=repeats, train_ml=train_ml
    )
    audit(
        db,
        user,
        "RUN_PIPELINE",
        "crispdm",
        None,
        f"year={result['year']} índices={result['created_indexes']} "
        f"total={result['total_seconds']}s",
    )
    return result
