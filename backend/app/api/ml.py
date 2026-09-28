"""Endpoints del modelo predictivo: entrenamiento comparativo, métricas, predicción y reportes.

Hay un modelo por objetivo (resultado de salud). `target` es opcional en todas las
rutas de lectura: si se omite, se usa el objetivo activo (el que consume el Dashboard 3D).
"""

from typing import Dict, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import require_permission
from app.core.database import get_db
from app.models.user import User
from app.schemas.report import ReportOut
from app.services import ml_report_service, ml_service
from app.services.audit_service import audit

router = APIRouter(prefix="/ml", tags=["Machine Learning"])


class PredictRequest(BaseModel):
    features: Dict[str, Optional[float]] = {}
    target: Optional[str] = None


def _check_target(target: Optional[str]) -> None:
    if target is not None and target not in ml_service.TARGETS:
        raise HTTPException(status_code=400, detail=f"Objetivo no válido: {target}")


@router.get("/status")
def ml_status(
    target: Optional[str] = Query(None),
    _: User = Depends(require_permission("sdoh:read")),
):
    """Métricas de los 4 modelos del objetivo (o `trained: false`) y objetivos entrenados."""
    _check_target(target)
    return ml_service.status(target)


@router.post("/train")
def ml_train(
    target: str = Query(ml_service.DEFAULT_TARGET),
    year: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("sdoh:compute")),
):
    """Entrena y compara los 4 modelos; despliega el mejor y lo deja como activo."""
    try:
        metrics = ml_service.train(db, target=target, year=year)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    audit(
        db,
        user,
        "TRAIN_MODEL",
        "ml",
        None,
        f"target={target} mejor={metrics['best_model']} "
        f"F1cv={metrics['models'][0]['cv_f1_mean']} {metrics['total_seconds']}s",
    )
    return metrics


@router.post("/activate")
def ml_activate(
    target: str = Query(...),
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("sdoh:compute")),
):
    """Elige qué modelo entrenado consumen el Dashboard 3D y CRISP-DM."""
    _check_target(target)
    if target not in ml_service.trained_targets():
        raise HTTPException(status_code=409, detail=f"No hay modelo entrenado para {target}.")
    ml_service.activate(target)
    audit(db, user, "ACTIVATE_MODEL", "ml", None, f"target={target}")
    return ml_service.status(target)


@router.post("/predict")
def ml_predict(
    body: PredictRequest = Body(...),
    _: User = Depends(require_permission("sdoh:read")),
):
    """Predicción detallada de un escenario: nivel, % estimado, percentil y factores."""
    _check_target(body.target)
    try:
        return ml_service.predict(body.features, body.target)
    except LookupError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/predictions")
def ml_predictions(
    year: Optional[int] = Query(None),
    target: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("sdoh:read")),
):
    """Predicciones del modelo para cada census tract."""
    _check_target(target)
    try:
        return ml_service.predict_tracts(db, year, target)
    except LookupError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/report", response_model=ReportOut)
def ml_model_report(
    target: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("reports:generate")),
):
    """PDF con la comparación de los 4 modelos del objetivo; se descarga en /reports/download/{id}."""
    _check_target(target)
    target = target or ml_service.active_target()
    try:
        report = ml_report_service.model_report(db, user, target)
    except LookupError as e:
        raise HTTPException(status_code=409, detail=str(e))
    audit(db, user, "GENERATE", "report", report.id, f"{report.filename} (pdf)")
    return report


@router.post("/predict/report", response_model=ReportOut)
def ml_scenario_report(
    body: PredictRequest = Body(...),
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("reports:generate")),
):
    """PDF de un escenario simulado con su interpretación y factores explicativos."""
    _check_target(body.target)
    target = body.target or ml_service.active_target()
    try:
        report = ml_report_service.scenario_report(db, user, target, body.features)
    except LookupError as e:
        raise HTTPException(status_code=409, detail=str(e))
    audit(db, user, "GENERATE", "report", report.id, f"{report.filename} (pdf)")
    return report
