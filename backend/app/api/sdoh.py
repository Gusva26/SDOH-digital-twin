from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_permission
from app.core.database import get_db
from app.models.geo import (
    CensusTract,
    CatchmentMembership,
    HospitalCatchment,
)
from app.models.sdoh import Alert, AlertRule, EquityIndex, IndicatorCatalog, SDOHIndicator
from app.models.user import User
from app.schemas.sdoh import (
    AlertOut,
    AlertRuleIn,
    AlertRuleOut,
    EquityIndexOut,
    IndicatorCatalogIn,
    IndicatorCatalogOut,
    SDOHValueIn,
)
from app.services.audit_service import audit
from app.services.equity_service import (
    compute_equity_indexes,
    compute_z_scores,
)

router = APIRouter(prefix="/sdoh", tags=["SDOH"])


# --- Catalog ---
@router.get("/catalog", response_model=List[IndicatorCatalogOut])
def list_catalog(
    domain: Optional[str] = None,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    q = db.query(IndicatorCatalog)
    if domain:
        q = q.filter(IndicatorCatalog.domain == domain)
    return q.order_by(IndicatorCatalog.domain, IndicatorCatalog.code).all()


@router.post("/catalog", response_model=IndicatorCatalogOut, status_code=status.HTTP_201_CREATED)
def create_catalog_item(
    payload: IndicatorCatalogIn,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("sdoh:write")),
):
    if db.query(IndicatorCatalog).filter(IndicatorCatalog.code == payload.code).first():
        raise HTTPException(status_code=400, detail="Código ya existe")
    item = IndicatorCatalog(**payload.model_dump())
    db.add(item)
    db.commit()
    db.refresh(item)
    audit(db, user, "CREATE", "catalog", item.id, item.code)
    return item


# --- Values ---
@router.get("/values")
def get_values(
    tract_geoid: Optional[str] = None,
    catalog_code: Optional[str] = None,
    year: Optional[int] = None,
    limit: int = Query(200, le=5000),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    q = db.query(SDOHIndicator)
    if tract_geoid:
        tract = db.query(CensusTract).filter(CensusTract.geoid == tract_geoid).first()
        if not tract:
            raise HTTPException(status_code=404, detail="Tract no encontrado")
        q = q.filter(SDOHIndicator.tract_id == tract.id)
    if catalog_code:
        item = (
            db.query(IndicatorCatalog).filter(IndicatorCatalog.code == catalog_code).first()
        )
        if item:
            q = q.filter(SDOHIndicator.catalog_id == item.id)
    if year:
        q = q.filter(SDOHIndicator.year == year)
    rows = q.limit(limit).all()
    result = []
    for r in rows:
        tract = r.census_tract
        item = r.catalog_item
        result.append(
            {
                "tract_geoid": tract.geoid if tract else None,
                "tract_name": tract.name if tract else None,
                "catalog_code": item.code if item else None,
                "catalog_name": item.name if item else None,
                "year": r.year,
                "value": r.value,
                "low_ci": r.low_ci,
                "high_ci": r.high_ci,
            }
        )
    return result


@router.post("/values", status_code=status.HTTP_201_CREATED)
def upsert_value(
    payloads: List[SDOHValueIn],
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("sdoh:write")),
):
    count = 0
    for p in payloads:
        tract = db.query(CensusTract).filter(CensusTract.geoid == p.tract_geoid).first()
        item = (
            db.query(IndicatorCatalog).filter(IndicatorCatalog.code == p.catalog_code).first()
        )
        if not tract or not item:
            continue
        existing = (
            db.query(SDOHIndicator)
            .filter(
                SDOHIndicator.tract_id == tract.id,
                SDOHIndicator.catalog_id == item.id,
                SDOHIndicator.year == p.year,
            )
            .first()
        )
        if existing:
            existing.value = p.value
            existing.low_ci = p.low_ci
            existing.high_ci = p.high_ci
            existing.sample_size = p.sample_size
        else:
            db.add(
                SDOHIndicator(
                    tract_id=tract.id,
                    catalog_id=item.id,
                    year=p.year,
                    value=p.value,
                    low_ci=p.low_ci,
                    high_ci=p.high_ci,
                    sample_size=p.sample_size,
                )
            )
        count += 1
    db.commit()
    audit(db, user, "UPSERT", "sdoh_values", None, f"{count} valores")
    return {"upserted": count}


# --- Equity indexes ---
@router.get("/equity")
def get_equity(
    index_type: Optional[str] = "composite_equity",
    year: Optional[int] = None,
    risk_level: Optional[str] = None,
    hospital_id: Optional[int] = None,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Equity indexes, optionally restricted to a hospital's catchment area."""
    q = db.query(EquityIndex)
    if index_type:
        q = q.filter(EquityIndex.index_type == index_type)
    if year:
        q = q.filter(EquityIndex.year == year)
    if risk_level:
        q = q.filter(EquityIndex.risk_level == risk_level)
    if hospital_id:
        tract_ids = (
            db.query(CatchmentMembership.tract_id)
            .join(
                HospitalCatchment,
                CatchmentMembership.catchment_id == HospitalCatchment.id,
            )
            .filter(HospitalCatchment.hospital_id == hospital_id)
            .distinct()
        )
        q = q.filter(EquityIndex.tract_id.in_(tract_ids))
    rows = q.all()
    tracts = {
        t.id: t
        for t in db.query(CensusTract)
        .filter(CensusTract.id.in_([r.tract_id for r in rows]))
        .all()
    }
    return [
        {
            "id": r.id,
            "tract_id": r.tract_id,
            "tract_geoid": (
                tracts[r.tract_id].geoid if r.tract_id in tracts else None
            ),
            "tract_name": tracts[r.tract_id].name if r.tract_id in tracts else None,
            "population": (
                tracts[r.tract_id].total_population if r.tract_id in tracts else None
            ),
            "year": r.year,
            "index_type": r.index_type,
            "value": r.value,
            "percentile": r.percentile,
            "risk_level": r.risk_level,
        }
        for r in rows
    ]


@router.post("/equity/compute")
def compute_equity(
    year: int = 2022,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("sdoh:compute")),
):
    created = compute_equity_indexes(db, year)
    audit(db, user, "COMPUTE", "equity_indexes", None, f"year={year} created={created}")
    return {"created": created, "year": year}


# --- Alert rules ---
@router.get("/alert-rules", response_model=List[AlertRuleOut])
def list_alert_rules(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    return db.query(AlertRule).all()


@router.post("/alert-rules", response_model=AlertRuleOut)
def create_alert_rule(
    payload: AlertRuleIn,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("alerts:write")),
):
    rule = AlertRule(**payload.model_dump())
    db.add(rule)
    db.commit()
    db.refresh(rule)
    audit(db, user, "CREATE", "alert_rule", rule.id, rule.name)
    return rule


# --- Alerts ---
@router.get("/alerts", response_model=List[AlertOut])
def list_alerts(
    status_filter: Optional[str] = Query(None, alias="status"),
    severity: Optional[str] = None,
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    q = db.query(Alert)
    if status_filter:
        q = q.filter(Alert.status == status_filter)
    if severity:
        q = q.filter(Alert.severity == severity)
    return q.order_by(Alert.id.desc()).limit(limit).all()


@router.post("/alerts/generate")
def generate_alerts(
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("alerts:write")),
):
    rules = db.query(AlertRule).filter(AlertRule.enabled == "true").all()
    generated = 0
    from datetime import datetime

    for rule in rules:
        q = db.query(SDOHIndicator)
        if rule.catalog_id:
            q = q.filter(SDOHIndicator.catalog_id == rule.catalog_id)
        for val in q.all():
            trigger = False
            if rule.comparison == "gt":
                trigger = val.value > rule.threshold
            elif rule.comparison == "lt":
                trigger = val.value < rule.threshold
            elif rule.comparison == "gte":
                trigger = val.value >= rule.threshold
            elif rule.comparison == "lte":
                trigger = val.value <= rule.threshold
            if trigger:
                db.add(
                    Alert(
                        rule_id=rule.id,
                        tract_id=val.tract_id,
                        indicator_code=val.catalog_item.code if val.catalog_item else None,
                        indicator_name=val.catalog_item.name if val.catalog_item else None,
                        observed_value=val.value,
                        severity=rule.severity,
                        status="open",
                        message=f"{rule.name}: {val.value} {rule.comparison} {rule.threshold}",
                        created_at=datetime.utcnow(),
                    )
                )
                generated += 1
    db.commit()
    audit(db, user, "GENERATE", "alerts", None, f"{generated} alertas")
    return {"generated": generated}
