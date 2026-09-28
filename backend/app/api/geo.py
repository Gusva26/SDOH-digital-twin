from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from shapely import wkb, wkt
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.geo import CensusTract, County, Hospital, HospitalCatchment
from app.models.user import User

router = APIRouter(prefix="/geo", tags=["Geo"])


def _geom_to_geojson(geom_obj):
    """Convert a GeoAlchemy/SQLAlchemy geom value to a GeoJSON dict."""
    if geom_obj is None:
        return None
    try:
        data = getattr(geom_obj, "data", None) or geom_obj
        if isinstance(data, str):
            return wkt.loads(data).__geo_interface__
        return wkb.loads(bytes(data)).__geo_interface__
    except Exception:
        return None


@router.get("/tracts")
def list_tracts(
    state: Optional[str] = None,
    county: Optional[str] = None,
    limit: int = Query(100, le=5000),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    q = db.query(CensusTract)
    if state:
        q = q.filter(CensusTract.state_fips == state)
    if county:
        q = q.filter(CensusTract.county_fips == county)
    rows = q.limit(limit).all()
    return [
        {
            "geoid": r.geoid,
            "name": r.name,
            "state_fips": r.state_fips,
            "county_fips": r.county_fips,
            "total_population": r.total_population,
        }
        for r in rows
    ]


@router.get("/tracts/geojson/{hospital_id}")
def tracts_geojson(
    hospital_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Return census tracts within the default catchment of a hospital as GeoJSON."""
    catchments = (
        db.query(HospitalCatchment)
        .filter(HospitalCatchment.hospital_id == hospital_id)
        .all()
    )
    tract_ids = []
    for c in catchments:
        tract_ids += [m.tract_id for m in c.memberships]
    if not tract_ids:
        raise HTTPException(status_code=404, detail="Sin catchment definido")
    tracts = db.query(CensusTract).filter(CensusTract.id.in_(tract_ids)).all()
    features = []
    for t in tracts:
        geom = _geom_to_geojson(t.geom)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": t.id,
                    "geoid": t.geoid,
                    "name": t.name,
                    "population": t.total_population,
                },
                "geometry": geom,
            }
        )
    return {"type": "FeatureCollection", "features": features}


@router.get("/catchments")
def list_catchments(
    hospital_id: Optional[int] = None,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    q = db.query(HospitalCatchment)
    if hospital_id:
        q = q.filter(HospitalCatchment.hospital_id == hospital_id)
    return [
        {
            "id": c.id,
            "hospital_id": c.hospital_id,
            "name": c.name,
            "catchment_type": c.catchment_type,
            "radius_km": c.radius_km,
            "tract_count": len(c.memberships),
        }
        for c in q.all()
    ]


@router.get("/health-system/stats")
def health_system_stats(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    tract_count = db.query(CensusTract).count()
    county_count = db.query(County).count()
    hospital_count = db.query(Hospital).count()
    catchment_count = db.query(HospitalCatchment).count()
    return {
        "tracts": tract_count,
        "counties": county_count,
        "hospitals": hospital_count,
        "catchments": catchment_count,
    }
