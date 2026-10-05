"""
ETL pipeline with REAL public datasets (no API key required).

Data source (default):
  CDC PLACES - Local Data for Better Health, Census Tract Data, 2025 release
  (https://data.cdc.gov, dataset ``cwsq-ngmh``), public domain, accessed via
  the Socrata Open Data API.

Tract geometries:
  U.S. Census Bureau TIGERweb (``tigerWMS_Current`` MapServer, layer 8
  "Census Tracts").

The pipeline:
  1. Downloads CDC PLACES estimates (values + 95% CI + population) per census
     tract for the configured counties (default New York County, NY and Cook
     County, IL). Both reported years are loaded (2023: 35 measures, 2022: 5
     measures reported every other year).
  2. Fetches real census-tract polygons from TIGERweb.
  3. Registers the SDOH indicator catalog mapped from the CDC measures.
  4. Persists counties, census tracts (with geometry + center), SDOH values.
  5. Assigns census tracts to hospital catchments by distance.
  6. Computes composite equity indexes (weighted normalized composite) for every
     loaded year (app.services.equity_service).
  7. Seeds alert rules and generates alerts for the loaded indicators.

Usage (inside the container):
  python -m app.scripts.etl_pipeline --public                  # default
  python -m app.scripts.etl_pipeline --public --year 2023      # single year
  python -m app.scripts.etl_pipeline --public --counties 17031,36061 --radius 12
  python -m app.scripts.etl_pipeline --public --offline        # reuse cached CSV
  python -m app.scripts.etl_pipeline --cdc backend/data/raw/places.csv
  python -m app.scripts.etl_pipeline --demo                    # synthetic seed
"""

import argparse
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import requests
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely import wkb

from sqlalchemy import func
from app.core.database import SessionLocal
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

# ---------------------------------------------------------------------------
# Public data sources
# ---------------------------------------------------------------------------

CDC_PLACES_BASE = "https://data.cdc.gov/resource/cwsq-ngmh.json"
CDC_PLACES_DATASET_ID = "cwsq-ngmh"  # PLACES 2025 release, census tract level
TIGER_TRACTS_URL = (
    "https://tigerweb.geo.census.gov/arcgis/rest/services/"
    "TIGERweb/tigerWMS_Current/MapServer/8/query"
)

#: Year with the largest measure coverage in the 2025 release (BRFSS 2023).
PREFERRED_YEAR = 2023

#: Descriptive info about each county loaded from the public dataset.
DEFAULT_COUNTIES = [
    {
        "geoid": "36061",
        "state_fips": "36",
        "county_fips": "061",
        "name": "New York County",
        "state_name": "New York",
    },
    {
        "geoid": "17031",
        "state_fips": "17",
        "county_fips": "031",
        "name": "Cook County",
        "state_name": "Illinois",
    },
]

#: Hospitals (kept stable across imports) used to build catchments.
HOSPITALS = [
    {
        "name": "Metropolitan General Hospital",
        "slug": "metropolitan-general",
        "city": "New York",
        "state": "NY",
        "zip_code": "10001",
        "latitude": 40.75,
        "longitude": -73.98,
    },
    {
        "name": "Lakefront Community Medical Center",
        "slug": "lakefront-community",
        "city": "Chicago",
        "state": "IL",
        "zip_code": "60601",
        "latitude": 41.88,
        "longitude": -87.62,
    },
]

#: (CDC measure name, catalog code, catalog name, domain, higher_is_better,
#:  alert threshold). Threshold ``None`` means no alert rule is created.
CDC_MEASURE_MAP: List[Tuple[str, str, str, str, int, Optional[float]]] = [
    ("Obesity among adults", "pct_obesity", "Obesidad (%)", "Salud crónica", 0, 35),
    ("Diagnosed diabetes among adults", "pct_diabetes", "Diabetes (%)", "Salud crónica", 0, 15),
    ("Current lack of health insurance among adults aged 18-64 years", "pct_no_health_insurance", "Sin seguro médico (%)", "Acceso a salud", 0, 15),
    ("No leisure-time physical activity among adults", "pct_inactivity", "Inactividad física (%)", "Comportamiento", 0, 30),
    ("Food insecurity in the past 12 months among adults", "pct_food_insecurity", "Inseguridad alimentaria (%)", "Alimentación", 0, 18),
    ("Current cigarette smoking among adults", "pct_smoking", "Tabaquismo (%)", "Comportamiento", 0, 18),
    ("Depression among adults", "pct_depression", "Depresión (%)", "Salud mental", 0, 22),
    ("Frequent mental distress among adults", "pct_mental_distress", "Malestar mental frecuente (%)", "Salud mental", 0, 15),
    ("Short sleep duration among adults", "pct_short_sleep", "Sueño insuficiente (%)", "Comportamiento", 0, 40),
    ("Fair or poor self-rated health status among adults", "pct_fair_poor_health", "Salud autorreportada regular o mala (%)", "Salud", 0, 20),
    ("Lack of social and emotional support among adults", "pct_lack_social_support", "Falta de apoyo social (%)", "Social", 0, 25),
    ("Loneliness among adults", "pct_loneliness", "Soledad (%)", "Social", 0, 25),
    ("Housing insecurity in the past 12 months among adults", "pct_housing_insecurity", "Inseguridad de vivienda (%)", "Vivienda", 0, 12),
    ("Utility services shut-off threat in the past 12 months among adults", "pct_utility_shutoff", "Amenaza de corte de servicios (%)", "Vivienda", 0, 12),
    ("Lack of reliable transportation in the past 12 months among adults", "pct_no_transportation", "Falta de transporte confiable (%)", "Transporte", 0, 8),
    ("Received food stamps in the past 12 months among adults", "pct_food_stamps", "Recibe ayuda alimentaria (SNAP) (%)", "Ingreso", 0, 20),
    ("Binge drinking among adults", "pct_binge_drinking", "Consumo excesivo de alcohol (%)", "Comportamiento", 0, 18),
    ("Chronic obstructive pulmonary disease among adults", "pct_copd", "EPOC (%)", "Salud crónica", 0, 10),
    ("Coronary heart disease among adults", "pct_chd", "Enfermedad coronaria (%)", "Salud crónica", 0, 10),
    ("Stroke among adults", "pct_stroke", "Accidente cerebrovascular (%)", "Salud crónica", 0, 5),
    ("Current asthma among adults", "pct_asthma", "Asma (%)", "Salud crónica", 0, 12),
    ("Any disability among adults", "pct_any_disability", "Discapacidad (%)", "Discapacidad", 0, 20),
    ("High blood pressure among adults", "pct_high_bp", "Hipertensión (%)", "Salud crónica", 0, 35),
    ("Visits to doctor for routine checkup within the past year among adults", "pct_routine_checkup", "Chequeo médico anual (%)", "Prevención", 1, None),
    ("Cholesterol screening among adults", "pct_cholesterol_screening", "Cribado de colesterol (%)", "Prevención", 1, None),
    ("Colorectal cancer screening among adults aged 45-75 years", "pct_colorectal_screening", "Cribado de cáncer colorrectal (%)", "Prevención", 1, None),
    ("Visited dentist or dental clinic in the past year among adults", "pct_dental_visit", "Visita al dentista en el último año (%)", "Prevención", 1, None),
]

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

SELECT_COLUMNS = (
    "year,locationid,locationname,countyfips,statedesc,category,measure,"
    "data_value,low_confidence_limit,high_confidence_limit,"
    "totalpopulation,totalpop18plus,geolocation"
)


def _norm(s: str) -> str:
    """Normalize a measure name so dash variants do not break matching."""
    return re.sub(r"\s+", " ", (s or "").lower().replace("\u2013", "-").replace("\u2014", "-")).strip()


def _to_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _point_wkb(lon: float, lat: float):
    from geoalchemy2.elements import WKBElement

    from shapely.wkb import dumps

    return WKBElement(dumps(Point(lon, lat), srid=4326), srid=4326)


def _center_to_point(center) -> Optional[Point]:
    """Convert a stored geometry / WKBElement to a shapely Point."""
    try:
        from geoalchemy2.shape import to_shape
        shape = to_shape(center)
        if isinstance(shape, Point):
            return shape
        return Point(shape.x, shape.y)
    except Exception:
        try:
            from shapely import wkb
            if hasattr(center, "data"):
                data = bytes.fromhex(str(center.data)) if isinstance(center.data, str) else bytes(center.data)
                return wkb.loads(data)
            return None
        except Exception:
            return None


# ---------------------------------------------------------------------------
# 1. Download CDC PLACES (Socrata Open Data API, no API key required)
# ---------------------------------------------------------------------------

def download_cdc_places(
    county_fips: List[str],
    years: Optional[List[int]] = None,
    out_csv: Optional[str] = None,
) -> List[Dict]:
    """Download PLACES 2025 census-tract records for the given counties.

    Returns a list of records. When ``out_csv`` is provided the raw data is
    also cached as CSV for reproducibility / offline runs.
    """
    where = f"countyfips in ({','.join(repr(c) for c in county_fips)})"
    if years:
        where += f" AND year in ({','.join(repr(str(y)) for y in years)})"
    records: List[Dict] = []
    offset = 0
    page_size = 10000
    while True:
        params = {
            "$where": where,
            "$select": SELECT_COLUMNS,
            "$order": "locationid, year, measure",
            "$limit": page_size,
            "$offset": offset,
        }
        resp = requests.get(CDC_PLACES_BASE, params=params, timeout=120)
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break
        records.extend(page)
        offset += len(page)
        if len(page) < page_size:
            break
    if out_csv:
        try:
            import pandas as pd

            pd.DataFrame(records).to_csv(out_csv, index=False)
        except Exception:  # cache is optional
            pass
    return records


# ---------------------------------------------------------------------------
# 2. Fetch real tract geometries from the U.S. Census TIGERweb service
# ---------------------------------------------------------------------------

def _multipolygon_from_rings(rings) -> Optional[MultiPolygon]:
    """Build a corrected MultiPolygon from ESRI-style ring arrays.

    Each element of ``rings`` is a ring (closed list of [x, y]). Holes and
    disjoint parts are handled by unioning the individual ring polygons.
    """
    pieces = []
    for ring in rings or []:
        if not ring or len(ring) < 4:
            continue
        try:
            poly = Polygon(ring)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area <= 0:
                continue
            pieces.append(poly)
        except Exception:
            continue
    if not pieces:
        return None
    try:
        merged = unary_union(pieces)
    except Exception:
        return None

    if merged.geom_type == "Polygon":
        polys = [merged]
    elif merged.geom_type == "MultiPolygon":
        polys = list(merged.geoms)
    elif merged.geom_type == "GeometryCollection":
        polys = [g for g in merged.geoms if g.geom_type == "Polygon"]
    else:
        return None

    if not polys:
        return None

    try:
        oriented = [orient(p, sign=1.0) for p in polys if not p.is_empty and p.area > 0]
        if not oriented:
            return None
        from shapely import multipolygons
        return multipolygons(oriented)
    except Exception:
        try:
            return MultiPolygon(oriented)
        except Exception:
            return None


_WKB_CACHE: Dict[str, object] = {}


def _wkb_from_geom(geom) -> object:
    """Memoized WKBElement wrapper (hex EWKB) for storing in PostGIS."""
    from geoalchemy2.elements import WKBElement

    from shapely.wkb import dumps

    key = geom.wkt
    if key not in _WKB_CACHE:
        _WKB_CACHE[key] = WKBElement(dumps(geom, srid=4326), srid=4326)
    return _WKB_CACHE[key]


def fetch_tract_geometries(
    counties: List[Dict], page_size: int = 500
) -> Dict[str, Dict[str, object]]:
    """Fetch {geoid: {geom: wkb, centroid: wkb}} from TIGERweb."""
    result: Dict[str, Dict[str, object]] = {}

    for county in counties:
        where = (
            f"STATE='{county['state_fips']}' AND COUNTY='{county['county_fips']}'"
        )
        offset = 0
        while True:
            params = {
                "where": where,
                "outFields": "GEOID",
                "returnGeometry": "true",
                "outSR": "4326",
                "f": "json",
                "resultRecordCount": page_size,
                "resultOffset": offset,
            }
            resp = requests.get(TIGER_TRACTS_URL, params=params, timeout=120)
            resp.raise_for_status()
            payload = resp.json()
            features = payload.get("features", [])
            if not features:
                break
            for feat in features:
                geoid = str(feat["attributes"].get("GEOID", "")).strip()
                if not geoid:
                    continue
                geom = _multipolygon_from_rings(feat["geometry"].get("rings", []))
                if geom is None:
                    result[geoid] = {"geom": None, "centroid": None}
                    continue
                result[geoid] = {
                    "geom": _wkb_from_geom(geom),
                    "centroid": _point_wkb(*geom.centroid.coords[0]),
                }
            offset += len(features)
            if len(features) < page_size:
                break
    return result


# ---------------------------------------------------------------------------
# 3-4. Catalog + geographies + indicator values
# ---------------------------------------------------------------------------

def _reset_public_tables(db):
    """Drop SDOH / geo data while preserving users, roles and permissions."""
    for model in (
        Alert,
        AlertRule,
        EquityIndex,
        SDOHIndicator,
        CatchmentMembership,
        HospitalCatchment,
        Hospital,
        CensusTract,
        County,
        IndicatorCatalog,
    ):
        db.query(model).delete(synchronize_session=False)
    db.commit()


def seed_catalog(db) -> Dict[str, IndicatorCatalog]:
    for _, code, name, domain, higher_is_better, threshold in CDC_MEASURE_MAP:
        db.add(
            IndicatorCatalog(
                code=code,
                name=name,
                domain=domain,
                source="CDC-PLACES",
                higher_is_better=higher_is_better,
                threshold_alert=threshold,
            )
        )
    db.commit()
    return {c.code: c for c in db.query(IndicatorCatalog).all()}


def _parse_geolocation(geo) -> Optional[Tuple[float, float]]:
    if isinstance(geo, str):
        try:
            import json

            geo = json.loads(geo)
        except Exception:
            return None
    if isinstance(geo, dict):
        coords = geo.get("coordinates")
        if coords and len(coords) >= 2:
            try:
                return (float(coords[0]), float(coords[1]))
            except (TypeError, ValueError):
                return None
    return None


def build_record_map(records: List[Dict]) -> Dict[str, Dict[str, object]]:
    """Normalize PLACES records into ``{geoid: {meta, values}}``.

    ``values`` maps ``year`` -> ``code`` -> indicator dict. Only measures in
    ``CDC_MEASURE_MAP`` are kept; the first non-null value wins per
    (tract, measure, year).
    """
    measure_keys = {_norm(m[0]): m for m in CDC_MEASURE_MAP}
    by_tract: Dict[str, Dict[str, object]] = {}
    for rec in records:
        geoid = str(rec.get("locationid", "")).strip()
        if not geoid:
            continue
        year = _to_int(rec.get("year"))
        measure = _norm(rec.get("measure", ""))
        spec = measure_keys.get(measure)
        value = _to_float(rec.get("data_value"))
        if spec is None or value is None or year is None:
            continue
        tract = by_tract.setdefault(
            geoid,
            {
                "name": str(rec.get("locationname", "")).strip(),
                "county_fips": str(rec.get("countyfips", "")).strip(),
                "state_desc": str(rec.get("statedesc", "")).strip(),
                "total_population": _to_int(rec.get("totalpopulation")),
                "adult_population": _to_int(rec.get("totalpop18plus")),
                "center": None,
                "values": {},
            },
        )
        lonlat = _parse_geolocation(rec.get("geolocation"))
        if lonlat and tract["center"] is None:
            tract["center"] = lonlat
        per_year = tract["values"].setdefault(year, {})
        code = spec[1]
        ind = per_year.setdefault(
            code,
            {
                "catalog_id": None,
                "value": value,
                "low_ci": _to_float(rec.get("low_confidence_limit")),
                "high_ci": _to_float(rec.get("high_confidence_limit")),
            },
        )
        if ind["value"] is None:
            ind["value"] = value
    return by_tract


def ingest_public_records(
    db,
    records: List[Dict],
    counties: Optional[List[Dict]] = None,
    geometries: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, int]:
    """Persist counties, census tracts and SDOH values from CDC PLACES."""
    counties = counties if counties is not None else DEFAULT_COUNTIES
    geometries = geometries or {}
    catalog = seed_catalog(db)

    county_objs = [
        County(
            geoid=c["geoid"],
            name=c["name"],
            state_fips=c["state_fips"],
            county_fips=c["county_fips"],
            state_name=c["state_name"],
        )
        for c in counties
    ]
    db.add_all(county_objs)
    db.commit()
    county_by_fips = {(c.state_fips, c.county_fips): c for c in county_objs}

    by_tract = build_record_map(records)
    tract_count = 0
    indicator_count = 0
    for geoid, data in by_tract.items():
        state_fips = geoid[:2]
        county_fips = geoid[2:5]
        county = county_by_fips.get((state_fips, county_fips))
        if county is None:
            continue
        shape = geometries.get(geoid, {})
        center = shape.get("centroid")
        if center is None and data["center"]:
            center = _point_wkb(data["center"][0], data["center"][1])
        tract = CensusTract(
            geoid=geoid,
            name=data["name"],
            state_fips=state_fips,
            county_fips=county_fips,
            tract_fips=geoid,
            county_id=county.id,
            total_population=data["total_population"],
            land_area=None,
            geom=shape.get("geom"),
            center=center,
        )
        db.add(tract)
        db.flush()
        tract_count += 1
        for year, indicators in data["values"].items():
            for code, ind in indicators.items():
                db.add(
                    SDOHIndicator(
                        tract_id=tract.id,
                        catalog_id=catalog[code].id,
                        year=year,
                        value=ind["value"],
                        low_ci=ind["low_ci"],
                        high_ci=ind["high_ci"],
                    )
                )
                indicator_count += 1
    db.commit()
    return {"counties": len(counties), "tracts": tract_count, "indicators": indicator_count}


# ---------------------------------------------------------------------------
# 5. Hospitals and catchments
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def create_hospitals_and_catchments(db, radius_km: float = 15) -> int:
    memberships = 0
    for spec in HOSPITALS:
        hospital = Hospital(
            name=spec["name"],
            slug=spec["slug"],
            city=spec["city"],
            state=spec["state"],
            zip_code=spec["zip_code"],
            latitude=spec["latitude"],
            longitude=spec["longitude"],
            geom=_point_wkb(spec["longitude"], spec["latitude"]),
        )
        db.add(hospital)
        db.flush()
        catchment = HospitalCatchment(
            hospital_id=hospital.id,
            name=f"{hospital.name} - Catchment",
            catchment_type="radius",
            radius_km=radius_km,
        )
        db.add(catchment)
        db.flush()
        for tract_id, lon, lat in (
            db.query(CensusTract.id, func.st_x(CensusTract.center), func.st_y(CensusTract.center))
            .filter(CensusTract.center.isnot(None))
            .all()
        ):
            if lon is None or lat is None:
                continue
            if _haversine_km(hospital.latitude, hospital.longitude, lat, lon) <= radius_km:
                db.add(
                    CatchmentMembership(
                        catchment_id=catchment.id, tract_id=tract_id
                    )
                )
                memberships += 1
        db.commit()
    return memberships


# ---------------------------------------------------------------------------
# 6-7. Equity indexes and alerts (real computations on loaded data)
# ---------------------------------------------------------------------------

def compute_equity(db, years: Optional[List[int]] = None) -> Dict[int, int]:
    from app.services.equity_service import compute_equity_indexes

    years = years or [
        y for (y,) in db.query(SDOHIndicator.year).distinct().order_by(SDOHIndicator.year).all()
    ]
    return {year: compute_equity_indexes(db, year) for year in years}


def seed_alert_rules_and_generate(db, catalog: Dict[str, IndicatorCatalog]) -> int:
    rules = []
    for item in catalog.values():
        if item.threshold_alert is None:
            continue
        rule = AlertRule(
            name=f"Alerta: {item.name}",
            catalog_id=item.id,
            comparison="gte",
            threshold=item.threshold_alert,
            severity="high",
            enabled="true",
            description=(
                f"Detecta census tracts con {item.name} igual o superior a "
                f"{item.threshold_alert} (fuente CDC PLACES)."
            ),
        )
        db.add(rule)
        rules.append(rule)
    db.commit()

    generated = 0
    for rule in rules:
        for val in (
            db.query(SDOHIndicator)
            .filter(SDOHIndicator.catalog_id == rule.catalog_id)
            .all()
        ):
            if val.value >= rule.threshold:
                db.add(
                    Alert(
                        rule_id=rule.id,
                        tract_id=val.tract_id,
                        indicator_code=val.catalog_item.code if val.catalog_item else None,
                        indicator_name=val.catalog_item.name if val.catalog_item else None,
                        observed_value=val.value,
                        severity=rule.severity,
                        status="open",
                        message=(
                            f"{rule.name}: {val.value} >= {rule.threshold} "
                            f"en tract {val.census_tract.geoid if val.census_tract else ''}"
                        ),
                    )
                )
                generated += 1
    db.commit()
    return generated


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_public_pipeline(
    county_fips: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    radius_km: float = 15,
    counties: Optional[List[Dict]] = None,
    offline: bool = False,
    cache_csv: bool = True,
) -> Dict[str, object]:
    """Full public-dataset pipeline (CDC PLACES + TIGERweb geometries)."""
    counties = counties if counties is not None else DEFAULT_COUNTIES
    county_fips = county_fips if county_fips is not None else [c["geoid"] for c in counties]
    suffix = "-".join(county_fips)

    from app.main import init_default_data

    init_default_data()

    csv_path = os.path.join(RAW_DIR, f"places_cdc_{suffix}.csv")
    if offline and os.path.exists(csv_path):
        import pandas as pd

        records = pd.read_csv(csv_path).to_dict(orient="records")
    else:
        print(f"[ETL] Descargando CDC PLACES 2025 para counties {','.join(county_fips)} ...")
        records = download_cdc_places(
            county_fips, years=years, out_csv=csv_path if cache_csv else None
        )
        print(f"[ETL] {len(records)} registros descargados.")

    print("[ETL] Descargando geometrías de census tracts (TIGERweb) ...")
    try:
        geometries = fetch_tract_geometries(counties)
        print(f"[ETL] {len(geometries)} geometrías obtenidas.")
    except Exception as exc:  # geometry is non fatal
        geometries = {}
        print(f"[ETL] Aviso: no se pudieron obtener geometrías ({exc})")

    db = SessionLocal()
    try:
        _reset_public_tables(db)
        summary = ingest_public_records(
            db, records, counties=counties, geometries=geometries
        )
        memberships = create_hospitals_and_catchments(db, radius_km=radius_km)
        catalog = {c.code: c for c in db.query(IndicatorCatalog).all()}
        equity = compute_equity(db)
        alerts = seed_alert_rules_and_generate(db, catalog)
        summary.update(
            {
                "memberships": memberships,
                "equity_indexes": equity,
                "alerts": alerts,
                "source": f"CDC PLACES 2025 ({CDC_PLACES_DATASET_ID})",
                "geometries": len(geometries),
            }
        )
        print("[ETL] Pipeline público completado:")
        for k, v in summary.items():
            print(f"  - {k}: {v}")
        return summary
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Backwards-compatible helpers (local CSVs)
# ---------------------------------------------------------------------------

def ingest_cdc_places_csv(db, filepath, year=None):
    """Import a local CDC PLACES CSV (compat). Accepts either the Socrata field
    names or the classic ``LocationID/Measure/Data_Value`` ones."""
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas requerido para ETL real")
    df = pd.read_csv(filepath)
    df = df.rename(
        columns={
            "LocationID": "locationid",
            "Measure": "measure",
            "Data_Value": "data_value",
            "Year": "year",
        }
    )
    if year is not None and "year" not in df.columns:
        df["year"] = year
    records = df.to_dict(orient="records")
    catalog = {c.code: c for c in db.query(IndicatorCatalog).all()}
    if not catalog:
        catalog = seed_catalog(db)
    for geoid, data in build_record_map(records).items():
        tract = db.query(CensusTract).filter(CensusTract.geoid == geoid).first()
        if not tract:
            continue
        for yr, indicators in data["values"].items():
            for code, ind in indicators.items():
                item = catalog.get(code)
                if not item:
                    continue
                db.add(
                    SDOHIndicator(
                        tract_id=tract.id,
                        catalog_id=item.id,
                        year=yr,
                        value=ind["value"],
                        low_ci=ind["low_ci"],
                        high_ci=ind["high_ci"],
                    )
                )
    db.commit()
    return len(records)


def run_demo_pipeline():
    """Load the synthetic demo dataset (requires running seed first)."""
    from app.scripts.seed import seed

    seed()


def main():
    parser = argparse.ArgumentParser(description="ETL de datasets públicos SDOH")
    parser.add_argument(
        "--public",
        action="store_true",
        help="Pipeline con dataset público real: CDC PLACES + geometrías TIGERweb (default)",
    )
    parser.add_argument("--cdc", help="Ruta a CSV de CDC PLACES (compat)")
    parser.add_argument("--demo", action="store_true", help="Cargar dataset demo/sintético")
    parser.add_argument(
        "--counties",
        default=None,
        help="County FIPS separados por coma, p. ej. 17031,36061 (default: 17031,36061)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Año de estimaciones (default: todos los años disponibles)",
    )
    parser.add_argument("--radius", type=float, default=15.0, help="Radio de catchment en km")
    parser.add_argument("--offline", action="store_true", help="Reusar CSV cacheado si existe")
    args = parser.parse_args()

    if args.demo:
        run_demo_pipeline()
        return

    if args.cdc:
        db = SessionLocal()
        try:
            n = ingest_cdc_places_csv(db, args.cdc, year=args.year)
            print(f"Importados {n} registros CDC PLACES")
        finally:
            db.close()
        return

    county_fips = [c.strip() for c in args.counties.split(",")] if args.counties else None
    years = [args.year] if args.year else None
    run_public_pipeline(
        county_fips=county_fips,
        years=years,
        radius_km=args.radius,
        offline=args.offline,
    )


if __name__ == "__main__":
    main()