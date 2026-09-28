"""Fase II — Carga del dataset real (CDC PLACES) desde PostGIS o desde un CSV."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import func

from app.core.database import SessionLocal
from app.models.geo import CensusTract, County
from app.models.sdoh import IndicatorCatalog, SDOHIndicator
from app.scripts.etl_pipeline import CDC_MEASURE_MAP
from app.services import ml_service

FEATURES: List[str] = list(ml_service.FEATURES)
TARGETS: List[str] = list(ml_service.TARGETS)
TARGET_INFO: Dict[str, str] = dict(ml_service.TARGET_INFO)

#: Nombre y dominio de cada código según el mapa del ETL (respaldo sin BD).
_ETL_NAMES = {code: name for _m, code, name, _d, _h, _t in CDC_MEASURE_MAP}
_ETL_DOMAINS = {code: dom for _m, code, _n, dom, _h, _t in CDC_MEASURE_MAP}
_MEASURE_TO_CODE = {m.lower(): code for m, code, *_ in CDC_MEASURE_MAP}


@dataclass
class Dataset:
    df: pd.DataFrame
    source: str
    year: Optional[int]
    names: Dict[str, str]
    domains: Dict[str, str]
    notes: List[str] = field(default_factory=list)

    @property
    def features(self) -> List[str]:
        return [c for c in FEATURES if c in self.df.columns and self.df[c].notna().any()]

    @property
    def targets(self) -> List[str]:
        return [c for c in TARGETS if c in self.df.columns and self.df[c].notna().sum() >= 50]

    def label(self, code: str) -> str:
        return self.names.get(code, code)


def available_years() -> List[int]:
    db = SessionLocal()
    try:
        return [y for (y,) in db.query(SDOHIndicator.year).distinct().order_by(SDOHIndicator.year)]
    finally:
        db.close()


def _catalog(db) -> tuple[Dict[str, str], Dict[str, str]]:
    rows = db.query(IndicatorCatalog.code, IndicatorCatalog.name, IndicatorCatalog.domain).all()
    names = {**_ETL_NAMES, **{c: n for c, n, _d in rows}}
    domains = {**_ETL_DOMAINS, **{c: d for c, _n, d in rows if d}}
    return names, domains


def load_from_db(year: Optional[int] = None) -> Dataset:
    """Tabla tract × indicador con los valores reales cargados por el ETL."""
    db = SessionLocal()
    try:
        df = ml_service.build_dataset(db, year)
        names, domains = _catalog(db)
        counties = {g[-5:]: n for g, n in db.query(County.geoid, County.name).all()}
        n_values = db.query(func.count(SDOHIndicator.id)).scalar() or 0
        n_tracts = db.query(func.count(CensusTract.id)).scalar() or 0
    finally:
        db.close()
    df = df.copy()
    df["county"] = df["geoid"].str[:5].map(lambda f: counties.get(f, f))
    return Dataset(
        df=df,
        source="PostGIS · CDC PLACES (data.cdc.gov, cwsq-ngmh) cargado por el ETL",
        year=year,
        names=names,
        domains=domains,
        notes=[
            f"{n_tracts} census tracts y {n_values} valores de indicadores en la base.",
            "Cuando un indicador no existe en el año pedido se toma su año más reciente "
            "(CDC publica algunas medidas en años alternos).",
        ],
    )


def load_csv(raw: bytes, filename: str = "") -> Dataset:
    """Admite el CSV crudo de CDC PLACES (formato largo) o una tabla ancha con códigos."""
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    notes = [f"Archivo: {filename or 'CSV'} · {len(df)} filas × {df.shape[1]} columnas."]

    if {"locationid", "measure", "data_value"} <= set(cols):
        long = pd.DataFrame(
            {
                "geoid": df[cols["locationid"]].astype(str).str.zfill(11),
                "name": df[cols.get("locationname", cols["locationid"])].astype(str),
                "code": df[cols["measure"]].astype(str).str.strip().str.lower().map(_MEASURE_TO_CODE),
                "year": pd.to_numeric(df[cols["year"]], errors="coerce") if "year" in cols else 0,
                "value": pd.to_numeric(df[cols["data_value"]], errors="coerce"),
            }
        )
        dropped = int(long["code"].isna().sum() + long["value"].isna().sum())
        long = long.dropna(subset=["code", "value"]).sort_values("year", ascending=False)
        long = long.drop_duplicates(["geoid", "code"], keep="first")
        wide = long.pivot(index=["geoid", "name"], columns="code", values="value").reset_index()
        notes.append(
            f"Formato largo CDC PLACES convertido a tabla ancha; {dropped} registros sin medida "
            "reconocida o sin valor fueron descartados (misma regla que el ETL)."
        )
        year = int(long["year"].max()) if long["year"].notna().any() else None
    elif "geoid" in cols:
        wide = df.rename(columns={cols["geoid"]: "geoid"})
        wide["geoid"] = wide["geoid"].astype(str).str.zfill(11)
        if "name" not in wide.columns:
            wide["name"] = wide["geoid"]
        year = None
        notes.append("Tabla ancha detectada (una columna por código de indicador).")
    else:
        raise ValueError(
            "Formato no reconocido: se espera el CSV de CDC PLACES (LocationID, Measure, "
            "Data_Value) o una tabla ancha con columna 'geoid'."
        )

    for c in FEATURES + TARGETS:
        if c not in wide.columns:
            wide[c] = np.nan
    wide["county"] = wide["geoid"].str[:5]
    return Dataset(
        df=wide[["geoid", "name", "county"] + FEATURES + TARGETS],
        source=f"CSV subido ({filename})",
        year=year,
        names=dict(_ETL_NAMES),
        domains=dict(_ETL_DOMAINS),
        notes=notes,
    )


def quality_table(ds: Dataset) -> pd.DataFrame:
    """Completitud y tipo de cada variable del modelo."""
    rows = []
    n = len(ds.df)
    for c in FEATURES + TARGETS:
        s = ds.df[c]
        rows.append(
            {
                "Código": c,
                "Variable": ds.label(c),
                "Dominio": ds.domains.get(c, ""),
                "Rol": "Entrada (X)" if c in FEATURES else "Objetivo (y)",
                "Observados": int(s.notna().sum()),
                "Faltantes": int(s.isna().sum()),
                "Completitud %": round(100 * s.notna().mean(), 2) if n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def interpret_quality(q: pd.DataFrame, n_rows: int) -> str:
    full = q[q["Completitud %"] >= 99.9]
    gaps = q[(q["Completitud %"] < 99.9) & (q["Observados"] > 0)]
    empty = q[q["Observados"] == 0]
    text = (
        f"El dataset tiene {n_rows} census tracts. {len(full)} de {len(q)} variables están "
        f"completas (≥ 99,9 %)."
    )
    if len(gaps):
        worst = gaps.sort_values("Completitud %").iloc[0]
        text += (
            f" {len(gaps)} tienen huecos; la peor es «{worst['Variable']}» con "
            f"{worst['Completitud %']:.1f} % de completitud ({worst['Faltantes']} tracts sin dato). "
            "Los faltantes se imputan con la mediana de entrenamiento dentro del pipeline, "
            "sin usar información del conjunto de prueba."
        )
    if len(empty):
        text += (
            f" {len(empty)} variables no tienen ningún valor en esta fuente "
            f"({', '.join(empty['Variable'].head(5))}) y quedan fuera del modelado."
        )
    return text
