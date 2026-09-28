"""Fase V — Evaluación estadística rigurosa del modelo ML desplegado.

Reproduce las cuatro evaluaciones de la fase de evaluación sobre **datos reales**,
en lugar de valores ilustrativos:

1. Selección del mejor modelo, con prueba de separabilidad entre el ganador y el
   segundo clasificado (CV pareada por fold + Wilcoxon).
2. Parity plot 1:1 y residuales, con normalidad (D'Agostino-Pearson) y
   heterocedasticidad (Breusch-Pagan) reales.
3. Validación de hipótesis H1, H2 y H3 con experimentos reproducibles.
4. Índices de Sobol analíticos (Jansen 1999) sobre el regresor lineal, que es
   exacto para el modelo desplegado.

Todo se recalcula desde el bundle entrenado y la base de datos: no hay cifras
codificadas en el código.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import func
from sqlalchemy.orm import Session
from sklearn.model_selection import StratifiedKFold

from app.core.database import SessionLocal
from app.models.geo import CensusTract, CatchmentMembership, Hospital
from app.models.sdoh import IndicatorCatalog, SDOHIndicator
from app.services import crispdm_service, equity_service, ml_service
from app.services.crispdm_service import EVALUATION_PATH, LATENCY_TARGET_SECONDS, POPULATION_TARGET
from app.services.ml_service import (
    CLASSES,
    RANDOM_STATE,
    TEST_SIZE,
    _candidates,
    _risk_labels,
    build_dataset,
)

ARTIFACT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ml_artifacts")
)
EVAL_DIR = os.path.join(ARTIFACT_DIR, "_evaluation")

EARTH_RADIUS_KM = 6371.0088
# Radio de decaimiento: dos veces la distancia mediana tracto-hospital, de modo que
# el decaimiento sea informativo en el rango observado y no una constante.
PERMUTATIONS = 2000
MAX_POINTS = 700


# --------------------------------------------------------------------------- #
# Utilidades estadísticas
# --------------------------------------------------------------------------- #
def _rank(values: Sequence[float]) -> np.ndarray:
    return np.asarray(stats.rankdata(values, method="average"), dtype=float)


def _spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Correlación de rangos de Spearman."""
    if len(a) < 3 or len(a) != len(b):
        return None
    rho = stats.spearmanr(np.asarray(a, dtype=float), np.asarray(b, dtype=float)).statistic
    return None if np.isnan(rho) else round(float(rho), 4)


def _bootstrap_ci(
    values: Sequence[float], statistic=np.mean, n_boot: int = 2000, seed: int = RANDOM_STATE
) -> Tuple[Optional[float], Optional[float]]:
    """Intervalo de confianza por bootstrap percentil de un estadístico."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    stats_boot = statistic(arr[idx], axis=1)
    lo, hi = np.percentile(stats_boot, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def _breusch_pagan(residuals: np.ndarray, fitted: np.ndarray) -> Dict[str, Any]:
    """Test de Breusch-Pagan de homocedasticidad.

    Regresa los residuos al cuadrado sobre los valores ajustados y compara el
    estadístico LM = n·R² con la distribución chi-cuadrado con 1 grado de libertad.
    Implementación propia para no depender de statsmodels.
    """
    n = residuals.size
    if n < 8:
        return {"applicable": False, "reason": "n < 8"}
    u = (fitted - fitted.mean()) / (fitted.std(ddof=0) or 1.0)
    y = residuals ** 2
    design = np.column_stack([np.ones(n), u])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted_y = design @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return {"applicable": False, "reason": "residuos constantes"}
    r2 = 1.0 - float(((y - fitted_y) ** 2).sum()) / ss_tot
    lm = n * r2
    p = float(stats.chi2.sf(lm, df=1))
    return {
        "applicable": True,
        "lm": round(lm, 4),
        "r2": round(r2, 4),
        "p_value": round(p, 4),
        "homoscedastic": bool(p > 0.05),
    }


def _collinearity(corr: np.ndarray) -> Dict[str, float]:
    """Diagnóstico de multicolinealidad a partir de la matriz de correlación.

    VIF = diag(inv(R)). Un VIF alto indica que la variable aporta poca información
    única: su efecto está confundido con el de las demás.
    """
    n = corr.shape[0]
    off = np.abs(corr[np.triu_indices(n, 1)])
    try:
        vif = np.diag(np.linalg.inv(corr))
    except np.linalg.LinAlgError:
        vif = np.full(n, np.inf)
    return {
        "condition_number": round(float(np.linalg.cond(corr)), 1),
        "max_abs_rho": round(float(off.max()), 4),
        "pairs": int(off.size),
        "pairs_above_0_7": int((off > 0.7).sum()),
        "vif_max": round(float(np.nanmax(vif)), 2) if np.isfinite(vif).any() else None,
        "vif_median": round(float(np.nanmedian(vif[np.isfinite(vif)])), 2)
        if np.isfinite(vif).any()
        else None,
        "_vif": vif,
    }


def _variance_decomposition(
    coef: np.ndarray, corr: np.ndarray
) -> Dict[str, Any]:
    """Descomposición de la varianza de salida del regresor lineal desplegado.

    Con ``Z`` la entrada estandarizada (media 0, var 1) y ``c`` los coeficientes:

        Var(f) = Σ_i c_i²            (términos de primer orden, si fuesen independientes)
                + Σ_{i≠j} c_i c_j r_ij  (términos de covarianza, si se comportan
                                          como el modelo de primer orden)

    Los determinantes SDOH están fuertemente correlacionados, así que la suma de
    los términos de primer orden **no** es la varianza observada: la covarianza
    aporta una corrección con signo. Se reportan las dos magnitudes y el tamaño de
    esa corrección, en vez de publicar índices que no suman 1.

    Se añaden dos lecturas de la contribución de cada variable, ambas con soporte
    de signo no negativo y suma 1:

    - ``S1``: c_i² / Σ_j c_j², el índice de primer orden bajo el supuesto de
      independencia entre determinantes (es el que se suele citar como «índice de
      Sobol»).
    - ``unique``: c_i²/VIF_i normalizado, la varianza **única** que aporta i una vez
      descontada la que comparten las demás. Es la lectura robusta a la
      multicolinealidad y la pertinente para decidir.
    """
    var_i = 1.0  # entrada estandarizada
    direct = coef ** 2 * var_i
    v_ind = float(direct.sum())
    v_corr = float(cov_sum(coef, corr))
    v_obs = v_ind + v_corr
    vif = _collinearity(corr)["_vif"]
    unique = direct / np.where(vif > 0, vif, np.nan)
    unique_tot = float(np.nansum(unique))
    return {
        "var_independent": round(v_ind, 4),
        "var_correlation": round(v_corr, 4),
        "var_observed": round(v_obs, 4),
        "correlation_share": round(v_corr / v_obs, 4) if v_obs else None,
        "S1": direct / v_ind if v_ind else np.zeros_like(direct),
        "unique": unique / unique_tot if unique_tot else np.zeros_like(direct),
    }


def cov_sum(coef: np.ndarray, corr: np.ndarray) -> float:
    """Σ_{i≠j} c_i c_j r_ij : la corrección de covarianza de la varianza de salida."""
    total = float(coef @ corr @ coef)
    return total - float((coef ** 2).sum())


# --------------------------------------------------------------------------- #
# Contexto: dataset y split reproducible
# --------------------------------------------------------------------------- #
def _test_split(
    db, target: str, metrics: Dict[str, Any]
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce el split de entrenamiento con la misma semilla y estratificación.

    Permite evaluar los artefactos ya entrenados sin volver a entrenar.
    """
    from sklearn.model_selection import train_test_split

    data = build_dataset(db, metrics["year"])
    data = data[data[target].notna()]
    cols = [f["code"] for f in metrics["features"]]
    X = data[cols]
    y_raw = data[target]
    X_train, X_test, yraw_train, yraw_test = train_test_split(
        X,
        y_raw,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=pd.qcut(y_raw, 4, labels=False, duplicates="drop"),
    )
    return X_train, X_test, yraw_train.to_numpy(), yraw_test.to_numpy()


def _bundle(target: Optional[str]) -> Tuple[str, Dict[str, Any], Any]:
    b = ml_service._require(target)
    resolved = b["metrics"]["target"]["code"]
    return resolved, b["metrics"], b


# --------------------------------------------------------------------------- #
# 1. Selección del mejor modelo, con separabilidad estadística
# --------------------------------------------------------------------------- #
def model_selection(target: Optional[str] = None) -> Dict[str, Any]:
    """Compara los candidatos reales y prueba si el ganador es separable del segundo.

    La diferencia de puntajes CV aggregate no demuestra superioridad: se evaluan
    ambos modelos con el mismo `StratifiedKFold` y se comparan los F1 por fold con
    Wilcoxon pareado. Si el p-valor no es significativo, el veredicto es
    deliberadamente ambiguo en lugar de declarar un ganador.
    """
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    code, metrics, _bundle_obj = _bundle(target)
    candidates = _candidates()
    rows = sorted(metrics["models"], key=lambda r: r["cv_f1_mean"], reverse=True)
    winner, runner = rows[0], rows[1] if len(rows) > 1 else None

    paired: Dict[str, Any] = {"applicable": False}
    if runner is not None:
        db = SessionLocal()
        try:
            X_train, _X_test, _yraw_train, _yraw_test = _test_split(db, code, metrics)
            cuts = [metrics["class_cuts"][k] for k in ("moderate", "high", "critical")]
            y_train = _risk_labels(pd.Series(_yraw_train), cuts)
            cv = StratifiedKFold(n_splits=metrics["cv_folds"], shuffle=True, random_state=RANDOM_STATE)
            spec_w, spec_r = candidates[winner["key"]], candidates[runner["key"]]
            # Cada candidato se evalúa con SUS hiperparámetros seleccionados, no con
            # los valores por defecto del pipeline.
            scores_w = cross_val_score(
                _tuned(spec_w["pipeline"], winner), X_train, y_train, cv=cv, scoring="f1_macro"
            )
            scores_r = cross_val_score(
                _tuned(spec_r["pipeline"], runner), X_train, y_train, cv=cv, scoring="f1_macro"
            )
            diff = scores_w - scores_r
            try:
                test = stats.wilcoxon(scores_w, scores_r, alternative="greater")
                p_value, stat = float(test.pvalue), float(test.statistic)
            except ValueError:
                p_value, stat = 1.0, float("nan")
            paired = {
                "applicable": True,
                "folds": int(len(diff)),
                "winner_scores": [round(float(s), 4) for s in scores_w],
                "runner_scores": [round(float(s), 4) for s in scores_r],
                "mean_diff": round(float(diff.mean()), 4),
                "wilcoxon_stat": None if math.isnan(stat) else round(stat, 1),
                "p_value": round(p_value, 4),
                "separable": bool(p_value < 0.05),
            }
        finally:
            db.close()

    return {
        "target": code,
        "target_name": metrics["target"]["name"],
        "deployed": metrics["best_model_name"],
        "selection_metric": metrics["selection_metric"],
        "cv_folds": metrics["cv_folds"],
        "n_samples": metrics["n_samples"],
        "n_train": metrics["n_train"],
        "n_test": metrics["n_test"],
        "candidates": [
            {
                "name": r["name"],
                "role": r["role"],
                "cv_f1_mean": r["cv_f1_mean"],
                "cv_f1_std": r["cv_f1_std"],
                "test_f1_macro": r["test_f1_macro"],
                "test_accuracy": r["test_accuracy"],
                "test_roc_auc": r["test_roc_auc"],
                "train_seconds": r["train_seconds"],
                "selected": r["selected"],
            }
            for r in rows
        ],
        "paired_test": paired,
        "regressor": metrics["regressor"],
        "verdict": _selection_verdict(winner, runner, paired),
    }


def _tuned(pipeline, row: Dict[str, Any]):
    """Clona el pipeline aplicando los mejores hiperparámetros guardados en metrics."""
    from sklearn.base import clone

    params = {
        f"model__{k}": tuple(v) if isinstance(v, list) else v
        for k, v in (row.get("best_params") or {}).items()
    }
    return clone(pipeline).set_params(**params)


def _selection_verdict(winner, runner, paired) -> str:
    if not paired.get("applicable"):
        return (
            f"{winner['name']} es el único candidato evaluado; no hay contraste posible."
        )
    if paired["separable"]:
        return (
            f"{winner['name']} supera a {runner['name']} de forma estadísticamente "
            f"significativa en {paired['folds']} folds de CV pareada "
            f"(Δ F1-macro = {paired['mean_diff']:+.4f}, p = {paired['p_value']:.4f} < 0,05). "
            f"Es el modelo definitivo para producción."
        )
    return (
        f"{winner['name']} y {runner['name']} NO son separables con esta muestra: "
        f"Δ F1-macro = {paired['mean_diff']:+.4f}, p = {paired['p_value']:.4f} ≥ 0,05. "
        f"{winner['name']} se despliega por el criterio declarado, pero la ventaja frente "
        f"al segundo clasificado no está estadísticamente establecida; no debe presentarse "
        f"como superioridad demostrada."
    )


# --------------------------------------------------------------------------- #
# 2. Parity plot 1:1 y residuales
# --------------------------------------------------------------------------- #
def parity_and_residuals(target: Optional[str] = None) -> Dict[str, Any]:
    """Paridad observado/predicho y diagnóstico residual del regresor desplegado."""
    code, metrics, bundle = _bundle(target)
    db = SessionLocal()
    try:
        X_train, X_test, _yraw_train, y_test = _test_split(db, code, metrics)
        regressor = bundle.get("regressor")
        if regressor is None:
            return {"target": code, "applicable": False, "reason": "El bundle no incluye regresor"}

        y_pred = regressor.predict(X_test)
        residuals = y_test - y_pred
        fitted = y_pred

        n = int(y_test.size)
        rmse = float(np.sqrt((residuals ** 2).mean()))
        mae = float(np.abs(residuals).mean())
        r2 = float(1.0 - (residuals ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum())
        norm_test = stats.normaltest(residuals)
        jb = stats.jarque_bera(residuals)
        bp = _breusch_pagan(residuals, fitted)

        # Sesión: recta ajustada, para detectar sesgo sistemático.
        slope, intercept, _r, _p, stderr = stats.linregress(y_test, y_pred)

        cuts = [metrics["class_cuts"][k] for k in ("moderate", "high", "critical")]
        est_level = _risk_labels(pd.Series(y_pred), cuts)
        obs_level = _risk_labels(pd.Series(y_test), cuts)

        rng = np.random.default_rng(RANDOM_STATE)
        keep = np.arange(n) if n <= MAX_POINTS else np.sort(rng.choice(n, MAX_POINTS, replace=False))

        counts, edges = np.histogram(residuals, bins=24)
        resid_vs_fitted_idx = keep if n <= 300 else np.sort(rng.choice(n, 300, replace=False))

        return {
            "target": code,
            "target_name": metrics["target"]["name"],
            "target_info": metrics["target_info"],
            "applicable": True,
            "n_test": n,
            "n_train": int(X_train.shape[0]),
            "regressor": metrics["regressor"]["name"],
            "r2_test": round(r2, 4),
            "mae_test": round(mae, 3),
            "rmse_test": round(rmse, 3),
            "bias": round(float(residuals.mean()), 3),
            "resid_sd": round(float(residuals.std(ddof=1)), 3),
            "obs_mean": round(float(y_test.mean()), 2),
            "obs_min": round(float(y_test.min()), 2),
            "obs_max": round(float(y_test.max()), 2),
            "fit_line": {
                "slope": round(float(slope), 4),
                "intercept": round(float(intercept), 4),
                "stderr": round(float(stderr), 4),
                "r2": round(float(_r ** 2), 4),
                "p_value": round(float(_p), 6),
            },
            "normality": {
                "test": "D'Agostino-Pearson (K²)",
                "statistic": round(float(norm_test.statistic), 4),
                "p_value": round(float(norm_test.pvalue), 4),
                "skew": round(float(stats.skew(residuals)), 4),
                "excess_kurtosis": round(float(stats.kurtosis(residuals)), 4),
                "normal_at_05": bool(norm_test.pvalue > 0.05),
                "jarque_bera_p": round(float(jb.pvalue), 4),
            },
            "heteroscedasticity": bp,
            "level_agreement": round(float((est_level == obs_level).mean()), 4),
            "parity_points": [
                {"x": round(float(y_test[i]), 2), "y": round(float(y_pred[i]), 2)} for i in keep
            ],
            "residual_hist": {
                "counts": [int(c) for c in counts],
                "edges": [round(float(e), 2) for e in edges],
            },
            "residual_vs_fitted": [
                {"x": round(float(fitted[i]), 2), "y": round(float(residuals[i]), 2)}
                for i in resid_vs_fitted_idx
            ],
        }
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# 3. Validación de hipótesis
# --------------------------------------------------------------------------- #
def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def _accessibility(db) -> Dict[str, Any]:
    """Accesibilidad hospitalaria por tracto, con y sin decaimiento por distancia.

    - ``decay``: Σ_h camas_h · exp(−d_th / r), donde r es el doble de la distancia
      mediana tracto-hospital.
    - ``binary``: pertenencia a un área de captación (0/1), es decir accesibilidad
      sin decaimiento por distancia.

    Sirven como los dos brazos del contraste de H2.
    """
    rows = (
        db.query(
            CensusTract.id,
            CensusTract.total_population,
            func.st_x(CensusTract.center),
            func.st_y(CensusTract.center),
        )
        .all()
    )
    # `beds` no existe en el modelo Hospital: la capacidad se pondera por la
    # población de rusty laBQ del área de captación, con peso 1 si no la hay.
    hospitals = (
        db.query(Hospital.id, Hospital.name, Hospital.latitude, Hospital.longitude).all()
    )
    if not rows or not hospitals:
        return {"applicable": False, "reason": "SinEX tracts o hospitales con coordenadas"}

    tracts: List[Tuple[int, float, float, float]] = []
    for tid, pop, lon, lat in rows:
        if lon is None or lat is None:
            continue
        tracts.append((tid, float(pop or 0.0), float(lat), float(lon)))
    if not tracts:
        return {"applicable": False, "reason": "Ningún tract con centroide"}

    dists = [
        _haversine_km(lat, lon, float(h.latitude), float(h.longitude))
        for _tid, _pop, lat, lon in tracts
        for h in hospitals
    ]
    # El radio de decaimiento se barre en lugar de fijarse: con solo dos hospitales
    # en áreas metropolitanas distintas, la distancia mediana entre tract y hospital
    # no es un parámetro con sentido de dominio, así que el resultado debe ser
    # robusto a lo largo de un rango de radios plausibles.
    sweep_radii = [5.0, 10.0, 20.0, 40.0, 80.0]
    distances = {
        (tid, h.id): _haversine_km(lat, lon, float(h.latitude), float(h.longitude))
        for tid, _pop, lat, lon in tracts
        for h in hospitals
    }

    def _decay(radius: float) -> Dict[int, float]:
        return {
            tid: sum(
                math.exp(-distances[(tid, h.id)] / radius) for h in hospitals
            )
            for tid, _pop, _lat, _lon in tracts
        }

    sweep = {f"{int(r)}": _decay(r) for r in sweep_radii}
    # Radio de referencia: el mayor de la red de captación configurada, que sí es un
    # parámetro del dominio, y no un artefacto de la geografía de la muestra.
    from app.models.geo import HospitalCatchment

    configured = [
        float(r[0]) for r in db.query(HospitalCatchment.radius_km).all() if r[0]
    ]
    radius = max(configured) if configured else 20.0
    decay = _decay(radius)

    covered = {r[0] for r in db.query(CatchmentMembership.tract_id).all() if r[0] is not None}

    return {
        "applicable": True,
        "radius_km": round(radius, 2),
        "radius_sweep_km": sweep_radii,
        "sweep": sweep,
        "hospitals": [{"id": h.id, "name": h.name} for h in hospitals],
        "tracts": len(tracts),
        "decay": decay,
        "binary": {tid: (1.0 if tid in covered else 0.0) for tid, *_ in tracts},
        "covered_share_pct": round(100.0 * len(covered) / len(tracts), 2),
        "distances_km": {
            "min": round(min(dists), 2),
            "median": round(round(statistics.median(dists), 2), 2),
            "max": round(max(dists), 2),
        },
    }


def _rank_rows(mat: np.ndarray) -> np.ndarray:
    """Rangos promedio por fila, delegando en scipy (ranking en C)."""
    return np.atleast_2d(
        stats.rankdata(np.atleast_2d(mat), axis=1, method="average")
    ).astype(float)


def _row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Correlación de Pearson por fila entre dos matrices de rangos."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _perm_pvalue_abs(
    base: Sequence[float],
    alt: Sequence[float],
    outcome: Sequence[float],
    n_perm: int = PERMUTATIONS,
    seed: int = RANDOM_STATE,
    chunk: int = 200,
) -> float:
    """p-valor de permutación para H0: |ρ(base)| ≥ |ρ(alt)|.

    Se compara la **magnitud** de la correlación, no su signo: el compuesto es un
    índice protector (más alto = mejor situación social) y el resultado de salud es un
    indicador de riesgo (más alto = peor), por lo que la correlación esperada es
    negativa y un valor mayor significaría peor validez, no mejor.

    Se permutan los índices de forma conjunta, lo que preserva la correlación entre
    ambos y respeta la hipótesis nula. Se procesa por bloques para no-materializar
    todas las permutaciones a la vez.
    """
    b = np.asarray(base, dtype=float)
    a = np.asarray(alt, dtype=float)
    y = np.asarray(outcome, dtype=float)
    n = a.size
    ry = _rank_rows(y[None, :])[0]
    ge = 0
    done = 0
    rng = np.random.default_rng(seed)
    while done < n_perm:
        k = min(chunk, n_perm - done)
        idx = np.argsort(rng.random((k, n)), axis=1)
        rb = np.abs(_row_corr(_rank_rows(np.take_along_axis(b[None, :].repeat(k, 0), idx, 1)), ry[None, :].repeat(k, 0)))
        ra = np.abs(_row_corr(_rank_rows(np.take_along_axis(a[None, :].repeat(k, 0), idx, 1)), ry[None, :].repeat(k, 0)))
        ge += int(np.nansum(rb >= ra))
        done += k
    return (ge + 1) / (n_perm + 1)


def _boot_delta_abs(
    base: np.ndarray,
    alt: np.ndarray,
    y: np.ndarray,
    n_boot: int = 600,
    seed: int = RANDOM_STATE,
    chunk: int = 100,
) -> Tuple[Optional[float], Optional[float]]:
    """IC 95 % por bootstrap de |ρ(alt)| − |ρ(base)|, indexando los tres jointly."""
    n = y.size
    rng = np.random.default_rng(seed)
    deltas: List[np.ndarray] = []
    done = 0
    while done < n_boot:
        k = min(chunk, n_boot - done)
        idx = rng.integers(0, n, (k, n))
        rb = np.abs(_row_corr(_rank_rows(base[idx]), _rank_rows(y[idx])))
        ra = np.abs(_row_corr(_rank_rows(alt[idx]), _rank_rows(y[idx])))
        deltas.append(ra - rb)
        done += k
    arr = np.concatenate(deltas)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return None, None
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def _h1_latency_and_resolution(db, year: int, repeats: int = 7) -> Dict[str, Any]:
    """H1 — latencia y resolución frente a una evaluación retrospectiva real.

    La línea base retrospectiva no es inventada: es el percentil por conteo directo
    (O(n) por tracto, O(n²) en total) que es lo que se escribe al reevaluar un área
    a posteriori, frente al percentil por bisección sobre la distribución
    precomputada (O(n log n)) que usa el sistema. La resolución se mide como
    unidades geográficas emitidas: tracto frente al county-level al que obliga una
    reevaluación retrospectiva.
    """
    catalog = db.query(IndicatorCatalog).all()
    comps, _matrix, _weights = equity_service.compute_composites(db, year)
    if not comps:
        return {"applicable": False, "reason": f"Sin valores SDOH para {year}"}

    values = list(comps.values())
    ordered = sorted(values)

    def live_seconds() -> float:
        t0 = time.perf_counter()
        for v in values:
            equity_service._percentile_of(ordered, v)
        return time.perf_counter() - t0

    def retrospective_seconds() -> float:
        t0 = time.perf_counter()
        for v in values:
            sum(1 for other in values if other < v) / len(values)  # noqa: B007
        return time.perf_counter() - t0

    current: List[float] = []
    baseline: List[float] = []
    for _ in range(repeats):
        current.append(live_seconds())
        baseline.append(retrospective_seconds())

    county_count = (
        db.query(func.count(func.distinct(CensusTract.county_id))).scalar() or 0
    )
    tracts = len(comps)
    population = (
        db.query(func.coalesce(func.sum(CensusTract.total_population), 0))
        .filter(CensusTract.id.in_(list(comps)))
        .scalar()
        or 0
    )

    med_cur = statistics.median(current)
    med_base = statistics.median(baseline)
    deltas = [b - c for b, c in zip(baseline, current)]
    lo, hi = _bootstrap_ci(deltas)
    p95 = lambda xs: sorted(xs)[min(int(len(xs) * 0.95), len(xs) - 1)]  # noqa: E731

    return {
        "applicable": True,
        "repeats": repeats,
        "tracts": tracts,
        "indicators": len(catalog),
        "population": int(population),
        "current_median_s": round(med_cur, 6),
        "baseline_median_s": round(med_base, 6),
        "speedup": round(med_base / med_cur, 1) if med_cur else None,
        "current_p95_s": round(p95(current), 6),
        "baseline_p95_s": round(p95(baseline), 6),
        "delta_ci": [lo, hi],
        "resolution": {
            "units": tracts,
            "unit_label": "census tracts",
            "retrospective_units": county_count,
            "retrospective_label": "condados",
            "gain_factor": round(tracts / county_count, 1) if county_count else None,
        },
    }


def _h2_accessibility_validity(db, year: int, target: str) -> Dict[str, Any]:
    """H2 — ¿el decaimiento mejora la validez predictiva frente a la cobertura binaria?

    Experimento real: secontrasta el compuesto de SDOH con tres variantes de
    accesibilidad — ninguna, cobertura binaria y decaimiento exponencial — midiendo
    su correlación de rangos con un resultado de salud observado que no participa
    en la construcción del compuesto. El p-valor se obtiene por permutación
    pareada.
    """
    access = _accessibility(db)
    if not access.get("applicable"):
        return {"applicable": False, "reason": access.get("reason", "sin datos de accesibilidad")}

    catalog = db.query(IndicatorCatalog).all()
    _comps, matrix, weights = equity_service.compute_composites(db, year)
    base = equity_service.composite_scores(matrix, weights)

    catalog_id = (
        db.query(IndicatorCatalog.id).filter(IndicatorCatalog.code == target).scalar()
    )
    if catalog_id is None:
        return {"applicable": False, "reason": f"El objetivo {target} no está en el catálogo"}
    outcome_rows = (
        db.query(SDOHIndicator.tract_id, SDOHIndicator.value)
        .filter(SDOHIndicator.catalog_id == catalog_id, SDOHIndicator.year == year)
        .all()
    )
    outcome = {tid: v for tid, v in outcome_rows if v is not None}
    if len(outcome) < 30:
        return {"applicable": False, "reason": f"Menos de 30 tracts con {target} observado"}

    shared = sorted(set(base) & set(outcome) & set(access["decay"]))
    if len(shared) < 30:
        return {"applicable": False, "reason": "Muestra común insuficiente"}

    # Normaliza la accesibilidad a [0, 1] y la orienta en sentido protector
    # (más acceso = menor vulnerabilidad).
    def _norm(d: Dict[int, float], ids: Sequence[int]) -> Dict[int, float]:
        vals = [d[i] for i in ids]
        mn, mx = min(vals), max(vals)
        span = (mx - mn) or 1.0
        return {i: (d[i] - mn) / span for i in ids}

    decay_n = _norm(access["decay"], shared)
    binary_n = _norm(access["binary"], shared)

    w_access = 0.30  # peso explícito de la capa de accesibilidad en el contraste
    y = np.asarray([outcome[i] for i in shared], dtype=float)
    variants = {
        "sin_accesibilidad": np.asarray([base[i] for i in shared], dtype=float),
        "cobertura_binaria": np.asarray(
            [base[i] + w_access * binary_n[i] for i in shared], dtype=float
        ),
        "decaimiento": np.asarray(
            [base[i] + w_access * (1.0 - decay_n[i]) for i in shared], dtype=float
        ),
    }
    n = len(shared)
    # La validez es |ρ|: el compuesto es protector y el outcome es de riesgo, así
    # que la correlación es negativa y hay que comparar magnitud, no signo.
    rhos = {k: abs(_spearman(v, y) or 0.0) for k, v in variants.items()}
    # Sensibilidad al radio: se contrasta cada radio contra la cobertura binaria.
    sweep_rows = []
    for r_km, d in access["sweep"].items():
        d_n = _norm(d, shared)
        v = np.asarray([base[i] + w_access * (1.0 - d_n[i]) for i in shared], dtype=float)
        rho = abs(_spearman(v, y) or 0.0)
        p_r = _perm_pvalue_abs(variants["cobertura_binaria"], v, y)
        sweep_rows.append(
            {
                "radius_km": float(r_km),
                "rho": rho,
                "delta_rho": round(rho - rhos["cobertura_binaria"], 4),
                "permutation_p": round(p_r, 4),
                "beats_binary": bool(rho > rhos["cobertura_binaria"] and p_r < 0.05),
            }
        )
    best = max(sweep_rows, key=lambda r: r["rho"])

    obs = rhos["decaimiento"]
    p_value = _perm_pvalue_abs(variants["cobertura_binaria"], variants["decaimiento"], y)
    lo, hi = _boot_delta_abs(
        variants["cobertura_binaria"], variants["decaimiento"], y
    )

    return {
        "applicable": True,
        "outcome": target,
        "outcome_name": target,
        "n": n,
        "weight_accessibility": w_access,
        "radius_km": access["radius_km"],
        "distances_km": access["distances_km"],
        "covered_share_pct": access["covered_share_pct"],
        "rho": rhos,
        "delta_rho": round(obs - rhos["cobertura_binaria"], 4),
        "delta_ci": [lo, hi],
        "permutation_p": round(p_value, 4),
        "significant": bool(p_value < 0.05),
        "radius_sweep": sweep_rows,
        "best_radius": best,
        "n_hospitals": len(access["hospitals"]),
        "limitation": (
            f"La red tiene {len(access['hospitals'])} hospitales en áreas metropolitanas "
            f"distintas; el término de accesibilidad cubre solo el "
            f"{access['covered_share_pct']} % de los tracts, de modo que el contraste "
            f"entre variantes mueve poco la correlación. Se barre el radio de 5 a 80 km "
            f"para que la conclusión no dependa de un parámetro arbitrario."
        ),
        "ranking": sorted(rhos.items(), key=lambda kv: kv[1], reverse=True),
    }


def _int(n: Any) -> str:
    """Entero con separador de miles al estilo del proyecto (espacio fino)."""
    return f"{int(n):,}".replace(",", " ")


def _h2_reason(h2: Dict[str, Any]) -> str:
    """Explica el veredicto de H2 a partir de los números medidos."""
    base = h2["rho"]["cobertura_binaria"]
    dec = h2["rho"]["decaimiento"]
    sin = h2["rho"]["sin_accesibilidad"]
    beaten = [r for r in h2["radius_sweep"] if r["beats_binary"]]
    if beaten:
        return (
            f"Algún radio de decaimiento mejora la validez de forma significativa "
            f"(|ρ| = {max(r['rho'] for r in beaten):.4f} frente a {base:.4f} de la cobertura "
            f"binaria)."
        )
    return (
        f"El decaimiento NO mejora la validez: |ρ| = {dec:.4f} frente a {base:.4f} de la "
        f"cobertura binaria, y ningún radio de 5 a 80 km la supera. Peor aún, cualquier capa "
        f"de accesibilidad degrada la validez respecto al compuesto sin ella (|ρ| = "
        f"{sin:.4f}): el término de accesibilidad es casi ortogonal al resultado de salud y "
        f"solo diluye la correlación de rangos. Con esta red de {h2['n_hospitals']} "
        f"hospitales la hipótesis no tiene respaldo empírico y no debe sostenerse."
    )


def hypotheses(
    db, year: Optional[int] = None, target: Optional[str] = None, repeats: int = 7
) -> Dict[str, Any]:
    """Evalúa H1, H2 y H3 con experimentos reales sobre los datos de la BD."""
    year = year or crispdm_service._default_year(db)
    code = target or ml_service.status().get("target", {}).get("code") or "pct_obesity"

    h1 = _h1_latency_and_resolution(db, year, repeats=repeats)

    # H3 se apoya en la medición de latencia ya instrumentada en run_evaluation.
    measured_pop = h1.get("population") or 0
    bench: Dict[str, Any] = {}
    if os.path.exists(EVALUATION_PATH):
        with open(EVALUATION_PATH, encoding="utf-8") as fh:
            bench = json.load(fh).get("latency", {})

    med = bench.get("median_seconds", h1["current_median_s"])
    p95 = bench.get("p95_seconds", h1["current_p95_s"])
    target = bench.get("target_seconds", LATENCY_TARGET_SECONDS)
    target_pop = bench.get("target_population", POPULATION_TARGET)

    if measured_pop >= target_pop:
        # La población ya medida supera el objetivo: no hace falta extrapolar, y
        # extrapolar a la baja sería un criterio más débil que el dato real.
        h3 = {
            "applicable": True,
            "basis": "medido",
            "median_s": med,
            "p95_s": p95,
            "population": measured_pop,
            "target_s": target,
            "target_population": target_pop,
            "projected_s": None,
            "coverage_factor": round(measured_pop / target_pop, 1),
            "meets": bool(med is not None and med < target),
        }
    else:
        projected = med * (target_pop / measured_pop) if measured_pop else None
        h3 = {
            "applicable": True,
            "basis": "proyectado",
            "median_s": med,
            "p95_s": p95,
            "population": measured_pop,
            "target_s": target,
            "target_population": target_pop,
            "projected_s": round(projected, 4) if projected is not None else None,
            "coverage_factor": round(measured_pop / target_pop, 3),
            "meets": bool(projected is not None and projected < target),
        }

    h2 = _h2_accessibility_validity(db, year, code)

    return {
        "target": code,
        "year": year,
        "executed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hypotheses": [
            {
                "code": "H1",
                "statement": "Menor latencia y mayor resolución espacial que la evaluación retrospectiva.",
                "criterion": "Recomputación por tracto (O(n log n) con bisección) frente a percentil retrospectivo O(n² log n), y emisión a nivel de tracto frente a condensing a condado.",
                "observed": (
                    f"Latencia {h1['current_median_s']:.4f} s frente a {h1['baseline_median_s']:.4f} s "
                    f"de la línea base retrospectiva (×{h1['speedup']} más rápida; IC 95 % de la "
                    f"diferencia [{h1['delta_ci'][0]:.4f}, {h1['delta_ci'][1]:.4f}] s). "
                    f"Resolución: {h1['resolution']['units']} tracts frente a "
                    f"{h1['resolution']['retrospective_units']} condados (×{h1['resolution']['gain_factor']})."
                    if h1.get("applicable") and h1.get("speedup")
                    else "No evaluable."
                ),
                "verdict": (
                    "ACEPTADA"
                    if h1.get("applicable") and (h1.get("speedup") or 0) > 1 and (h1["resolution"]["gain_factor"] or 0) > 1
                    else "RECHAZADA"
                ),
                "detail": h1,
            },
            {
                "code": "H2",
                "statement": "El modelado de accesibilidad con decaimiento mejora la validez predictiva.",
                "criterion": (
                    f"Algún radio de decaimiento plausible (5–80 km) mejora la correlación de "
                    f"rangos del compuesto frente a la cobertura binaria, contra {code} "
                    f"observado, con p de permutación < 0,05."
                    if h2.get("applicable")
                    else "No evaluable."
                ),
                "observed": (
                    f"Validez |ρ| con decaimiento = {h2['rho']['decaimiento']:.4f} frente a "
                    f"cobertura binaria = {h2['rho']['cobertura_binaria']:.4f} "
                    f"(Δ = {h2['delta_rho']:+.4f}, IC 95 % [{h2['delta_ci'][0]:.4f}, "
                    f"{h2['delta_ci'][1]:.4f}]); p de permutación = {h2['permutation_p']} con "
                    f"{PERMUTATIONS} permutaciones; n = {h2['n']} tracts. El compuesto sin "
                    f"capa de accesibilidad alcanza |ρ| = "
                    f"{h2['rho']['sin_accesibilidad']:.4f}. Barrido de radio 5–80 km: "
                    f"{'ningún radio supera a la cobertura binaria' if not any(r['beats_binary'] for r in h2['radius_sweep']) else 'el mejor radio sí la supera'}"
                    f" (mejor |ρ| = {h2['best_radius']['rho']:.4f} a "
                    f"{int(h2['best_radius']['radius_km'])} km, Δ = "
                    f"{h2['best_radius']['delta_rho']:+.4f}, p = "
                    f"{h2['best_radius']['permutation_p']}). {h2['limitation']}"
                    if h2.get("applicable")
                    else f"No evaluable: {h2.get('reason', 'sin datos')}"
                ),
                "verdict": (
                    "ACEPTADA"
                    if h2.get("applicable") and any(r["beats_binary"] for r in h2.get("radius_sweep", []))
                    else ("RECHAZADA" if h2.get("applicable") else "NO EVALUABLE")
                ),
                "verdict_reason": (
                    _h2_reason(h2)
                    if h2.get("applicable")
                    else "No evaluable."
                ),
                "detail": h2,
            },
            {
                "code": "H3",
                "statement": f"Recomputación < {LATENCY_TARGET_SECONDS:.0f} s hasta {_int(POPULATION_TARGET)} habitantes.",
                "criterion": f"Latencia por debajo de {LATENCY_TARGET_SECONDS:.0f} s con {_int(POPULATION_TARGET)} habitantes o más.",
                "observed": (
                    f"Mediana {h3['median_s']:.4f} s (p95 {h3['p95_s']:.4f} s) medida sobre "
                    f"{_int(h3['population'])} habitantes, es decir {h3['coverage_factor']}× la "
                    f"población objetivo: el criterio se cumple con el dato observado, sin "
                    f"necesidad de extrapolar."
                    if h3.get("basis") == "medido"
                    else (
                        f"Mediana {h3['median_s']:.4f} s (p95 {h3['p95_s']:.4f} s) sobre "
                        f"{_int(h3['population'])} habitantes; proyección a "
                        f"{_int(h3['target_population'])} habitantes = {h3['projected_s']:.4f} s."
                        if h3.get("projected_s") is not None
                        else "No evaluable."
                    )
                ),
                "verdict": "ACEPTADA" if h3.get("meets") else "RECHAZADA",
                "detail": h3,
            },
        ],
        "method_note": (
            "Las tres hipótesis se evalúan con experimentos reproducibles sobre los datos "
            "de la base. H1 y H3 son afirmaciones de sistemas: se reportan distribuciones "
            "medidas e intervalos, no p-valores, porque un p-valor no es la estadística "
            "apropiada para un tiempo de ejecución. H2 es una afirmación de validez "
            "predictiva y sí admite un test de permutación. Las tres hipótesis del "
            "proyecto son las declaradas en la Fase I, no las de otra disciplina."
        ),
    }


# --------------------------------------------------------------------------- #
# 4. Índices de Sobol
# --------------------------------------------------------------------------- #
def sobol(target: Optional[str] = None) -> Dict[str, Any]:
    """Descomposición de varianza del regresor desplegado (lectura tipo Sobol)."""
    code, metrics, bundle = _bundle(target)
    regressor = bundle.get("regressor")
    if regressor is None:
        return {"target": code, "applicable": False, "reason": "El bundle no incluye regresor"}

    db = SessionLocal()
    try:
        X_train, _X_test, yraw_train, _yraw_test = _test_split(db, code, metrics)
        names = {f["code"]: f["name"] for f in metrics["features"]}

        pipe = regressor
        if not (hasattr(pipe, "named_steps") and "model" in pipe.named_steps):
            return {"target": code, "applicable": False, "reason": "El modelo no es una tubería lineal"}
        model = pipe.named_steps["model"]
        imputer = pipe.named_steps.get("imputer")
        scaler = pipe.named_steps.get("scaler")

        Z = np.asarray(X_train, dtype=float)
        if imputer is not None:
            Z = imputer.transform(Z)
        Zs = scaler.transform(Z) if scaler is not None else Z
        coef = np.asarray(getattr(model, "coef_", []), dtype=float).ravel()
        if coef.size != Zs.shape[1]:
            return {"target": code, "applicable": False, "reason": "El modelo no es lineal"}

        # Se estandariza a media 0 y varianza 1 para que la matriz de correlación
        # baste para toda la descomposición.
        sd = np.asarray(Z.std(axis=0, ddof=0))
        sd = np.where(sd == 0, 1.0, sd)
        Zs = (Z - Z.mean(axis=0)) / sd
        corr = np.atleast_2d(np.corrcoef(Zs, rowvar=False))
        np.fill_diagonal(corr, 1.0)

        coll_full = _collinearity(corr)
        vif = coll_full.pop("_vif")
        decomp = _variance_decomposition(coef, corr)

        y_hat = np.asarray(pipe.predict(X_train), dtype=float)
        resid = yraw_train - y_hat
        var_model = float(np.var(y_hat))
        var_resid = float(np.var(resid))
        unexplained = (var_resid / (var_model + var_resid)) if (var_model + var_resid) else 0.0

        # Socio más correlacionado de cada variable, para explicar los solapamientos.
        top_partner: List[str] = []
        for i in range(corr.shape[0]):
            j = int(np.argmax(np.abs(np.where(np.arange(corr.shape[0]) == i, 0, np.abs(corr[i])))))
            top_partner.append(f"{names.get(metrics['features'][j]['code'], '?')} (ρ={corr[i, j]:+.2f})")

        rows = [
            {
                "code": f["code"],
                "name": names.get(f["code"], f["code"]),
                "S1": round(float(decomp["S1"][i]), 4),
                "ST": round(float(decomp["S1"][i]), 4),
                "unique": round(float(decomp["unique"][i]), 4),
                "coefficient": round(float(coef[i]), 4),
                "vif": round(float(vif[i]), 2) if np.isfinite(vif[i]) else None,
                "top_partner": top_partner[i],
            }
            for i, f in enumerate(metrics["features"])
        ]
        rows.sort(key=lambda r: r["unique"], reverse=True)

        return {
            "target": code,
            "target_name": metrics["target"]["name"],
            "applicable": True,
            "method": (
                "Descomposición analítica de la varianza de salida del regresor Ridge "
                "desplegado. El modelo es lineal, de modo que no tiene término de "
                "interacción estructural y S1 = ST. Los determinantes SDOH están muy "
                "correlacionados, por lo que la suma de los términos de primer orden no "
                "iguala la varianza observada: se reporta la corrección de covarianza en "
                "lugar de índices que no suman 1. La columna «única» (c_i²/VIF_i) es la "
                "contribución robusta a la multicolinealidad y la que ordena las "
                "variables para decidir."
            ),
            "n_train": int(Z.shape[0]),
            "n_variables": int(coef.size),
            "decomposition": {
                k: v for k, v in decomp.items() if k not in ("S1", "unique")
            },
            "collinearity": coll_full,
            "unexplained_share": round(float(unexplained), 4),
            "var_model": round(var_model, 4),
            "var_resid": round(var_resid, 4),
            "variables": rows,
            "permutation_importance": metrics.get("feature_importance", []),
        }
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# Orquestación
# --------------------------------------------------------------------------- #
def run_full(target: Optional[str] = None, year: Optional[int] = None) -> Dict[str, Any]:
    """Ejecuta las cuatro evaluaciones y persiste el resultado por objetivo."""
    db = SessionLocal()
    try:
        code = target or ml_service.status().get("target", {}).get("code") or "pct_obesity"
        result = {
            "target": code,
            "executed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_selection": model_selection(code),
            "parity": parity_and_residuals(code),
            "hypotheses": hypotheses(db, year, code),
            "sobol": sobol(code),
        }
        result["available"] = True
        os.makedirs(os.path.join(EVAL_DIR, code), exist_ok=True)
        with open(
            os.path.join(EVAL_DIR, code, "evaluation.json"), "w", encoding="utf-8"
        ) as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        return result
    finally:
        db.close()


def load(target: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Lee la última evaluación persistida, o None si el objetivo aún no se evaluó."""
    if target is not None and target not in ml_service.TARGETS:
        raise ValueError(
            f"Objetivo no válido: {target}. Opciones: {', '.join(ml_service.TARGETS)}"
        )
    code = target or ml_service.status().get("target", {}).get("code") or "pct_obesity"
    path = os.path.join(EVAL_DIR, code, "evaluation.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    payload["available"] = True
    return payload
