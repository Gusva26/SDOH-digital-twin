"""Modelo predictivo supervisado sobre los datos reales de CDC PLACES.

Entrena y compara cuatro clasificadores que predicen el nivel de riesgo de un
resultado de salud por census tract (low / moderate / high / critical) a partir
de sus determinantes sociales. El mejor —por F1-macro en validación cruzada— se
serializa con joblib y la API lo sirve para predicciones.

El objetivo NO es el índice de equidad propio del sistema (eso sería circular:
el modelo aprendería la fórmula), sino un resultado de salud medido por CDC; las
variables de salud se excluyen de las entradas para evitar fuga de información.
"""

import bisect
import json
import os
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import func
from threadpoolctl import threadpool_limits
from sqlalchemy.orm import Session

from app.models.geo import CensusTract, County
from app.models.sdoh import IndicatorCatalog, SDOHIndicator

ARTIFACT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ml_artifacts"))
#: Objetivo cuyo modelo consumen el Dashboard 3D y CRISP-DM.
ACTIVE_PATH = os.path.join(ARTIFACT_DIR, "active.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CLASSES = ["low", "moderate", "high", "critical"]

#: Determinantes sociales, acceso y conductas: las entradas (X) del modelo.
FEATURES: List[str] = [
    "pct_no_health_insurance",
    "pct_food_insecurity",
    "pct_food_stamps",
    "pct_housing_insecurity",
    "pct_utility_shutoff",
    "pct_no_transportation",
    "pct_lack_social_support",
    "pct_loneliness",
    "pct_routine_checkup",
    "pct_cholesterol_screening",
    "pct_smoking",
    "pct_inactivity",
    "pct_binge_drinking",
]

#: Resultados de salud que se pueden predecir (la salida y).
TARGETS: List[str] = [
    "pct_fair_poor_health",
    "pct_diabetes",
    "pct_obesity",
    "pct_high_bp",
    "pct_chd",
    "pct_depression",
    "pct_mental_distress",
]
DEFAULT_TARGET = "pct_fair_poor_health"

#: Qué mide cada objetivo (texto para interpretar la predicción y los reportes).
TARGET_INFO: Dict[str, str] = {
    "pct_fair_poor_health": "adultos que califican su salud como regular o mala",
    "pct_diabetes": "adultos con diabetes diagnosticada",
    "pct_obesity": "adultos con obesidad (IMC >= 30)",
    "pct_high_bp": "adultos con hipertensión arterial",
    "pct_chd": "adultos con enfermedad coronaria",
    "pct_depression": "adultos con depresión diagnosticada",
    "pct_mental_distress": "adultos con malestar mental frecuente (14+ días al mes)",
}

LEVEL_NAMES = {"low": "Bajo", "moderate": "Moderado", "high": "Alto", "critical": "Crítico"}
LEVEL_MEANING = {
    "low": "entre el 25 % de tracts con menor prevalencia del área de estudio",
    "moderate": "por debajo de la mediana del área de estudio, pero no entre los mejores",
    "high": "por encima de la mediana del área de estudio",
    "critical": "entre el 25 % de tracts con mayor prevalencia del área de estudio (prioridad de intervención)",
}


def _candidates() -> Dict[str, Dict[str, Any]]:
    """Los cuatro modelos comparados, cada uno con su rejilla de hiperparámetros."""

    def pipe(model) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

    return {
        "logistic_regression": {
            "name": "Regresión Logística",
            "role": "Línea base lineal e interpretable",
            "pipeline": pipe(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            "grid": {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        },
        "random_forest": {
            "name": "Random Forest",
            "role": "Ensamble de árboles (bagging), robusto a no linealidades",
            "pipeline": pipe(RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
            "grid": {"model__max_depth": [None, 12], "model__min_samples_leaf": [1, 3]},
        },
        "gradient_boosting": {
            "name": "Gradient Boosting",
            "role": "Boosting de árboles (tipo XGBoost), fuerte en datos tabulares",
            "pipeline": pipe(HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
            "grid": {"model__learning_rate": [0.05, 0.1], "model__max_depth": [None, 6]},
        },
        "mlp": {
            "name": "Red Neuronal (MLP)",
            "role": "Perceptrón multicapa, enfoque de deep learning",
            "pipeline": pipe(
                MLPClassifier(max_iter=1500, early_stopping=True, random_state=RANDOM_STATE)
            ),
            "grid": {
                "model__hidden_layer_sizes": [(32,), (64, 32)],
                "model__alpha": [1e-4, 1e-2],
            },
        },
    }


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

def _latest_year(db: Session, code: str) -> Optional[int]:
    return (
        db.query(func.max(SDOHIndicator.year))
        .join(IndicatorCatalog, IndicatorCatalog.id == SDOHIndicator.catalog_id)
        .filter(IndicatorCatalog.code == code)
        .scalar()
    )


def build_dataset(db: Session, year: Optional[int] = None) -> pd.DataFrame:
    """Tabla ancha tract × indicador con los valores reales cargados por el ETL.

    Cada indicador se toma del año pedido o, si no existe, de su año más
    reciente (CDC publica algunas medidas solo en años alternos).
    """
    codes = FEATURES + TARGETS
    rows = (
        db.query(
            CensusTract.geoid,
            CensusTract.name,
            IndicatorCatalog.code,
            SDOHIndicator.year,
            SDOHIndicator.value,
        )
        .join(SDOHIndicator, SDOHIndicator.tract_id == CensusTract.id)
        .join(IndicatorCatalog, IndicatorCatalog.id == SDOHIndicator.catalog_id)
        .filter(IndicatorCatalog.code.in_(codes))
        .all()
    )
    if not rows:
        return pd.DataFrame(columns=["geoid", "name"] + codes)

    df = pd.DataFrame(rows, columns=["geoid", "name", "code", "year", "value"])
    if year is not None:
        # Prioriza el año pedido; si falta, el más reciente disponible.
        df["prio"] = np.where(df["year"] == year, 1, 0)
        df = df.sort_values(["prio", "year"], ascending=False)
    else:
        df = df.sort_values("year", ascending=False)
    df = df.drop_duplicates(["geoid", "code"], keep="first")
    wide = df.pivot(index=["geoid", "name"], columns="code", values="value").reset_index()
    for c in codes:
        if c not in wide.columns:
            wide[c] = np.nan
    return wide[["geoid", "name"] + codes]


def _risk_labels(values: pd.Series, cuts: List[float]) -> pd.Series:
    """Cuartiles del resultado de salud → low / moderate / high / critical."""
    bins = [-np.inf] + list(cuts) + [np.inf]
    return pd.cut(values, bins=bins, labels=CLASSES).astype(str)


def level_for_value(metrics: Dict[str, Any], value: float) -> str:
    """Nivel que corresponde a un % del resultado de salud con los cortes del modelo.

    Es la fuente única de verdad del nivel: el regresor y el nivel nunca pueden
    contradecirse porque ambos salen de los mismos cortes.
    """
    cuts = metrics["class_cuts"]
    bounds = [cuts["moderate"], cuts["high"], cuts["critical"]]
    return CLASSES[bisect.bisect_right(bounds, float(value))]


def study_area(db: Session) -> Dict[str, Any]:
    """Condados que componen el área de estudio y cuántos tracts aporta cada uno.

    Los cuartiles se calculan sobre el área completa (no por condado), así que el
    texto de interpretación debe nombrar el área real y no un condado supuesto.
    """
    rows = (
        db.query(County.name, County.state_name, func.count(CensusTract.id))
        .join(CensusTract, CensusTract.county_id == County.id)
        .group_by(County.name, County.state_name)
        .order_by(func.count(CensusTract.id).desc())
        .all()
    )
    counties = [{"name": n, "state": s, "tracts": int(c)} for n, s, c in rows]
    total = sum(c["tracts"] for c in counties)
    if len(counties) == 1:
        label = counties[0]["name"]
    elif len(counties) > 1:
        label = "el área de estudio"
    else:
        label = "el área de estudio"
    return {"label": label, "counties": counties, "tracts": total}


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #

def train(db: Session, target: str = DEFAULT_TARGET, year: Optional[int] = None) -> Dict[str, Any]:
    """Entrena los 4 modelos, elige el mejor, lo guarda y devuelve las métricas."""
    # Con ~1.6k filas, los hilos de OpenMP/BLAS compiten entre sí dentro del
    # contenedor y multiplican el tiempo por ~10; en un solo hilo es mucho más rápido.
    with threadpool_limits(1):
        return _train(db, target, year)


def _train(db: Session, target: str, year: Optional[int]) -> Dict[str, Any]:
    if target not in TARGETS:
        raise ValueError(f"Objetivo no válido: {target}. Opciones: {', '.join(TARGETS)}")
    year = year or _latest_year(db, target)
    started = time.perf_counter()

    data = build_dataset(db, year)
    data = data[data[target].notna()]
    usable = [f for f in FEATURES if data[f].notna().any()]
    if len(data) < 50 or not usable:
        raise ValueError(
            "No hay datos suficientes para entrenar. Ejecuta primero el ETL (make seed-etl)."
        )

    X = data[usable]
    y_raw = data[target]
    X_train, X_test, yraw_train, yraw_test = train_test_split(
        X, y_raw, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=pd.qcut(y_raw, 4, labels=False, duplicates="drop"),
    )
    # Cortes de los cuartiles calculados SOLO con train para no filtrar el test.
    cuts = [float(q) for q in np.quantile(yraw_train, [0.25, 0.5, 0.75])]
    y_train = _risk_labels(yraw_train, cuts)
    y_test = _risk_labels(yraw_test, cuts)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results: List[Dict[str, Any]] = []
    fitted: Dict[str, Any] = {}

    for key, spec in _candidates().items():
        t0 = time.perf_counter()
        search = GridSearchCV(spec["pipeline"], spec["grid"], scoring="f1_macro", cv=cv)
        with warnings.catch_warnings():
            # Avisos de convergencia/optimizador sin efecto en el resultado.
            warnings.simplefilter("ignore")
            search.fit(X_train, y_train)
        est = search.best_estimator_
        fitted[key] = est

        pred = est.predict(X_test)
        proba = est.predict_proba(X_test)
        best_idx = search.best_index_
        results.append(
            {
                "key": key,
                "name": spec["name"],
                "role": spec["role"],
                "best_params": {
                    k.replace("model__", ""): (list(v) if isinstance(v, tuple) else v)
                    for k, v in search.best_params_.items()
                },
                "cv_f1_mean": round(float(search.cv_results_["mean_test_score"][best_idx]), 4),
                "cv_f1_std": round(float(search.cv_results_["std_test_score"][best_idx]), 4),
                "test_accuracy": round(float(accuracy_score(y_test, pred)), 4),
                "test_f1_macro": round(float(f1_score(y_test, pred, average="macro")), 4),
                "test_roc_auc": round(
                    float(roc_auc_score(y_test, proba, multi_class="ovr", labels=est.classes_)), 4
                ),
                "confusion_matrix": confusion_matrix(y_test, pred, labels=CLASSES).tolist(),
                "train_seconds": round(time.perf_counter() - t0, 2),
            }
        )

    # Selección por F1-macro en validación cruzada (el test queda como evaluación final).
    results.sort(key=lambda r: (r["cv_f1_mean"], r["test_f1_macro"]), reverse=True)
    best = results[0]
    for r in results:
        r["selected"] = r["key"] == best["key"]
    best_est = fitted[best["key"]]

    # Regresión complementaria: estima el % exacto del resultado de salud, no solo el nivel.
    regressor = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]
    )
    regressor.fit(X_train, yraw_train)
    reg_pred = regressor.predict(X_test)
    # Evalúa el regresor como estimador de nivel (no solo de %): el nivel que
    # consumirá la app sale de aquí, así que su desempeño es el que importa.
    est_test_level = _risk_labels(pd.Series(reg_pred, index=y_test.index), cuts)
    reg_metrics = {
        "name": "Regresión Ridge (estimación del %)",
        "r2_test": round(float(r2_score(yraw_test, reg_pred)), 4),
        "mae_test": round(float(mean_absolute_error(yraw_test, reg_pred)), 3),
        "level_agreement_test": round(
            float((est_test_level == y_test).mean()), 4
        ),
        "bias_by_level": {
            lvl: round(float((reg_pred[est_test_level.values == lvl] - yraw_test.values[est_test_level.values == lvl]).mean()), 3)
            for lvl in CLASSES
            if (est_test_level.values == lvl).sum() > 0
        },
    }

    # Concordancia entre el nivel del regresor y el voto del clasificador: se reporta
    # como diagnóstico, pero el nivel publicado es siempre el del regresor.
    clf_test = best_est.predict(X_test)
    reg_metrics["clf_vs_reg_level_agreement"] = round(
        float((est_test_level.values == clf_test).mean()), 4
    )

    imp = permutation_importance(
        best_est, X_test, y_test, scoring="f1_macro", n_repeats=5, random_state=RANDOM_STATE
    )
    names = _catalog_names(db, usable + [target])
    importance = sorted(
        (
            {"code": f, "name": names.get(f, f), "importance": round(float(m), 4)}
            for f, m in zip(usable, imp.importances_mean)
        ),
        key=lambda d: d["importance"],
        reverse=True,
    )

    metrics = {
        "target_info": TARGET_INFO.get(target, target),
        "target_stats": {
            "min": round(float(yraw_train.min()), 2),
            "median": round(float(yraw_train.median()), 2),
            "max": round(float(yraw_train.max()), 2),
            "quantiles": [round(float(q), 3) for q in np.quantile(yraw_train, np.linspace(0, 1, 101))],
        },
        "regressor": reg_metrics,
        "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source": "CDC PLACES (cwsq-ngmh) · valores por census tract cargados por el ETL",
        "year": year,
        "study_area": study_area(db),
        "level_source": "estimator",
        "target": {"code": target, "name": names.get(target, target)},
        "classes": CLASSES,
        "class_cuts": {"moderate": cuts[0], "high": cuts[1], "critical": cuts[2]},
        "features": [{"code": f, "name": names.get(f, f)} for f in usable],
        "feature_medians": {f: round(float(X_train[f].median()), 3) for f in usable},
        "n_samples": int(len(data)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "cv_folds": CV_FOLDS,
        "selection_metric": "F1-macro (validación cruzada estratificada)",
        "best_model": best["key"],
        "best_model_name": best["name"],
        "models": results,
        "feature_importance": importance,
        "total_seconds": round(time.perf_counter() - started, 2),
    }

    model_path, metrics_path = _paths(target)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(
        {"model": best_est, "regressor": regressor, "features": usable, "metrics": metrics}, model_path
    )
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    activate(target)
    return metrics


def _catalog_names(db: Session, codes: List[str]) -> Dict[str, str]:
    return dict(
        db.query(IndicatorCatalog.code, IndicatorCatalog.name)
        .filter(IndicatorCatalog.code.in_(codes))
        .all()
    )


# --------------------------------------------------------------------------- #
# Artefactos: un modelo por objetivo + el objetivo activo
# --------------------------------------------------------------------------- #

def _paths(target: str):
    folder = os.path.join(ARTIFACT_DIR, target)
    return os.path.join(folder, "best_model.joblib"), os.path.join(folder, "metrics.json")


def model_path(target: str) -> str:
    return _paths(target)[0]


def _migrate_legacy() -> None:
    """Mueve el best_model.joblib único de la versión anterior a su carpeta por objetivo."""
    legacy_model = os.path.join(ARTIFACT_DIR, "best_model.joblib")
    legacy_metrics = os.path.join(ARTIFACT_DIR, "metrics.json")
    if not os.path.exists(legacy_model):
        return
    target = joblib.load(legacy_model)["metrics"]["target"]["code"]
    new_model, new_metrics = _paths(target)
    os.makedirs(os.path.dirname(new_model), exist_ok=True)
    os.replace(legacy_model, new_model)
    if os.path.exists(legacy_metrics):
        os.replace(legacy_metrics, new_metrics)
    if not os.path.exists(ACTIVE_PATH):
        activate(target)


def trained_targets() -> List[str]:
    _migrate_legacy()
    return [t for t in TARGETS if os.path.exists(model_path(t))]


def active_target() -> Optional[str]:
    trained = trained_targets()
    try:
        with open(ACTIVE_PATH, encoding="utf-8") as fh:
            target = json.load(fh).get("target")
        if target in trained:
            return target
    except (OSError, ValueError):
        pass
    return trained[0] if trained else None


def activate(target: str) -> None:
    """Marca el modelo de `target` como el que consume el resto de la app."""
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(ACTIVE_PATH, "w", encoding="utf-8") as fh:
        json.dump({"target": target}, fh)


_cache: Dict[str, Any] = {}


def load_bundle(target: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Carga el modelo de un objetivo (por defecto el activo), en caché mientras no cambie."""
    target = target or active_target()
    if not target or not os.path.exists(model_path(target)):
        return None
    path = model_path(target)
    mtime = os.path.getmtime(path)
    cached = _cache.get(target)
    if not cached or cached[0] != mtime:
        _cache[target] = (mtime, joblib.load(path))
    return _cache[target][1]


def _require(target: Optional[str]) -> Dict[str, Any]:
    bundle = load_bundle(target)
    if not bundle:
        raise LookupError(
            f"No hay modelo entrenado para {target}." if target else "El modelo aún no ha sido entrenado."
        )
    return bundle


def status(target: Optional[str] = None) -> Dict[str, Any]:
    base = {
        "targets": TARGETS,
        "target_info": TARGET_INFO,
        "default_target": DEFAULT_TARGET,
        "trained_targets": trained_targets(),
        "active_target": active_target(),
    }
    bundle = load_bundle(target)
    if not bundle:
        return {**base, "trained": False, "requested_target": target}
    return {**base, "trained": True, **bundle["metrics"]}


# --------------------------------------------------------------------------- #
# Predicción
# --------------------------------------------------------------------------- #

def _predict_frame(bundle: Dict[str, Any], X: pd.DataFrame) -> List[Dict[str, Any]]:
    model = bundle["model"]
    pred = model.predict(X)
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    out = []
    for label, p in zip(pred, proba):
        probs = {c: round(float(p[classes.index(c)]), 4) if c in classes else 0.0 for c in CLASSES}
        out.append({"risk_level": str(label), "confidence": round(float(max(p)), 4), "probabilities": probs})
    return out


def _level_range(metrics: Dict[str, Any], level: str) -> Dict[str, Any]:
    """Banda del nivel. Las bandas extremas son abiertas: no tienen borde duro, el
    mín/máx observado es solo un referente (una estimación puede salir del soporte
    histórico sin que el nivel cambie)."""
    cuts = metrics["class_cuts"]
    stats = metrics["target_stats"]
    bounds = {
        "low": (stats["min"], cuts["moderate"]),
        "moderate": (cuts["moderate"], cuts["high"]),
        "high": (cuts["high"], cuts["critical"]),
        "critical": (cuts["critical"], stats["max"]),
    }[level]
    return {
        "min": round(bounds[0], 2),
        "max": round(bounds[1], 2),
        "open_min": level == "low",
        "open_max": level == "critical",
    }


def _range_text(rng: Dict[str, Any]) -> str:
    """Describe la banda respetando los bordes abiertos de las bandas extremas."""
    if rng.get("open_min") and rng.get("open_max"):
        return f"La banda de ese nivel es abierta (referente: {rng['min']:.1f} % a {rng['max']:.1f} %)."
    if rng.get("open_min"):
        return f"La banda de ese nivel llega hasta {rng['max']:.1f} % (sin cota inferior)."
    if rng.get("open_max"):
        return f"La banda de ese nivel arranca en {rng['min']:.1f} % (sin cota superior)."
    return f"La banda de ese nivel va de {rng['min']:.1f} % a {rng['max']:.1f} %."


def _percentile(metrics: Dict[str, Any], value: float) -> float:
    q = metrics["target_stats"]["quantiles"]
    return round(float(np.interp(value, q, np.linspace(0, 100, len(q)))), 1)


def _cut_gap(metrics: Dict[str, Any], value: float, level: str, other: str) -> float:
    """Distancia en pp entre la estimación y el corte que separa `level` de `other`."""
    if level == other:
        return 0.0
    cuts = metrics["class_cuts"]
    # El corte que separa dos niveles consecutivos es el borde superior del inferior.
    lower = min(CLASSES.index(level), CLASSES.index(other))
    cut = cuts[CLASSES[lower + 1]]
    return abs(float(value) - float(cut))


def predict(features: Dict[str, Optional[float]], target: Optional[str] = None) -> Dict[str, Any]:
    """Predicción detallada de un escenario (valores faltantes → mediana de entrenamiento).

    El nivel publicado sale SIEMPRE de la estimación puntual del regresor usando los
    mismos cortes que definen las bandas, de modo que nivel y % no pueden contradecirse.
    El voto del clasificador se conserva como verificación cruzada y se señala
    explícitamente cuando no coincide (habitual en escenarios cerca de un corte).
    """
    bundle = _require(target)
    metrics = bundle["metrics"]
    cols = bundle["features"]
    medians = metrics["feature_medians"]
    names = {f["code"]: f["name"] for f in metrics["features"]}
    row = {c: features.get(c) if features.get(c) is not None else medians.get(c) for c in cols}
    frame = pd.DataFrame([row], columns=cols)

    classifier = _predict_frame(bundle, frame)[0]
    unit = metrics["target_info"]
    stats = metrics["target_stats"]
    mae = metrics["regressor"]["mae_test"]

    regressor = bundle.get("regressor")
    estimate: Optional[float] = None
    if regressor is not None:
        # Se redondea antes de derivar el nivel para que el % publicado y la banda
        # que lo contiene no puedan contradecirse por redondeo.
        estimate = float(round(float(np.clip(regressor.predict(frame)[0], 0, 100)), 2))

    classifier_level = classifier["risk_level"]
    rng_est = _level_range(metrics, level_for_value(metrics, estimate)) if estimate is not None else None
    if estimate is not None:
        level = level_for_value(metrics, estimate)
        coherence = "aligned" if classifier_level == level else "discrepancy"
        cut_gap = _cut_gap(metrics, estimate, level, classifier_level)
        detail: Dict[str, Any] = {
            "level_range": rng_est,
            "level_source": "estimator",
            "coherence": coherence,
            "cut_gap_pp": round(cut_gap, 2),
        }
    else:
        level = classifier_level
        coherence = "unavailable"
        detail = {
            "level_range": _level_range(metrics, level),
            "level_source": "classifier",
            "coherence": coherence,
            "cut_gap_pp": 0.0,
        }

    if estimate is not None:
        drivers = []
        for c in cols:
            alt = dict(row)
            alt[c] = medians[c]
            base_alt = float(regressor.predict(pd.DataFrame([alt], columns=cols))[0])
            drivers.append(
                {
                    "code": c,
                    "name": names.get(c, c),
                    "value": round(float(row[c]), 2),
                    "median": medians[c],
                    "effect_pp": round(estimate - base_alt, 2),
                }
            )
        drivers.sort(key=lambda d: abs(d["effect_pp"]), reverse=True)
        detail.update(
            {
                "estimated_value": round(estimate, 2),
                "estimate_interval": [round(max(estimate - mae, 0), 2), round(min(estimate + mae, 100), 2)],
                "percentile": _percentile(metrics, estimate),
                "area_median": stats["median"],
                "drivers": drivers,
            }
        )

    rng = detail["level_range"]
    if estimate is not None:
        diff = estimate - stats["median"]
        text = (
            f"La estimación puntual es {estimate:.1f} % de {unit} (±{mae:.1f} puntos), "
            f"lo que sitúa este escenario en nivel {LEVEL_NAMES[level].upper()}: "
            f"{LEVEL_MEANING[level]}. {_range_text(rng)} Supera al "
            f"{detail['percentile']:.0f} % de los tracts del área "
            f"de estudio y queda {abs(diff):.1f} puntos "
            f"{'por encima' if diff >= 0 else 'por debajo'} de su mediana ({stats['median']:.1f} %)."
        )
    else:
        text = (
            f"El modelo {metrics['best_model_name']} clasifica este escenario en nivel "
            f"{LEVEL_NAMES[level].upper()} con {classifier['confidence'] * 100:.1f} % de "
            f"confianza: se espera {_range_text(rng)} de {unit}."
        )

    if estimate is not None:
        up = [d for d in detail["drivers"] if d["effect_pp"] > 0.05][:3]
        down = [d for d in detail["drivers"] if d["effect_pp"] < -0.05][:3]
        if up:
            text += " Lo que más eleva el riesgo: " + ", ".join(
                f"{d['name']} ({d['value']:.1f} vs mediana {d['median']:.1f}; +{d['effect_pp']:.2f} pp)"
                for d in up
            ) + "."
        if down:
            text += " Lo que más lo reduce: " + ", ".join(
                f"{d['name']} ({d['value']:.1f} vs mediana {d['median']:.1f}; {d['effect_pp']:.2f} pp)"
                for d in down
            ) + "."
        if coherence == "discrepancy":
            text += (
                f" El clasificador de niveles ({metrics['best_model_name']}) vota "
                f"{LEVEL_NAMES[classifier_level].upper()} con "
                f"{classifier['confidence'] * 100:.0f} % de confianza: la estimación queda a "
                f"{detail['cut_gap_pp']:.2f} pp del corte entre ambos niveles, dentro del margen "
                f"de error (±{mae:.1f} pp), por lo que el nivel es sensible a ese umbral."
            )

    return {
        "model": metrics["best_model_name"],
        "target": metrics["target"],
        "target_info": unit,
        "study_area": metrics.get("study_area", {}),
        "inputs": row,
        "risk_level": level,
        "level_name": LEVEL_NAMES[level],
        "level_meaning": LEVEL_MEANING[level],
        "confidence": classifier["confidence"],
        "probabilities": classifier["probabilities"],
        "classifier_level": classifier_level,
        "classifier_level_name": LEVEL_NAMES[classifier_level],
        "classifier_model": metrics["best_model_name"],
        **detail,
        "interpretation": text,
    }


def predict_tracts(
    db: Session, year: Optional[int] = None, target: Optional[str] = None
) -> Dict[str, Any]:
    """Predicción del modelo de `target` (o el activo) para todos los tracts."""
    bundle = _require(target)
    metrics = bundle["metrics"]
    cols = bundle["features"]
    target = metrics["target"]["code"]
    cuts = metrics["class_cuts"]

    data = build_dataset(db, year or metrics["year"])
    if data.empty:
        return {"model": metrics["best_model_name"], "target": metrics["target"], "items": [], "summary": {}}
    preds = _predict_frame(bundle, data[cols])
    observed = _risk_labels(data[target], [cuts["moderate"], cuts["high"], cuts["critical"]])
    regressor = bundle.get("regressor")
    estimates = regressor.predict(data[cols]) if regressor is not None else [None] * len(data)

    items = []
    for (_, r), p, obs, est in zip(data.iterrows(), preds, observed, estimates):
        has_obs = pd.notna(r[target])
        clf_level = p["risk_level"]
        if est is not None:
            # Mismo criterio que `predict()`: el nivel publicado sale del estimador,
            # redondeado, para que el % mostrado y su banda no se contradigan.
            est = round(float(np.clip(est, 0, 100)), 2)
            level = level_for_value(metrics, est)
            est = float(est)
        else:
            level = clf_level

        items.append(
            {
                "geoid": r["geoid"],
                "name": r["name"],
                "observed_value": round(float(r[target]), 2) if has_obs else None,
                "observed_level": obs if has_obs else None,
                "estimated_value": round(est, 2) if est is not None else None,
                "risk_level": level,
                "level_name": LEVEL_NAMES[level],
                "level_source": "estimator" if est is not None else "classifier",
                "classifier_level": clf_level,
                "classifier_level_name": LEVEL_NAMES[clf_level],
                "coherence": "aligned" if clf_level == level else "discrepancy",
                "confidence": p["confidence"],
                "probabilities": p["probabilities"],
            }
        )
    summary = {c: sum(1 for i in items if i["risk_level"] == c) for c in CLASSES}
    with_obs = [i for i in items if i["observed_level"]]
    agreement = (
        round(100 * sum(i["risk_level"] == i["observed_level"] for i in with_obs) / len(with_obs), 2)
        if with_obs
        else None
    )
    clf_agreement = (
        round(100 * sum(i["classifier_level"] == i["observed_level"] for i in with_obs) / len(with_obs), 2)
        if with_obs
        else None
    )
    level_source = "estimator" if any(i["estimated_value"] is not None for i in items) else "classifier"
    return {
        "model": metrics["best_model_name"],
        "target": metrics["target"],
        "level_source": level_source,
        "summary": summary,
        "agreement_pct": agreement,
        "classifier_agreement_pct": clf_agreement,
        "coherence_pct": round(100 * sum(i["coherence"] == "aligned" for i in items) / len(items), 2)
        if items
        else None,
        "items": items,
    }
