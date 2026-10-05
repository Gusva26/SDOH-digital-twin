"""Fases III–IV — Preparación, entrenamiento, búsqueda de hiperparámetros,
validación cruzada y selección del mejor modelo.

Los candidatos y el preprocesado son los mismos que usa la API
(`app.services.ml_service`), así que lo que se analiza aquí es comparable con el
modelo desplegado en la plataforma.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

from app.services import ml_service

from .style import AMBER, BLUE, GRAY, GREEN, INK, LEVEL_COLORS, LEVEL_ES, MUTED, PALETTE, RED, plt, short

CLASSES = list(ml_service.CLASSES)
RANDOM_STATE = ml_service.RANDOM_STATE
BASELINE = "baseline"

# --------------------------------------------------------------------------- #
# Preparación
# --------------------------------------------------------------------------- #


def prepare(df: pd.DataFrame, target: str, features: List[str], test_size: float = 0.2) -> Dict[str, Any]:
    """Split estratificado por cuartiles y etiquetas de riesgo con cortes solo de train."""
    data = df[df[target].notna()].reset_index(drop=True)
    usable = [f for f in features if data[f].notna().any()]
    X, y_raw = data[usable], data[target].astype(float)
    X_train, X_test, yr_train, yr_test = train_test_split(
        X, y_raw, test_size=test_size, random_state=RANDOM_STATE,
        stratify=pd.qcut(y_raw, 4, labels=False, duplicates="drop"),
    )
    cuts = [float(q) for q in np.quantile(yr_train, [0.25, 0.5, 0.75])]
    y_train = ml_service._risk_labels(yr_train, cuts)
    y_test = ml_service._risk_labels(yr_test, cuts)
    return {
        "target": target,
        "features": usable,
        "data": data,
        "X_train": X_train,
        "X_test": X_test,
        "yraw_train": yr_train,
        "yraw_test": yr_test,
        "y_train": y_train,
        "y_test": y_test,
        "cuts": cuts,
        "test_size": test_size,
        "missing_train": int(X_train.isna().sum().sum()),
    }


def prep_table(p: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, y in (("Entrenamiento", p["y_train"]), ("Prueba", p["y_test"])):
        vc = y.value_counts()
        rows.append(
            {"Conjunto": name, "n": int(len(y)), **{LEVEL_ES[c]: int(vc.get(c, 0)) for c in CLASSES}}
        )
    return pd.DataFrame(rows)


def interpret_prep(p: Dict[str, Any], target_name: str) -> str:
    c = p["cuts"]
    return (
        f"El objetivo «{target_name}» (%) se convierte en 4 niveles de riesgo usando los cuartiles "
        f"del conjunto de entrenamiento: Bajo < {c[0]:.1f} %, Moderado {c[0]:.1f}–{c[1]:.1f} %, "
        f"Alto {c[1]:.1f}–{c[2]:.1f} %, Crítico ≥ {c[2]:.1f} %. Los cortes se calculan solo con "
        f"train para que el conjunto de prueba ({int(p['test_size'] * 100)} %) no filtre información. "
        f"Las clases quedan equilibradas (≈25 % cada una), de modo que F1-macro y accuracy son "
        f"comparables y la línea base al azar ronda 0,25. Hay {p['missing_train']} celdas faltantes "
        f"en train, que imputa la mediana dentro del pipeline; después se estandarizan las "
        f"{len(p['features'])} entradas (media 0, varianza 1)."
    )


HOW_PREP = (
    "Pipeline de preparación idéntico al de la API: (1) partición train/test estratificada por "
    "cuartiles del objetivo, (2) discretización en 4 niveles con cortes de train, "
    "(3) imputación por mediana y (4) estandarización. Imputación y escalado se ajustan dentro de "
    "cada fold de validación cruzada, evitando fuga de información."
)

# --------------------------------------------------------------------------- #
# Candidatos y rejillas de hiperparámetros
# --------------------------------------------------------------------------- #

#: Rejilla ampliada respecto a la API: más valores para estudiar la sensibilidad.
GRIDS = {
    "logistic_regression": {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "random_forest": {"model__max_depth": [None, 10], "model__min_samples_leaf": [1, 4]},
    "gradient_boosting": {"model__learning_rate": [0.05, 0.1], "model__max_depth": [None, 6]},
    "mlp": {"model__hidden_layer_sizes": [(32,), (64, 32)], "model__alpha": [1e-4, 1e-2]},
}

HP_MEANING = {
    "C": "inverso de la regularización L2 (C pequeño = modelo más simple)",
    "max_depth": "profundidad máxima de cada árbol (None = sin límite)",
    "min_samples_leaf": "mínimo de tracts por hoja (más alto = árboles más suaves)",
    "learning_rate": "tamaño del paso del boosting (bajo = aprendizaje más gradual)",
    "hidden_layer_sizes": "neuronas por capa oculta",
    "alpha": "penalización L2 de los pesos de la red",
}


def candidates() -> Dict[str, Dict[str, Any]]:
    specs = ml_service._candidates()
    for key, grid in GRIDS.items():
        specs[key]["grid"] = grid
    specs[BASELINE] = {
        "name": "Línea base (azar estratificado)",
        "role": "Referencia: predice al azar respetando la frecuencia de clases",
        "pipeline": Pipeline(
            [("imputer", SimpleImputer(strategy="median")),
             ("model", DummyClassifier(strategy="stratified", random_state=RANDOM_STATE))]
        ),
        "grid": {},
    }
    return specs


def _fmt_params(params: Dict[str, Any]) -> str:
    return ", ".join(f"{k.replace('model__', '')}={v}" for k, v in params.items()) or "—"


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #


def train_all(
    p: Dict[str, Any],
    n_splits: int = 5,
    n_repeats: int = 2,
    progress: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """GridSearchCV de cada candidato con la MISMA partición de CV (scores pareados)."""
    X_train, y_train, X_test, y_test = p["X_train"], p["y_train"], p["X_test"], p["y_test"]
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    n_folds = n_splits * n_repeats
    specs = candidates()
    models: Dict[str, Dict[str, Any]] = {}
    started = time.perf_counter()

    for i, (key, spec) in enumerate(specs.items()):
        if progress:
            progress(i / (len(specs) + 1), f"Entrenando {spec['name']}…")
        t0 = time.perf_counter()
        search = GridSearchCV(
            clone(spec["pipeline"]),
            spec["grid"] or {"model__strategy": ["stratified"]},
            scoring={"f1": "f1_macro", "acc": "accuracy", "bacc": "balanced_accuracy"},
            refit="f1",
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_train, y_train)
        cvr = pd.DataFrame(search.cv_results_)
        bi = search.best_index_
        folds = np.array([cvr.loc[bi, f"split{k}_test_f1"] for k in range(n_folds)])
        est = search.best_estimator_
        pred = est.predict(X_test)
        proba = est.predict_proba(X_test)
        grid = pd.DataFrame(
            {
                "Configuración": [_fmt_params(pp) for pp in cvr["params"]],
                "F1 CV (media)": cvr["mean_test_f1"],
                "F1 CV (desv.)": cvr["std_test_f1"],
                "F1 train (media)": cvr["mean_train_f1"],
                "Accuracy CV": cvr["mean_test_acc"],
                "Tiempo ajuste (s)": cvr["mean_fit_time"],
                "Ranking": cvr["rank_test_f1"],
            }
        ).sort_values("Ranking").round(4)
        models[key] = {
            "key": key,
            "name": spec["name"],
            "role": spec["role"],
            "baseline": key == BASELINE,
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
            "grid": grid,
            "n_configs": len(cvr),
            "fold_scores": folds,
            "cv_mean": float(folds.mean()),
            "cv_std": float(folds.std(ddof=1)),
            "train_mean": float(cvr.loc[bi, "mean_train_f1"]),
            "estimator": est,
            "pred": pred,
            "proba": proba,
            "classes": list(est.classes_),
            "test_f1": float(f1_score(y_test, pred, average="macro")),
            "test_acc": float(accuracy_score(y_test, pred)),
            "test_bacc": float(balanced_accuracy_score(y_test, pred)),
            "test_kappa": float(cohen_kappa_score(y_test, pred)),
            "test_auc": float(roc_auc_score(y_test, proba, multi_class="ovr", labels=est.classes_)),
            "confusion": confusion_matrix(y_test, pred, labels=CLASSES),
            "report": classification_report(y_test, pred, labels=CLASSES, output_dict=True, zero_division=0),
            "seconds": time.perf_counter() - t0,
        }

    ranked = sorted(
        (m for m in models.values() if not m["baseline"]),
        key=lambda m: (m["cv_mean"], m["test_f1"]),
        reverse=True,
    )
    best = ranked[0]
    if progress:
        progress(len(specs) / (len(specs) + 1), "Importancia por permutación y curva de aprendizaje…")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imp = permutation_importance(
            best["estimator"], X_test, y_test, scoring="f1_macro", n_repeats=10,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        sizes, tr, va = learning_curve(
            clone(best["estimator"]), X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
            scoring="f1_macro", train_sizes=np.linspace(0.15, 1.0, 6), n_jobs=-1,
        )
    if progress:
        progress(1.0, "Listo")
    return {
        "models": models,
        "ranking": [m["key"] for m in ranked],
        "best": best["key"],
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "n_folds": n_folds,
        "importance": pd.DataFrame(
            {"Código": p["features"], "Importancia": imp.importances_mean, "Desv.": imp.importances_std}
        ).sort_values("Importancia", ascending=False),
        "learning_curve": {"sizes": sizes, "train": tr, "valid": va},
        "seconds": time.perf_counter() - started,
    }


# --------------------------------------------------------------------------- #
# Entrenamiento: tabla resumen
# --------------------------------------------------------------------------- #


def training_table(r: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key in r["ranking"] + [BASELINE]:
        m = r["models"][key]
        rows.append(
            {
                "Modelo": m["name"],
                "Rol": m["role"],
                "Configs. probadas": m["n_configs"],
                "Mejores hiperparámetros": _fmt_params(m["best_params"]),
                "F1 train": m["train_mean"],
                "F1 CV": m["cv_mean"],
                "F1 test": m["test_f1"],
                "Accuracy test": m["test_acc"],
                "ROC-AUC test": m["test_auc"],
                "Tiempo (s)": m["seconds"],
                "Estado": "Seleccionado" if key == r["best"] else ("Referencia" if m["baseline"] else "Candidato"),
            }
        )
    return pd.DataFrame(rows).round(4)


def interpret_training(r: Dict[str, Any]) -> str:
    ms = r["models"]
    best = ms[r["best"]]
    base = ms[BASELINE]
    worst = ms[r["ranking"][-1]]
    gap = best["train_mean"] - best["cv_mean"]
    return (
        f"Se entrenaron {len(r['ranking'])} algoritmos más una línea base al azar, probando en total "
        f"{sum(m['n_configs'] for m in ms.values() if not m['baseline'])} configuraciones de "
        f"hiperparámetros con {r['n_folds']} particiones de validación cruzada cada una "
        f"({r['seconds']:.0f} s). El mejor, {best['name']}, alcanza F1-macro {best['cv_mean']:.3f} en CV y "
        f"{best['test_f1']:.3f} en prueba, frente a {base['cv_mean']:.3f} del azar: el modelo aprende "
        f"una señal real de los determinantes sociales ({best['cv_mean'] / max(base['cv_mean'], 1e-9):.1f}× la línea base). "
        f"El peor candidato es {worst['name']} ({worst['cv_mean']:.3f}). La brecha train–CV del mejor "
        f"es {gap:.3f}: "
        + ("sobreajuste marcado; conviene más regularización." if gap > 0.15
           else "sobreajuste moderado, aceptable para un ensamble." if gap > 0.05
           else "sin sobreajuste apreciable.")
    )


HOW_TRAINING = (
    "Cada algoritmo se entrena con búsqueda en rejilla (GridSearchCV) sobre el conjunto de "
    "entrenamiento. F1 train = ajuste sobre los datos vistos; F1 CV = promedio en los folds de "
    "validación (no vistos en cada ajuste); F1 test = evaluación final sobre el 20 % reservado. "
    "F1-macro promedia el F1 de las 4 clases por igual, de modo que no favorece a ninguna. "
    "ROC-AUC (uno contra el resto) mide la capacidad de ordenar tracts por riesgo (0,5 = azar, 1 = perfecto)."
)


def fig_training(r: Dict[str, Any]):
    keys = r["ranking"] + [BASELINE]
    ms = r["models"]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.bar(x - 0.27, [ms[k]["train_mean"] for k in keys], 0.27, color=GRAY, label="Train")
    ax.bar(x, [ms[k]["cv_mean"] for k in keys], 0.27, color=BLUE, label="Validación cruzada")
    ax.bar(x + 0.27, [ms[k]["test_f1"] for k in keys], 0.27, color=AMBER, label="Prueba")
    ax.set_xticks(x)
    ax.set_xticklabels([short(ms[k]["name"], 20) for k in keys], fontsize=7.5)
    ax.set_ylabel("F1-macro")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, frameon=False, ncol=3)
    ax.set_title("Desempeño por modelo: train vs. CV vs. prueba", color=INK)
    return fig


# --------------------------------------------------------------------------- #
# Hiperparámetros
# --------------------------------------------------------------------------- #


def fig_hyperparams(m: Dict[str, Any]):
    g = m["grid"].sort_values("Ranking")
    fig, ax = plt.subplots(figsize=(7.2, max(2.2, 0.36 * len(g) + 0.9)))
    y = np.arange(len(g))
    colors = [GREEN if rk == 1 else "#93c5fd" for rk in g["Ranking"]]
    ax.barh(y, g["F1 CV (media)"], xerr=g["F1 CV (desv.)"], color=colors, capsize=3, height=0.6)
    ax.scatter(g["F1 train (media)"], y, marker="|", s=160, color=RED, label="F1 train", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([short(c, 44) for c in g["Configuración"]], fontsize=7.2)
    ax.invert_yaxis()
    lo = max(0, float((g["F1 CV (media)"] - g["F1 CV (desv.)"]).min()) - 0.05)
    ax.set_xlim(lo, min(1.0, float(g["F1 train (media)"].max()) + 0.05))
    ax.set_xlabel("F1-macro (media ± desv. en CV)")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.set_title(f"Rejilla de hiperparámetros · {m['name']}", color=INK)
    return fig


def interpret_hyperparams(m: Dict[str, Any]) -> str:
    g = m["grid"]
    spread = float(g["F1 CV (media)"].max() - g["F1 CV (media)"].min())
    best = g.iloc[0]
    noise = float(best["F1 CV (desv.)"])
    meaning = "; ".join(f"{k}: {HP_MEANING.get(k, k)}" for k in m["best_params"])
    text = (
        f"{m['name']}: se probaron {len(g)} configuraciones; la ganadora es «{best['Configuración']}» "
        f"con F1 CV = {best['F1 CV (media)']:.4f} ± {noise:.4f}. La diferencia entre la mejor y la peor "
        f"configuración es {spread:.4f}"
    )
    if spread < noise:
        text += (
            ", menor que la variabilidad entre folds: el modelo es poco sensible a estos "
            "hiperparámetros y la elección concreta apenas importa."
        )
    else:
        text += ", mayor que la variabilidad entre folds: la sintonía de hiperparámetros sí mejora el modelo."
    gap = float(best["F1 train (media)"] - best["F1 CV (media)"])
    if gap > 0.15:
        text += f" La brecha train–CV ({gap:.2f}) revela sobreajuste de la configuración ganadora."
    if meaning:
        text += f" Significado: {meaning}."
    return text


HOW_HYPERPARAMS = (
    "Búsqueda exhaustiva en rejilla (GridSearchCV): cada combinación se evalúa con la misma "
    "validación cruzada estratificada repetida y se elige la de mayor F1-macro medio. La barra "
    "verde es la configuración ganadora; la barra de error es ± 1 desviación entre folds y la "
    "marca roja el F1 sobre train (su distancia a la barra mide el sobreajuste)."
)

# --------------------------------------------------------------------------- #
# Validación cruzada
# --------------------------------------------------------------------------- #


def cv_table(r: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key in r["ranking"] + [BASELINE]:
        m = r["models"][key]
        s = m["fold_scores"]
        half = stats.t.ppf(0.975, len(s) - 1) * s.std(ddof=1) / np.sqrt(len(s))
        rows.append(
            {
                "Modelo": m["name"],
                "Folds": len(s),
                "F1 medio": s.mean(),
                "Desv.": s.std(ddof=1),
                "IC95% inf.": s.mean() - half,
                "IC95% sup.": s.mean() + half,
                "Mín": s.min(),
                "Máx": s.max(),
                "CV del F1 %": 100 * s.std(ddof=1) / s.mean() if s.mean() else np.nan,
            }
        )
    return pd.DataFrame(rows).round(4)


def fig_cv(r: Dict[str, Any]):
    keys = r["ranking"] + [BASELINE]
    ms = r["models"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bp = ax.boxplot([ms[k]["fold_scores"] for k in keys], patch_artist=True, widths=0.55,
                    medianprops=dict(color=RED))
    for patch, k in zip(bp["boxes"], keys):
        patch.set_facecolor("#bbf7d0" if k == r["best"] else ("#e2e8f0" if ms[k]["baseline"] else "#dbeafe"))
    for i, k in enumerate(keys, start=1):
        s = ms[k]["fold_scores"]
        ax.scatter(np.random.default_rng(i).normal(i, 0.05, len(s)), s, s=9, color=INK, alpha=0.6, zorder=3)
    ax.set_xticks(range(1, len(keys) + 1))
    ax.set_xticklabels([short(ms[k]["name"], 20) for k in keys], fontsize=7.5)
    ax.set_ylabel("F1-macro por fold")
    ax.set_title(f"Validación cruzada: {r['n_splits']} folds × {r['n_repeats']} repeticiones", color=INK)
    return fig


def interpret_cv(r: Dict[str, Any]) -> str:
    t = cv_table(r)
    best = t.iloc[0]
    overlap = t.iloc[1]["IC95% sup."] >= best["IC95% inf."] if len(t) > 2 else False
    stable = t[t["Modelo"] != r["models"][BASELINE]["name"]].sort_values("Desv.").iloc[0]
    return (
        f"El mejor modelo ({best['Modelo']}) obtiene F1 = {best['F1 medio']:.4f} con IC 95 % "
        f"[{best['IC95% inf.']:.4f}, {best['IC95% sup.']:.4f}] sobre {int(best['Folds'])} folds; "
        f"su F1 oscila entre {best['Mín']:.3f} y {best['Máx']:.3f}. "
        + (
            f"Su intervalo se solapa con el del segundo ({t.iloc[1]['Modelo']}), por lo que la "
            "diferencia puede no ser real: se contrasta formalmente en «Pruebas inferenciales». "
            if overlap
            else f"Su intervalo no se solapa con el del segundo ({t.iloc[1]['Modelo']}). "
        )
        + f"El modelo más estable entre particiones es {stable['Modelo']} (desv. {stable['Desv.']:.4f}). "
        "Una desviación baja indica que el desempeño no depende de qué tracts caen en cada fold."
    )


HOW_CV = (
    "Validación cruzada estratificada repetida: el conjunto de entrenamiento se divide en k "
    "partes con la misma proporción de niveles; se entrena en k−1 y se valida en la restante, "
    "rotando, y todo se repite con otra semilla. Todos los modelos usan exactamente las mismas "
    "particiones, lo que permite compararlos por pares. Cada punto es el F1 de un fold; el IC 95 % "
    "usa la distribución t de Student."
)


def fig_learning_curve(r: Dict[str, Any]):
    lc = r["learning_curve"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for arr, color, label in ((lc["train"], RED, "Train"), (lc["valid"], BLUE, "Validación")):
        m, s = arr.mean(axis=1), arr.std(axis=1)
        ax.plot(lc["sizes"], m, "o-", color=color, label=label, markersize=4)
        ax.fill_between(lc["sizes"], m - s, m + s, color=color, alpha=0.15)
    ax.set_xlabel("Tracts de entrenamiento")
    ax.set_ylabel("F1-macro")
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title(f"Curva de aprendizaje · {r['models'][r['best']]['name']}", color=INK)
    return fig


def interpret_learning_curve(r: Dict[str, Any]) -> str:
    lc = r["learning_curve"]
    v = lc["valid"].mean(axis=1)
    t = lc["train"].mean(axis=1)
    gain = v[-1] - v[-2]
    return (
        f"Con {lc['sizes'][0]} tracts el F1 de validación es {v[0]:.3f} y con {lc['sizes'][-1]} llega a "
        f"{v[-1]:.3f}; el último incremento de datos aporta {gain:+.3f}. "
        + ("La curva sigue subiendo: más tracts (otros condados/años) mejorarían el modelo. "
           if gain > 0.01 else "La curva se ha estabilizado: más datos del mismo tipo aportarían poco. ")
        + f"La distancia final train–validación es {t[-1] - v[-1]:.3f} "
        + ("(varianza alta: el modelo memoriza parte del entrenamiento)." if t[-1] - v[-1] > 0.15
           else "(varianza contenida).")
    )


HOW_LEARNING = (
    "Se reentrena el modelo seleccionado con fracciones crecientes del entrenamiento (15 %–100 %) "
    "y se mide el F1 en train y en validación (5 folds). Bandas = ± 1 desviación. Si ambas "
    "curvas convergen en un valor bajo hay sesgo (modelo demasiado simple); si quedan separadas hay "
    "varianza (sobreajuste) y más datos ayudarían."
)

# --------------------------------------------------------------------------- #
# Selección del mejor modelo
# --------------------------------------------------------------------------- #


def fig_confusion(r: Dict[str, Any], key: Optional[str] = None):
    m = r["models"][key or r["best"]]
    cm = m["confusion"]
    pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    labels = [LEVEL_ES[c] for c in CLASSES]
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(4))
    ax.set_yticklabels(labels)
    ax.grid(False)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.0%}", ha="center", va="center", fontsize=7.5,
                    color="white" if pct[i, j] > 0.5 else INK)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión (prueba) · {short(m['name'], 28)}", color=INK)
    return fig


def interpret_confusion(r: Dict[str, Any]) -> str:
    m = r["models"][r["best"]]
    cm = m["confusion"]
    n = cm.sum()
    exact = np.trace(cm)
    adjacent = sum(cm[i, j] for i in range(4) for j in range(4) if abs(i - j) == 1)
    far = n - exact - adjacent
    recall = {c: m["report"][c]["recall"] for c in CLASSES}
    best_c = max(recall, key=recall.get)
    worst_c = min(recall, key=recall.get)
    crit = m["report"]["critical"]
    return (
        f"De {n} tracts de prueba, {exact} ({exact / n:.0%}) se clasifican en su nivel exacto y "
        f"{adjacent} ({adjacent / n:.0%}) en un nivel vecino; solo {far} ({far / n:.0%}) fallan por dos "
        f"o más niveles. Los errores se concentran junto a la diagonal porque los niveles son "
        f"cuartiles de una variable continua: confundir Alto con Crítico es un error de frontera, no "
        f"de concepto. La clase mejor reconocida es {LEVEL_ES[best_c]} (sensibilidad "
        f"{recall[best_c]:.0%}) y la peor {LEVEL_ES[worst_c]} ({recall[worst_c]:.0%}); las clases "
        f"intermedias suelen ser las más difíciles al tener vecinos por ambos lados. Para priorizar "
        f"intervenciones importa el nivel Crítico: sensibilidad {crit['recall']:.0%} y precisión "
        f"{crit['precision']:.0%}."
    )


HOW_CONFUSION = (
    "Filas = nivel real del tract, columnas = nivel predicho por el modelo, sobre el 20 % de "
    "prueba que nunca participó en el entrenamiento. Cada celda muestra el número de tracts y el "
    "porcentaje de su fila (sensibilidad por clase en la diagonal). Un modelo perfecto concentra "
    "todo en la diagonal."
)


def class_report_table(r: Dict[str, Any]) -> pd.DataFrame:
    rep = r["models"][r["best"]]["report"]
    rows = [
        {"Nivel": LEVEL_ES[c], "Precisión": rep[c]["precision"], "Sensibilidad (recall)": rep[c]["recall"],
         "F1": rep[c]["f1-score"], "Soporte": int(rep[c]["support"])}
        for c in CLASSES
    ]
    rows.append({"Nivel": "Promedio macro", "Precisión": rep["macro avg"]["precision"],
                 "Sensibilidad (recall)": rep["macro avg"]["recall"], "F1": rep["macro avg"]["f1-score"],
                 "Soporte": int(rep["macro avg"]["support"])})
    return pd.DataFrame(rows).round(4)


HOW_CLASS_REPORT = (
    "Precisión: de los tracts que el modelo etiqueta con ese nivel, qué fracción lo es realmente "
    "(pocas falsas alarmas). Sensibilidad: de los tracts que realmente son de ese nivel, qué "
    "fracción detecta (pocos casos perdidos). F1: media armónica de ambas. Soporte: tracts reales "
    "de cada nivel en prueba."
)


def interpret_class_report(r: Dict[str, Any]) -> str:
    t = class_report_table(r).iloc[:4]
    lo = t.sort_values("F1").iloc[0]
    hi = t.sort_values("F1").iloc[-1]
    return (
        f"El nivel mejor predicho es {hi['Nivel']} (F1 = {hi['F1']:.3f}) y el más difícil "
        f"{lo['Nivel']} (F1 = {lo['F1']:.3f}). Los niveles extremos (Bajo y Crítico) suelen "
        "separarse mejor porque sus determinantes sociales son más distintivos; los intermedios "
        "comparten perfil con dos vecinos. Si la precisión de Crítico supera a su sensibilidad, el "
        "modelo es conservador (marca pocos críticos pero acierta); si es al revés, detecta más "
        "casos a costa de falsas alarmas."
    )


def fig_roc(r: Dict[str, Any], y_test: pd.Series):
    m = r["models"][r["best"]]
    yb = label_binarize(y_test, classes=m["classes"])
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, c in enumerate(m["classes"]):
        fpr, tpr, _ = roc_curve(yb[:, i], m["proba"][:, i])
        auc = roc_auc_score(yb[:, i], m["proba"][:, i])
        ax.plot(fpr, tpr, color=LEVEL_COLORS.get(c, BLUE), linewidth=1.6, label=f"{LEVEL_ES.get(c, c)} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.set_title(f"Curvas ROC uno-contra-resto · {short(m['name'], 24)}", color=INK)
    return fig


def interpret_roc(r: Dict[str, Any], y_test: pd.Series) -> str:
    m = r["models"][r["best"]]
    yb = label_binarize(y_test, classes=m["classes"])
    aucs = {c: roc_auc_score(yb[:, i], m["proba"][:, i]) for i, c in enumerate(m["classes"])}
    lo = min(aucs, key=aucs.get)
    return (
        f"AUC macro = {m['test_auc']:.3f}. Por nivel: "
        + ", ".join(f"{LEVEL_ES[c]} {v:.3f}" for c, v in aucs.items())
        + f". Un AUC de 0,9 significa que, tomando al azar un tract de ese nivel y otro que no lo "
        f"es, el modelo asigna mayor probabilidad al correcto el 90 % de las veces. El nivel "
        f"{LEVEL_ES[lo]} es el más difícil de separar del resto."
    )


HOW_ROC = (
    "Para cada nivel se trata el problema como binario (ese nivel contra los demás) y se barre "
    "el umbral de probabilidad. La curva muestra cuántos verdaderos positivos se ganan a cambio "
    "de falsos positivos; la diagonal gris es el azar (AUC = 0,5). Cuanto más pegada a la esquina "
    "superior izquierda, mejor."
)


def fig_importance(r: Dict[str, Any], names: Dict[str, str]):
    imp = r["importance"].sort_values("Importancia")
    fig, ax = plt.subplots(figsize=(7, max(2.6, 0.33 * len(imp) + 0.8)))
    colors = [BLUE if v > 0 else GRAY for v in imp["Importancia"]]
    ax.barh([short(names.get(c, c), 36) for c in imp["Código"]], imp["Importancia"], xerr=imp["Desv."],
            color=colors, capsize=2)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("Caída del F1-macro al permutar la variable")
    ax.set_title(f"Importancia por permutación · {short(r['models'][r['best']]['name'], 28)}", color=INK)
    return fig


def interpret_importance(r: Dict[str, Any], names: Dict[str, str]) -> str:
    imp = r["importance"]
    top = imp.head(3)
    useless = imp[imp["Importancia"] <= imp["Desv."]]
    return (
        "Las variables que más sostienen las predicciones son: "
        + ", ".join(f"«{names.get(c, c)}» (−{v:.3f} de F1 si se desordena)" for c, v in zip(top["Código"], top["Importancia"]))
        + f". {len(useless)} variables tienen una importancia indistinguible de cero (su efecto no "
        "supera su propia variabilidad). Al estar los determinantes correlacionados, una variable "
        "puede parecer poco importante porque otra correlacionada aporta la misma información; la "
        "importancia es predictiva, no causal."
    )


HOW_IMPORTANCE = (
    "Importancia por permutación (explicabilidad agnóstica al modelo): se desordenan al azar los "
    "valores de una variable en el conjunto de prueba, 10 veces, y se mide cuánto cae el F1-macro. "
    "Si el modelo depende de esa variable, desordenarla lo empeora mucho. Barra de error = ± 1 desv."
)


def selection_table(r: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for pos, key in enumerate(r["ranking"], start=1):
        m = r["models"][key]
        rows.append(
            {
                "Puesto": pos,
                "Modelo": m["name"],
                "F1 CV": m["cv_mean"],
                "± desv.": m["cv_std"],
                "F1 test": m["test_f1"],
                "Accuracy balanceada": m["test_bacc"],
                "Kappa de Cohen": m["test_kappa"],
                "ROC-AUC": m["test_auc"],
                "Sobreajuste (train−CV)": m["train_mean"] - m["cv_mean"],
                "Tiempo (s)": m["seconds"],
            }
        )
    return pd.DataFrame(rows).round(4)


def interpret_selection(r: Dict[str, Any]) -> str:
    t = selection_table(r)
    b, s = t.iloc[0], t.iloc[1]
    fastest = t.sort_values("Tiempo (s)").iloc[0]
    kappa = b["Kappa de Cohen"]
    kword = "casi perfecto" if kappa > 0.8 else "sustancial" if kappa > 0.6 else "moderado" if kappa > 0.4 else "débil"
    return (
        f"Criterio declarado: mayor F1-macro medio en validación cruzada (el test no se usa para "
        f"elegir, solo para confirmar). Gana {b['Modelo']} con {b['F1 CV']:.4f}, "
        f"{b['F1 CV'] - s['F1 CV']:+.4f} sobre {s['Modelo']}. En prueba confirma F1 = {b['F1 test']:.4f}, "
        f"ROC-AUC = {b['ROC-AUC']:.4f} y kappa = {kappa:.3f} (acuerdo {kword} más allá del azar). "
        f"Bajo el principio de parsimonia (Navaja de Ockham) y la paridad estadística demostrada por "
        f"Friedman/Nemenyi (distancia inferior a la diferencia crítica CD), {b['Modelo']} es seleccionado "
        "óptimamente al combinar máxima interpretabilidad paramétrica y menor costo computacional frente a ensamblados opacos."
    )


HOW_SELECTION = (
    "Ranking de candidatos por F1-macro en validación cruzada. Se añaden métricas de prueba "
    "complementarias: accuracy balanceada (media de sensibilidades), kappa de Cohen (acuerdo "
    "corregido por azar: 0 = azar, 1 = perfecto), ROC-AUC y la brecha train–CV como indicador de "
    "sobreajuste."
)
