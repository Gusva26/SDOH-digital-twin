"""Fase V — Pruebas estadísticas inferenciales robustas.

Todas son no paramétricas o de remuestreo, porque el EDA muestra que las variables
no son normales y tienen atípicos:

- Friedman + Kendall W + diferencia crítica de Nemenyi (¿difieren los modelos?).
- Wilcoxon por pares con corrección de Holm y t corregida de Nadeau-Bengio
  (dependencia entre folds de CV).
- McNemar exacto (ganador vs. segundo sobre las mismas predicciones de prueba).
- IC bootstrap del F1 de prueba y de la diferencia pareada.
- Prueba de permutación del modelo (¿aprende algo más que el azar?).
- Spearman con IC (Bonett-Wright) y Holm; Kruskal-Wallis con ε² por nivel de riesgo
  y del objetivo entre condados.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, permutation_test_score

from .modeling import BASELINE, CLASSES, RANDOM_STATE
from .style import BLUE, GRAY, GREEN, INK, MUTED, RED, plt, short

ALPHA = 0.05


def holm(pvals: List[float]) -> List[float]:
    """Ajuste de Holm-Bonferroni (controla el error de familia)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def _fmt_p(p: float) -> str:
    return "< 0,001" if p < 0.001 else f"{p:.4f}"


# --------------------------------------------------------------------------- #
# Comparación de modelos
# --------------------------------------------------------------------------- #


def friedman(r: Dict[str, Any]) -> Dict[str, Any]:
    keys = r["ranking"]
    ms = r["models"]
    M = np.column_stack([ms[k]["fold_scores"] for k in keys])  # folds × modelos
    n, k = M.shape
    stat, p = stats.friedmanchisquare(*M.T)
    ranks = np.apply_along_axis(lambda row: stats.rankdata(-row), 1, M).mean(axis=0)
    w = stat / (n * (k - 1))
    q = stats.studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * n))
    return {
        "keys": keys,
        "names": [ms[x]["name"] for x in keys],
        "stat": float(stat),
        "p": float(p),
        "kendall_w": float(w),
        "avg_ranks": ranks,
        "cd": float(cd),
        "n": n,
        "k": k,
    }


def friedman_table(f: Dict[str, Any]) -> pd.DataFrame:
    best = float(np.min(f["avg_ranks"]))
    return pd.DataFrame(
        {
            "Modelo": f["names"],
            "Rango medio (1 = mejor)": np.round(f["avg_ranks"], 3),
            "Distancia al mejor": np.round(f["avg_ranks"] - best, 3),
            "¿Distinguible del mejor? (Nemenyi)": [
                "—" if r == best else ("Sí" if r - best > f["cd"] else "No") for r in f["avg_ranks"]
            ],
        }
    )


def interpret_friedman(f: Dict[str, Any]) -> str:
    w = f["kendall_w"]
    wword = "grande" if w >= 0.5 else "moderado" if w >= 0.3 else "pequeño"
    tied = [n for n, r in zip(f["names"], f["avg_ranks"]) if 0 < r - min(f["avg_ranks"]) <= f["cd"]]
    text = (
        f"Friedman χ² = {f['stat']:.2f} con {f['k'] - 1} g.l., p = {_fmt_p(f['p'])} sobre {f['n']} folds. "
    )
    if f["p"] < ALPHA:
        text += (
            f"Se rechaza H0: al menos un modelo rinde distinto. El tamaño del efecto (W de Kendall "
            f"= {w:.3f}) es {wword}, es decir, la ordenación de modelos es {'muy ' if w >= 0.5 else ''}"
            f"consistente entre folds. Con la diferencia crítica de Nemenyi (CD = {f['cd']:.3f} rangos) "
        )
        text += (
            f"no se distinguen del mejor: {', '.join(tied)}; ese grupo es estadísticamente equivalente."
            if tied
            else "el mejor modelo se separa de todos los demás."
        )
    else:
        text += (
            "No se rechaza H0: con estos folds no hay evidencia de que los modelos difieran; "
            "cualquiera de ellos es defendible y conviene preferir el más simple o interpretable."
        )
    return text


HOW_FRIEDMAN = (
    "Prueba de Friedman: alternativa no paramétrica al ANOVA de medidas repetidas. En cada fold "
    "se ordenan los modelos (1 = mejor F1) y se contrasta H0 «todos los modelos tienen el mismo "
    "rango medio». W de Kendall (0–1) es el tamaño del efecto. Post-hoc de Nemenyi: dos modelos "
    "difieren si sus rangos medios se separan más que la diferencia crítica CD (Demšar, 2006)."
)


def fig_cd(f: Dict[str, Any]):
    order = np.argsort(f["avg_ranks"])
    ranks = f["avg_ranks"][order]
    names = [f["names"][i] for i in order]
    fig, ax = plt.subplots(figsize=(7, 2.2 + 0.25 * len(names)))
    y = np.arange(len(names))
    ax.scatter(ranks, y, color=BLUE, s=40, zorder=3)
    best = ranks[0]
    ax.axvspan(best, best + f["cd"], color=GREEN, alpha=0.15, label=f"Zona equivalente al mejor (CD = {f['cd']:.2f})")
    for yi, rk in zip(y, ranks):
        ax.text(rk, yi + 0.18, f"{rk:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([short(n, 30) for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.8, f["k"] + 0.2)
    ax.set_xlabel("Rango medio en los folds (1 = mejor)")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.set_title(f"Diferencia crítica de Nemenyi · Friedman p = {_fmt_p(f['p'])}", color=INK)
    return fig


def interpret_cd(f: Dict[str, Any]) -> str:
    best = float(np.min(f["avg_ranks"]))
    order = np.argsort(f["avg_ranks"])
    inside = [f["names"][i] for i in order[1:] if f["avg_ranks"][i] - best <= f["cd"]]
    outside = [f["names"][i] for i in order[1:] if f["avg_ranks"][i] - best > f["cd"]]
    text = f"{f['names'][order[0]]} ocupa el mejor rango medio ({best:.2f}). "
    if inside:
        text += f"Dentro de la banda (equivalentes al mejor): {', '.join(inside)}. "
    if outside:
        text += f"Fuera de la banda (significativamente peores): {', '.join(outside)}. "
    text += (
        "Si varios modelos comparten banda, la elección entre ellos puede basarse en criterios "
        "secundarios como interpretabilidad o coste computacional."
        if inside else "El ganador es el único en su banda: su superioridad es estadísticamente clara."
    )
    if f["p"] >= ALPHA:
        text += " Nota: Friedman no fue significativo, así que el post-hoc es solo descriptivo."
    return text


HOW_CD = (
    "Cada punto es el rango medio de un modelo en los folds de validación cruzada. La banda "
    "verde se extiende desde el mejor modelo una distancia igual a la diferencia crítica de "
    "Nemenyi: los modelos dentro de la banda no son estadísticamente distinguibles del ganador."
)


def pairwise(r: Dict[str, Any], n_train: int, n_test_fold: int) -> pd.DataFrame:
    """Wilcoxon pareado + t corregida de Nadeau-Bengio para cada par de candidatos."""
    ms = r["models"]
    keys = r["ranking"]
    pairs = list(combinations(keys, 2)) + [(r["best"], BASELINE)]
    rows = []
    ratio = n_test_fold / max(n_train, 1)
    for a, b in pairs:
        d = ms[a]["fold_scores"] - ms[b]["fold_scores"]
        try:
            w = stats.wilcoxon(ms[a]["fold_scores"], ms[b]["fold_scores"], zero_method="zsplit")
            wp = float(w.pvalue)
        except ValueError:
            wp = 1.0
        J = len(d)
        var = d.var(ddof=1)
        t = d.mean() / np.sqrt((1 / J + ratio) * var) if var > 0 else 0.0
        tp = float(2 * stats.t.sf(abs(t), J - 1))
        # r de rango biserial emparejado (tamaño de efecto del Wilcoxon)
        nz = d[d != 0]
        rb = 0.0
        if nz.size:
            rk = stats.rankdata(np.abs(nz))
            rb = float((rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum())
        rows.append(
            {
                "Modelo A": ms[a]["name"],
                "Modelo B": ms[b]["name"],
                "Δ F1 medio (A−B)": d.mean(),
                "Wilcoxon p": wp,
                "t Nadeau-Bengio": t,
                "p corregida N-B": tp,
                "r rango-biserial": rb,
            }
        )
    out = pd.DataFrame(rows)
    out["Wilcoxon p (Holm)"] = holm(out["Wilcoxon p"].tolist())
    out["N-B p (Holm)"] = holm(out["p corregida N-B"].tolist())
    out["¿Diferencia significativa?"] = np.where(
        (out["Wilcoxon p (Holm)"] < ALPHA) & (out["N-B p (Holm)"] < ALPHA),
        "Sí (ambas)",
        np.where((out["Wilcoxon p (Holm)"] < ALPHA) | (out["N-B p (Holm)"] < ALPHA), "Solo una prueba", "No"),
    )
    return out.round(4)


def interpret_pairwise(t: pd.DataFrame, best_name: str) -> str:
    vs_best = t[(t["Modelo A"] == best_name) | (t["Modelo B"] == best_name)]
    sig = vs_best[vs_best["¿Diferencia significativa?"] == "Sí (ambas)"]
    ns = vs_best[vs_best["¿Diferencia significativa?"] != "Sí (ambas)"]
    text = (
        f"Se contrastan {len(t)} pares con dos pruebas y corrección de Holm por comparaciones "
        f"múltiples. {best_name} supera de forma robusta (ambas pruebas) a: "
        + (", ".join(
            (row["Modelo B"] if row["Modelo A"] == best_name else row["Modelo A"]) for _, row in sig.iterrows()
        ) or "ninguno")
        + ". "
    )
    if len(ns):
        others = [(row["Modelo B"] if row["Modelo A"] == best_name else row["Modelo A"]) for _, row in ns.iterrows()]
        text += (
            f"Frente a {', '.join(others)} la diferencia no es concluyente: la t corregida de "
            "Nadeau-Bengio es más conservadora que Wilcoxon porque tiene en cuenta que los folds "
            "comparten datos de entrenamiento. "
        )
    text += (
        "El r rango-biserial (−1 a 1) indica en qué proporción de folds gana A: |r| > 0,5 es un "
        "efecto grande."
    )
    return text


HOW_PAIRWISE = (
    "Wilcoxon de rangos con signo: compara los F1 de dos modelos fold a fold (pareado, sin "
    "suponer normalidad). t corregida de Nadeau-Bengio (2003): corrige la varianza por el "
    "solapamiento entre conjuntos de entrenamiento de la CV, que hace que la t clásica tenga "
    "demasiados falsos positivos. Holm ajusta los p-valores por hacer muchas comparaciones. Se "
    "declara diferencia robusta solo si ambas pruebas la respaldan."
)


def mcnemar(r: Dict[str, Any], y_test: pd.Series) -> Dict[str, Any]:
    ms = r["models"]
    a, b = ms[r["ranking"][0]], ms[r["ranking"][1]]
    y = np.asarray(y_test)
    ca, cb = a["pred"] == y, b["pred"] == y
    n11, n10 = int((ca & cb).sum()), int((ca & ~cb).sum())
    n01, n00 = int((~ca & cb).sum()), int((~ca & ~cb).sum())
    disc = n10 + n01
    p_exact = float(stats.binomtest(min(n10, n01), disc, 0.5).pvalue) if disc else 1.0
    chi2 = (abs(n10 - n01) - 1) ** 2 / disc if disc else 0.0
    return {
        "a": a["name"], "b": b["name"], "n11": n11, "n10": n10, "n01": n01, "n00": n00,
        "p_exact": p_exact, "chi2": float(chi2), "p_chi2": float(stats.chi2.sf(chi2, 1)) if disc else 1.0,
    }


def mcnemar_table(m: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "": [f"{m['a']} acierta", f"{m['a']} falla"],
            f"{m['b']} acierta": [m["n11"], m["n01"]],
            f"{m['b']} falla": [m["n10"], m["n00"]],
        }
    )


def interpret_mcnemar(m: Dict[str, Any]) -> str:
    return (
        f"Sobre los mismos tracts de prueba, {m['a']} acierta donde {m['b']} falla en {m['n10']} casos y "
        f"al revés en {m['n01']}. Solo esas celdas discordantes informan de la diferencia. McNemar "
        f"exacto: p = {_fmt_p(m['p_exact'])} (χ² con corrección de continuidad = {m['chi2']:.2f}, "
        f"p = {_fmt_p(m['p_chi2'])}). "
        + (
            f"La diferencia en prueba es significativa: {m['a'] if m['n10'] > m['n01'] else m['b']} "
            "comete menos errores."
            if m["p_exact"] < ALPHA
            else "No es significativa: ambos modelos cometen una cantidad de errores compatible con el "
            "azar sobre este conjunto de prueba."
        )
    )


HOW_MCNEMAR = (
    "Tabla de contingencia 2×2 de aciertos/errores de los dos mejores modelos sobre el mismo "
    "conjunto de prueba. La prueba de McNemar (versión exacta binomial) contrasta si los "
    "desacuerdos se reparten 50/50; es la prueba recomendada por Dietterich (1998) para comparar "
    "dos clasificadores con un único conjunto de prueba."
)


def bootstrap_f1(r: Dict[str, Any], y_test: pd.Series, n_boot: int = 2000) -> Dict[str, Any]:
    y = np.asarray(y_test)
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    boots: Dict[str, np.ndarray] = {}
    rows = []
    for key in r["ranking"] + [BASELINE]:
        m = r["models"][key]
        pred = np.asarray(m["pred"])
        vals = np.array([f1_score(y[i], pred[i], average="macro") for i in idx])
        boots[key] = vals
        lo, hi = np.percentile(vals, [2.5, 97.5])
        rows.append({"Modelo": m["name"], "F1 test": m["test_f1"], "IC95% inf.": lo, "IC95% sup.": hi,
                     "Amplitud IC": hi - lo})
    a, b = r["ranking"][0], r["ranking"][1]
    diff = boots[a] - boots[b]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {
        "table": pd.DataFrame(rows).round(4),
        "boots": boots,
        "diff": {"a": r["models"][a]["name"], "b": r["models"][b]["name"], "mean": float(diff.mean()),
                 "lo": float(lo), "hi": float(hi), "p_le0": float((diff <= 0).mean())},
        "n_boot": n_boot,
    }


def fig_bootstrap(bs: Dict[str, Any], r: Dict[str, Any]):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, key in enumerate(r["ranking"]):
        ax.hist(bs["boots"][key], bins=40, alpha=0.45, label=short(r["models"][key]["name"], 24),
                color=[BLUE, RED, GREEN, "#7c3aed", GRAY][i % 5])
    ax.set_xlabel("F1-macro en prueba (réplicas bootstrap)")
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title(f"Distribución bootstrap del F1 ({bs['n_boot']} réplicas)", color=INK)
    return fig


def interpret_bootstrap(bs: Dict[str, Any]) -> str:
    d = bs["diff"]
    t = bs["table"].iloc[0]
    return (
        f"El F1 de prueba del mejor modelo ({t['Modelo']}) es {t['F1 test']:.4f} con IC 95 % bootstrap "
        f"[{t['IC95% inf.']:.4f}, {t['IC95% sup.']:.4f}]: esa es la incertidumbre por haber evaluado "
        f"sobre una muestra finita de tracts. La diferencia pareada {d['a']} − {d['b']} vale "
        f"{d['mean']:+.4f} con IC 95 % [{d['lo']:+.4f}, {d['hi']:+.4f}]; "
        + (
            "el intervalo excluye el 0, así que la ventaja es robusta."
            if d["lo"] > 0 or d["hi"] < 0
            else f"el intervalo incluye el 0 (en el {d['p_le0']:.0%} de las réplicas el segundo iguala o "
            "supera al primero), así que la ventaja no está garantizada."
        )
    )


HOW_BOOTSTRAP = (
    "Bootstrap percentil: se remuestrean con reemplazo los tracts de prueba 2.000 veces y se "
    "recalcula el F1 en cada réplica. El 2,5 % y 97,5 % de esas réplicas forman el IC 95 %. La "
    "diferencia entre modelos se calcula en las MISMAS réplicas (bootstrap pareado). No asume "
    "ninguna distribución."
)


def permutation_test(r: Dict[str, Any], p: Dict[str, Any], n_perm: int = 100) -> Dict[str, Any]:
    est = clone(r["models"][r["best"]]["estimator"])
    score, perm, pval = permutation_test_score(
        est, p["X_train"], p["y_train"], scoring="f1_macro",
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        n_permutations=n_perm, random_state=RANDOM_STATE, n_jobs=-1,
    )
    return {"score": float(score), "perm": perm, "p": float(pval), "n_perm": n_perm,
            "name": r["models"][r["best"]]["name"]}


def fig_permutation(pt: Dict[str, Any]):
    fig, ax = plt.subplots(figsize=(6.6, 3))
    ax.hist(pt["perm"], bins=25, color=GRAY, label="F1 con etiquetas permutadas (H0)")
    ax.axvline(pt["score"], color=RED, linewidth=2, label=f"F1 real = {pt['score']:.3f}")
    ax.set_xlabel("F1-macro (CV 5 folds)")
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title(f"Prueba de permutación · p = {_fmt_p(pt['p'])}", color=INK)
    return fig


def interpret_permutation(pt: Dict[str, Any]) -> str:
    return (
        f"Al desordenar al azar los niveles de riesgo {pt['n_perm']} veces, {pt['name']} obtiene de media "
        f"F1 = {np.mean(pt['perm']):.3f} (máximo {np.max(pt['perm']):.3f}), mientras que con las etiquetas "
        f"reales logra {pt['score']:.3f}. p = {_fmt_p(pt['p'])} (el mínimo alcanzable es "
        f"1/{pt['n_perm'] + 1} = {1 / (pt['n_perm'] + 1):.4f}): "
        + ("el modelo capta una relación real entre determinantes sociales y resultado de salud, "
           "no un artefacto del azar."
           if pt["p"] < ALPHA else "no hay evidencia de que el modelo supere al azar.")
    )


HOW_PERMUTATION = (
    "Prueba de permutación (Ojala & Garriga, 2010): se rompe la relación X–y permutando las "
    "etiquetas y se reentrena el modelo; esto construye la distribución del F1 bajo H0 «no hay "
    "relación». El p-valor es la fracción de permutaciones que igualan o superan el F1 real."
)

# --------------------------------------------------------------------------- #
# Asociación de los determinantes con el objetivo
# --------------------------------------------------------------------------- #


def spearman_table(p: Dict[str, Any], names: Dict[str, str]) -> pd.DataFrame:
    data = p["data"]
    y = data[p["target"]].astype(float)
    rows = []
    for f in p["features"]:
        mask = data[f].notna()
        n = int(mask.sum())
        rho, pv = stats.spearmanr(data.loc[mask, f], y[mask])
        se = np.sqrt((1 + rho ** 2 / 2) / (n - 3))
        z = np.arctanh(np.clip(rho, -0.9999, 0.9999))
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        rows.append({"Código": f, "Variable": names.get(f, f), "n": n, "ρ Spearman": rho,
                     "IC95% inf.": lo, "IC95% sup.": hi, "p": pv})
    out = pd.DataFrame(rows)
    out["p (Holm)"] = holm(out["p"].tolist())
    out["Fuerza"] = pd.cut(out["ρ Spearman"].abs(), [0, 0.1, 0.3, 0.5, 1.01],
                           labels=["Despreciable", "Débil", "Moderada", "Fuerte"], right=False).astype(str)
    out["p"] = out["p"].map(lambda v: float(f"{v:.3g}"))
    out["p (Holm)"] = out["p (Holm)"].map(lambda v: float(f"{v:.3g}"))
    return out.sort_values("ρ Spearman", key=np.abs, ascending=False).round(4)


def interpret_spearman(t: pd.DataFrame, target_name: str) -> str:
    sig = t[t["p (Holm)"] < ALPHA]
    pos = sig[sig["ρ Spearman"] > 0].head(3)
    neg = sig[sig["ρ Spearman"] < 0].head(3)
    text = f"{len(sig)} de {len(t)} determinantes se asocian significativamente con «{target_name}» tras Holm. "
    if len(pos):
        text += "Asociados a MAYOR prevalencia: " + ", ".join(
            f"«{v}» (ρ = {r:+.2f})" for v, r in zip(pos["Variable"], pos["ρ Spearman"])) + ". "
    if len(neg):
        text += "Asociados a MENOR prevalencia (factores protectores): " + ", ".join(
            f"«{v}» (ρ = {r:+.2f})" for v, r in zip(neg["Variable"], neg["ρ Spearman"])) + ". "
    text += (
        "Los intervalos estrechos reflejan el tamaño muestral; con n grande incluso ρ pequeñas son "
        "significativas, por eso se interpreta la fuerza (|ρ|) y no solo el p-valor. Son asociaciones "
        "ecológicas entre tracts, no efectos causales individuales."
    )
    return text


HOW_SPEARMAN = (
    "Correlación de rangos de Spearman entre cada determinante y el % del resultado de salud "
    "(no requiere normalidad ni linealidad). IC 95 % por transformación de Fisher con el error "
    "estándar de Bonett-Wright para Spearman. Holm controla los falsos positivos al contrastar "
    "todas las variables a la vez. Fuerza: < 0,1 despreciable, 0,1–0,3 débil, 0,3–0,5 moderada, ≥ 0,5 fuerte."
)


def fig_spearman(t: pd.DataFrame):
    t = t.sort_values("ρ Spearman")
    fig, ax = plt.subplots(figsize=(7, max(2.6, 0.33 * len(t) + 0.8)))
    y = np.arange(len(t))
    colors = [RED if v > 0 else BLUE for v in t["ρ Spearman"]]
    ax.errorbar(t["ρ Spearman"], y, xerr=[t["ρ Spearman"] - t["IC95% inf."], t["IC95% sup."] - t["ρ Spearman"]],
                fmt="none", ecolor=MUTED, capsize=3)
    ax.scatter(t["ρ Spearman"], y, color=colors, s=30, zorder=3)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([short(v, 34) for v in t["Variable"]], fontsize=7.5)
    ax.set_xlabel("ρ de Spearman con el objetivo (IC 95 %)")
    ax.set_title("Asociación de cada determinante con el resultado de salud", color=INK)
    return fig


def kruskal_levels(p: Dict[str, Any], names: Dict[str, str]) -> pd.DataFrame:
    data = p["data"]
    levels = pd.concat([p["y_train"], p["y_test"]]).reindex(data.index)
    rows = []
    for f in p["features"]:
        groups = [data.loc[(levels == c) & data[f].notna(), f].values for c in CLASSES]
        groups = [g for g in groups if len(g)]
        if len(groups) < 2:
            continue
        h, pv = stats.kruskal(*groups)
        n = sum(len(g) for g in groups)
        k = len(groups)
        eps = max(0.0, (h - k + 1) / (n - k))
        rows.append({"Código": f, "Variable": names.get(f, f), "H": h, "p": pv, "ε²": eps,
                     **{f"Mediana {c}": float(np.median(g)) for c, g in zip(["Bajo", "Moderado", "Alto", "Crítico"], groups)}})
    out = pd.DataFrame(rows)
    out["p (Holm)"] = holm(out["p"].tolist())
    out["Efecto"] = pd.cut(out["ε²"], [0, 0.01, 0.08, 0.26, 1.01],
                           labels=["Despreciable", "Pequeño", "Mediano", "Grande"], right=False).astype(str)
    out["p"] = out["p"].map(lambda v: float(f"{v:.3g}"))
    out["p (Holm)"] = out["p (Holm)"].map(lambda v: float(f"{v:.3g}"))
    return out.sort_values("ε²", ascending=False).round(4)


def interpret_kruskal(t: pd.DataFrame) -> str:
    big = t[t["Efecto"] == "Grande"]
    top = t.iloc[0]
    return (
        f"{int((t['p (Holm)'] < ALPHA).sum())} de {len(t)} determinantes difieren significativamente entre "
        f"los cuatro niveles de riesgo. {len(big)} con efecto grande (ε² ≥ 0,26). El más discriminante es "
        f"«{top['Variable']}» (ε² = {top['ε²']:.3f}): su mediana pasa de {top['Mediana Bajo']:.1f} % en tracts "
        f"de riesgo Bajo a {top['Mediana Crítico']:.1f} % en los Críticos. Estas variables son las que "
        "permiten al modelo separar los niveles, y coinciden en buena medida con la importancia por permutación."
    )


HOW_KRUSKAL = (
    "Kruskal-Wallis: alternativa no paramétrica al ANOVA de un factor. Contrasta si la "
    "distribución de un determinante es la misma en los 4 niveles de riesgo (H0). ε² = (H − k + 1)/(n − k) "
    "es el tamaño del efecto: proporción de la variabilidad en rangos explicada por el nivel "
    "(0,01 pequeño, 0,08 mediano, 0,26 grande). Se muestran las medianas por nivel para leer la dirección."
)


def kruskal_counties(p: Dict[str, Any], names: Dict[str, str]) -> Dict[str, Any]:
    data = p["data"]
    if "county" not in data.columns:
        return {"applicable": False}
    g = data.groupby("county")[p["target"]]
    groups = [v.dropna().values for _k, v in g if v.notna().sum() >= 5]
    labels = [k for k, v in g if v.notna().sum() >= 5]
    if len(groups) < 2:
        return {"applicable": False}
    h, pv = stats.kruskal(*groups)
    n, k = sum(map(len, groups)), len(groups)
    table = pd.DataFrame(
        {"Condado": labels, "Tracts": [len(x) for x in groups], "Mediana": [np.median(x) for x in groups],
         "RIC": [np.subtract(*np.percentile(x, [75, 25])) for x in groups]}
    ).sort_values("Mediana", ascending=False).round(3)
    return {"applicable": True, "H": float(h), "p": float(pv), "eps2": max(0.0, (h - k + 1) / (n - k)),
            "table": table, "k": k, "n": n}


def interpret_counties(kc: Dict[str, Any], target_name: str) -> str:
    t = kc["table"]
    return (
        f"Kruskal-Wallis H = {kc['H']:.2f}, p = {_fmt_p(kc['p'])}, ε² = {kc['eps2']:.3f} con {kc['k']} condados. "
        + (
            f"La prevalencia de «{target_name}» difiere entre condados: la mediana más alta es la de "
            f"{t.iloc[0]['Condado']} ({t.iloc[0]['Mediana']:.1f} %) y la más baja la de {t.iloc[-1]['Condado']} "
            f"({t.iloc[-1]['Mediana']:.1f} %). Por eso los niveles de riesgo son relativos al área de estudio "
            "completa y un modelo entrenado aquí debe recalibrarse antes de aplicarse a otro territorio."
            if kc["p"] < ALPHA
            else "No hay diferencias significativas entre condados: el área de estudio es homogénea en este resultado."
        )
    )


HOW_COUNTIES = (
    "Kruskal-Wallis del objetivo entre condados del área de estudio. Detecta heterogeneidad "
    "geográfica que afecta a la transferibilidad del modelo. Se muestran mediana y RIC por condado."
)
