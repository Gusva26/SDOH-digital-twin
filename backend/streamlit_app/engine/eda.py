"""Análisis exploratorio: tendencia central (media, mediana, moda), dispersión,
forma (asimetría y curtosis), valores atípicos, normalidad y correlación."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from .style import AMBER, BLUE, GREEN, INK, MUTED, RED, plt, short

# --------------------------------------------------------------------------- #
# Estadística descriptiva
# --------------------------------------------------------------------------- #


def _mode(s: pd.Series) -> tuple[float, int, int]:
    """Moda de una variable continua publicada con 1 decimal (como CDC PLACES).

    Devuelve (moda, frecuencia, número de modas empatadas).
    """
    counts = s.round(1).value_counts()
    if counts.empty:
        return float("nan"), 0, 0
    top = counts.iloc[0]
    tied = counts[counts == top]
    return float(tied.index.min()), int(top), int(len(tied))


def describe(df: pd.DataFrame, cols: List[str], names: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c].dropna().astype(float)
        if s.empty:
            continue
        mode, freq, n_modes = _mode(s)
        q1, q3 = s.quantile([0.25, 0.75])
        mean = s.mean()
        rows.append(
            {
                "Código": c,
                "Variable": names.get(c, c),
                "n": int(s.size),
                "Media": mean,
                "Mediana": s.median(),
                "Moda": mode,
                "Frec. moda": freq,
                "Nº modas": n_modes,
                "Media recortada 10%": stats.trim_mean(s, 0.10),
                "Desv. típica": s.std(ddof=1),
                "CV %": 100 * s.std(ddof=1) / mean if mean else np.nan,
                "Mín": s.min(),
                "Q1": q1,
                "Q3": q3,
                "Máx": s.max(),
                "RIC": q3 - q1,
                "MAD": stats.median_abs_deviation(s, scale="normal"),
                "Asimetría": stats.skew(s, bias=False),
                "Curtosis (exceso)": stats.kurtosis(s, fisher=True, bias=False),
            }
        )
    return pd.DataFrame(rows).round(3)


def _skew_word(g: float) -> str:
    if abs(g) < 0.5:
        return "aproximadamente simétrica"
    side = "derecha (cola de valores altos)" if g > 0 else "izquierda (cola de valores bajos)"
    return f"{'moderadamente' if abs(g) < 1 else 'fuertemente'} asimétrica a la {side}"


def _kurt_word(k: float) -> str:
    if k > 1:
        return "leptocúrtica (colas pesadas: más valores extremos que una normal)"
    if k < -1:
        return "platicúrtica (colas ligeras, distribución aplanada)"
    return "mesocúrtica (colas similares a una normal)"


def interpret_variable(row: pd.Series) -> str:
    mean, med, mode = row["Media"], row["Mediana"], row["Moda"]
    gap = mean - med
    text = (
        f"«{row['Variable']}»: media {mean:.2f}, mediana {med:.2f} y moda {mode:.1f} "
        f"(se repite {row['Frec. moda']} veces"
        + (f", con {row['Nº modas']} valores empatados" if row["Nº modas"] > 1 else "")
        + "). "
    )
    if abs(gap) < 0.05 * (row["Desv. típica"] or 1):
        text += "Media y mediana prácticamente coinciden, señal de una distribución centrada. "
    else:
        text += (
            f"La media está {abs(gap):.2f} puntos {'por encima' if gap > 0 else 'por debajo'} "
            f"de la mediana: los valores {'altos' if gap > 0 else 'bajos'} arrastran el promedio, "
            "por lo que la mediana es el resumen más representativo del tract típico. "
        )
    text += (
        f"La distribución es {_skew_word(row['Asimetría'])} (asimetría = {row['Asimetría']:.2f}) y "
        f"{_kurt_word(row['Curtosis (exceso)'])} (curtosis de exceso = {row['Curtosis (exceso)']:.2f}). "
        f"El coeficiente de variación es {row['CV %']:.1f} %: "
        + (
            "dispersión alta entre tracts (desigualdad territorial marcada)."
            if row["CV %"] > 30
            else "dispersión moderada."
            if row["CV %"] > 15
            else "dispersión baja (tracts homogéneos)."
        )
    )
    return text


def interpret_describe(d: pd.DataFrame) -> str:
    right = d[d["Asimetría"] >= 0.5]
    lepto = d[d["Curtosis (exceso)"] > 1]
    top_cv = d.sort_values("CV %", ascending=False).iloc[0]
    low_cv = d.sort_values("CV %").iloc[0]
    return (
        f"Se describen {len(d)} variables. {len(right)} presentan asimetría positiva relevante "
        f"(media > mediana) y {len(lepto)} son leptocúrticas, lo que anticipa valores atípicos "
        f"y falta de normalidad. La mayor desigualdad entre tracts está en «{top_cv['Variable']}» "
        f"(CV = {top_cv['CV %']:.1f} %) y la menor en «{low_cv['Variable']}» "
        f"(CV = {low_cv['CV %']:.1f} %). Cuando media y mediana divergen, conviene resumir con "
        "mediana/RIC y usar pruebas no paramétricas en la fase inferencial."
    )


HOW_DESCRIBE = (
    "Cada fila resume una variable sobre todos los census tracts. Tendencia central: media "
    "(promedio aritmético, sensible a extremos), mediana (valor central, robusta), moda (valor "
    "más frecuente, redondeado a 1 decimal como publica CDC) y media recortada al 10 % "
    "(promedio sin el 10 % más alto ni el más bajo). Dispersión: desviación típica, CV "
    "(desv./media), RIC (Q3−Q1) y MAD (desviación absoluta mediana escalada, robusta). Forma: "
    "asimetría (0 = simétrica; > 0 cola derecha) y curtosis de exceso (0 = normal; > 0 colas "
    "pesadas)."
)


def fig_histogram(s: pd.Series, name: str):
    s = s.dropna().astype(float)
    mode = _mode(s)[0]
    fig, ax = plt.subplots(figsize=(7, 3.3))
    ax.hist(s, bins=35, color="#bfdbfe", edgecolor="white", density=True)
    if s.nunique() > 2:
        xs = np.linspace(s.min(), s.max(), 300)
        ax.plot(xs, stats.gaussian_kde(s)(xs), color=BLUE, linewidth=1.6, label="Densidad (KDE)")
    ax.axvline(s.mean(), color=RED, linewidth=1.8, label=f"Media {s.mean():.2f}")
    ax.axvline(s.median(), color=GREEN, linewidth=1.8, linestyle="--", label=f"Mediana {s.median():.2f}")
    ax.axvline(mode, color=AMBER, linewidth=1.8, linestyle=":", label=f"Moda {mode:.1f}")
    ax.set_xlabel(f"{name} (%)")
    ax.set_ylabel("Densidad")
    ax.set_title(f"Distribución de {short(name, 50)}", color=INK)
    ax.legend(fontsize=7.5, frameon=False)
    return fig


HOW_HISTOGRAM = (
    "Histograma (barras) con la curva de densidad estimada por núcleos (KDE). Las líneas "
    "verticales marcan la media (roja), la mediana (verde discontinua) y la moda (ámbar "
    "punteada). Si las tres coinciden la distribución es simétrica; si la media queda a la "
    "derecha de la mediana hay una cola de tracts con valores altos."
)


def fig_boxplots(df: pd.DataFrame, cols: List[str], names: Dict[str, str]):
    """Boxplots de variables estandarizadas robustamente (mediana/RIC) para compararlas."""
    data = []
    for c in cols:
        s = df[c].dropna().astype(float)
        iqr = s.quantile(0.75) - s.quantile(0.25) or 1.0
        data.append(((s - s.median()) / iqr).values)
    fig, ax = plt.subplots(figsize=(7.2, max(3, 0.34 * len(cols) + 0.8)))
    ax.boxplot(
        data,
        vert=False,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="#dbeafe", edgecolor=BLUE),
        medianprops=dict(color=RED, linewidth=1.5),
        flierprops=dict(marker="o", markersize=2.5, markerfacecolor=AMBER, markeredgecolor="none", alpha=0.6),
    )
    ax.set_yticks(range(1, len(cols) + 1))
    ax.set_yticklabels([short(names.get(c, c), 34) for c in cols], fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("(valor − mediana) / RIC")
    ax.set_title("Diagrama de caja estandarizado (outliers en ámbar)", color=INK)
    return fig


def interpret_boxplots(df: pd.DataFrame, cols: List[str], names: Dict[str, str]) -> str:
    rows = []
    for c in cols:
        s = df[c].dropna().astype(float)
        q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
        iqr = (q3 - q1) or 1.0
        rows.append((c, (q3 - med) / iqr, (s.max() - med) / iqr, (med - s.min()) / iqr))
    upper = max(rows, key=lambda r: r[2])
    lopsided = [r for r in rows if r[1] > 0.6]
    return (
        f"Tras estandarizar, la cola superior más larga es la de «{names.get(upper[0], upper[0])}»: su tract "
        f"más alto está a {upper[2]:.1f} RIC por encima de la mediana. "
        f"{len(lopsided)} de {len(rows)} variables tienen la mediana desplazada hacia el borde inferior de "
        "la caja (más masa por encima), otra señal de asimetría positiva. Colas superiores largas indican "
        "tracts con carga social excepcional, que el modelo debe aprender a reconocer y no descartar."
    )


HOW_BOXPLOTS = (
    "Cada caja abarca el 50 % central de los tracts (Q1–Q3) y la línea roja es la mediana. "
    "Las variables se estandarizan como (x − mediana)/RIC para que sean comparables en la "
    "misma escala. Los bigotes llegan a 1,5·RIC; los puntos ámbar fuera de ellos son valores "
    "atípicos según el criterio de Tukey."
)


# --------------------------------------------------------------------------- #
# Valores atípicos
# --------------------------------------------------------------------------- #


def outliers(df: pd.DataFrame, cols: List[str], names: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c].dropna().astype(float)
        if s.size < 5:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mad = stats.median_abs_deviation(s)
        mz = 0.6745 * (s - s.median()) / mad if mad else pd.Series(0, index=s.index)
        z = (s - s.mean()) / (s.std(ddof=1) or 1)
        n_iqr = int(((s < lo) | (s > hi)).sum())
        rows.append(
            {
                "Código": c,
                "Variable": names.get(c, c),
                "Límite inf. (Tukey)": lo,
                "Límite sup. (Tukey)": hi,
                "Atípicos IQR": n_iqr,
                "% IQR": 100 * n_iqr / s.size,
                "Atípicos z-robusto (>3,5)": int((mz.abs() > 3.5).sum()),
                "Atípicos z clásico (>3)": int((z.abs() > 3).sum()),
                "Altos": int((s > hi).sum()),
                "Bajos": int((s < lo).sum()),
            }
        )
    return pd.DataFrame(rows).round(3)


def interpret_outliers(o: pd.DataFrame) -> str:
    if o.empty:
        return "No hay variables suficientes para detectar atípicos."
    worst = o.sort_values("% IQR", ascending=False).iloc[0]
    none = o[o["Atípicos IQR"] == 0]
    high_side = int(o["Altos"].sum())
    low_side = int(o["Bajos"].sum())
    text = (
        f"En total se marcan {int(o['Atípicos IQR'].sum())} valores atípicos por Tukey "
        f"({high_side} por arriba y {low_side} por abajo) y {int(o['Atípicos z-robusto (>3,5)'].sum())} "
        f"por z-robusto. La variable con más atípicos es «{worst['Variable']}» "
        f"({worst['Atípicos IQR']} tracts, {worst['% IQR']:.1f} %). "
    )
    if len(none):
        text += f"{len(none)} variables no tienen atípicos. "
    text += (
        "Los atípicos por arriba son tracts con carga social o sanitaria excepcional: no son "
        "errores de medición (CDC ya publica estimaciones suavizadas) sino los casos de mayor "
        "prioridad, por eso no se eliminan. Para que no dominen el ajuste se usan medianas en la "
        "imputación, escalado estándar dentro del pipeline y modelos de árboles robustos a extremos."
    )
    return text


HOW_OUTLIERS = (
    "Tres criterios complementarios: (1) Tukey/IQR — fuera de [Q1 − 1,5·RIC, Q3 + 1,5·RIC]; "
    "(2) z-robusto de Iglewicz-Hoaglin — |0,6745·(x − mediana)/MAD| > 3,5, que no se deja "
    "arrastrar por los propios atípicos; (3) z clásico — |x − media|/desv. > 3, válido solo si la "
    "variable es aproximadamente normal. Coincidencia entre criterios = atípico robusto."
)


def fig_outliers(o: pd.DataFrame):
    o = o.sort_values("% IQR", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, max(2.6, 0.32 * len(o) + 0.8)))
    y = np.arange(len(o))
    ax.barh(y - 0.2, o["Altos"], height=0.4, color=RED, label="Por encima (Tukey)")
    ax.barh(y + 0.2, o["Bajos"], height=0.4, color=BLUE, label="Por debajo (Tukey)")
    ax.set_yticks(y)
    ax.set_yticklabels([short(v, 34) for v in o["Variable"]], fontsize=7.5)
    ax.set_xlabel("Nº de tracts atípicos")
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("Valores atípicos por variable", color=INK)
    return fig


# --------------------------------------------------------------------------- #
# Normalidad
# --------------------------------------------------------------------------- #


def normality(df: pd.DataFrame, cols: List[str], names: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c].dropna().astype(float).values
        if s.size < 8:
            continue
        sw = stats.shapiro(s[:5000])
        k2 = stats.normaltest(s)
        ad = stats.anderson(s, dist="norm")
        crit5 = float(ad.critical_values[list(ad.significance_level).index(5.0)])
        rows.append(
            {
                "Código": c,
                "Variable": names.get(c, c),
                "Shapiro-Wilk W": sw.statistic,
                "p Shapiro": sw.pvalue,
                "D'Agostino K²": k2.statistic,
                "p D'Agostino": k2.pvalue,
                "Anderson-Darling A²": ad.statistic,
                "A² crítico 5%": crit5,
                "¿Normal? (α=0,05)": "Sí"
                if (sw.pvalue > 0.05 and k2.pvalue > 0.05 and ad.statistic < crit5)
                else "No",
            }
        )
    out = pd.DataFrame(rows)
    for c in ("p Shapiro", "p D'Agostino"):
        out[c] = out[c].map(lambda p: float(f"{p:.3g}"))
    return out.round(4)


def interpret_normality(n: pd.DataFrame, n_obs: int) -> str:
    non = n[n["¿Normal? (α=0,05)"] == "No"]
    text = (
        f"{len(non)} de {len(n)} variables rechazan la normalidad en al menos una de las tres "
        f"pruebas (α = 0,05). "
    )
    if n_obs > 500:
        text += (
            f"Con n ≈ {n_obs} tracts las pruebas detectan desviaciones pequeñas; por eso se "
            "complementan con el tamaño de la asimetría/curtosis y el gráfico Q-Q. "
        )
    text += (
        "Conclusión metodológica: los supuestos de las pruebas paramétricas clásicas (t, ANOVA, "
        "Pearson) no se sostienen, y la fase inferencial usa alternativas robustas: Spearman, "
        "Kruskal-Wallis, Friedman/Nemenyi, Wilcoxon, McNemar exacto, bootstrap y permutaciones."
        if len(non) > len(n) / 2
        else "La mayoría es compatible con normalidad; aun así se prefieren pruebas robustas por "
        "la presencia de atípicos."
    )
    return text


HOW_NORMALITY = (
    "Tres pruebas con H0 = «la variable sigue una distribución normal». Shapiro-Wilk (la más "
    "potente en muestras moderadas), D'Agostino-Pearson K² (combina asimetría y curtosis) y "
    "Anderson-Darling (sensible a las colas; se rechaza si A² supera su valor crítico al 5 %). "
    "p < 0,05 → se rechaza la normalidad."
)


def fig_qq(s: pd.Series, name: str):
    s = s.dropna().astype(float).values
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    (osm, osr), (slope, inter, _r) = stats.probplot(s, dist="norm")
    ax.scatter(osm, osr, s=6, color=BLUE, alpha=0.6)
    ax.plot(osm, slope * np.asarray(osm) + inter, color=RED, linewidth=1.4)
    ax.set_xlabel("Cuantiles teóricos N(0,1)")
    ax.set_ylabel("Cuantiles observados")
    ax.set_title(f"Q-Q normal · {short(name, 36)}", color=INK)
    return fig


HOW_QQ = (
    "Cada punto enfrenta un cuantil observado con el que tendría una normal. Si los puntos "
    "siguen la recta roja la variable es normal; una curva en forma de «U» o de «J» indica "
    "asimetría y puntos que se separan en los extremos indican colas pesadas (atípicos)."
)


def interpret_qq(s: pd.Series, name: str) -> str:
    s = s.dropna().astype(float)
    g, k = stats.skew(s), stats.kurtosis(s)
    tail = "se separan de la recta en los extremos (colas más pesadas que la normal)" if k > 1 else (
        "se mantienen cerca de la recta en los extremos"
    )
    bend = (
        "se curvan hacia arriba a la derecha (asimetría positiva)"
        if g > 0.5
        else "se curvan hacia abajo a la izquierda (asimetría negativa)"
        if g < -0.5
        else "siguen en el centro la recta"
    )
    return f"En «{name}» los puntos {bend} y {tail}; asimetría {g:.2f}, curtosis de exceso {k:.2f}."


# --------------------------------------------------------------------------- #
# Correlación
# --------------------------------------------------------------------------- #


def correlation(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return df[cols].corr(method="spearman")


def fig_correlation(corr: pd.DataFrame, names: Dict[str, str]):
    n = len(corr)
    fig, ax = plt.subplots(figsize=(min(9, 0.45 * n + 3), min(8, 0.42 * n + 2.4)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [short(names.get(c, c), 22) for c in corr.columns]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=6.8)
    ax.grid(False)
    if n <= 22:
        for i in range(n):
            for j in range(n):
                v = corr.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.2,
                        color="white" if abs(v) > 0.6 else INK)
    fig.colorbar(im, ax=ax, fraction=0.04, label="ρ de Spearman")
    ax.set_title("Matriz de correlación de Spearman", color=INK)
    return fig


def interpret_correlation(corr: pd.DataFrame, names: Dict[str, str], target: str | None) -> str:
    cols = list(corr.columns)
    pairs = [
        (cols[i], cols[j], corr.iloc[i, j])
        for i in range(len(cols))
        for j in range(i + 1, len(cols))
    ]
    strong = sorted([p for p in pairs if abs(p[2]) >= 0.7], key=lambda p: -abs(p[2]))
    text = f"De {len(pairs)} pares, {len(strong)} tienen |ρ| ≥ 0,7 (asociación fuerte)."
    if strong:
        a, b, r = strong[0]
        text += (
            f" El más fuerte es «{names.get(a, a)}» con «{names.get(b, b)}» (ρ = {r:+.2f}). "
            "Esta multicolinealidad significa que los determinantes sociales se acumulan en los "
            "mismos tracts; los modelos de árboles la toleran, pero la importancia de cada "
            "variable debe leerse como compartida con sus correlacionadas."
        )
    if target and target in corr.columns:
        t = corr[target].drop(target).sort_values(key=np.abs, ascending=False)
        top = t.head(3)
        text += " Con el objetivo, las asociaciones más fuertes son: " + ", ".join(
            f"«{names.get(k, k)}» (ρ = {v:+.2f})" for k, v in top.items()
        ) + "."
    return text


HOW_CORRELATION = (
    "ρ de Spearman: correlación entre los rangos de dos variables, robusta a atípicos y a "
    "relaciones monótonas no lineales (adecuada porque las variables no son normales). Rojo = "
    "asociación positiva (suben juntas), azul = negativa; |ρ| ≥ 0,7 fuerte, 0,4–0,7 moderada, "
    "< 0,4 débil. Correlación no implica causalidad."
)


# --------------------------------------------------------------------------- #
# Dimensión espacial y autocorrelación (Moran's I)
# --------------------------------------------------------------------------- #


def moran_i(df: pd.DataFrame, col: str, k: int = 5) -> Dict[str, float]:
    """Calcula el índice I de Moran de autocorrelación espacial global (k-vecinos más cercanos)."""
    if "lon" not in df.columns or "lat" not in df.columns:
        return {"I": 0.0, "E_I": 0.0, "z": 0.0, "p": 1.0, "n": 0}
    sub = df[["lon", "lat", col]].dropna()
    n = len(sub)
    if n < 15:
        return {"I": 0.0, "E_I": 0.0, "z": 0.0, "p": 1.0, "n": n}

    coords = sub[["lon", "lat"]].values
    vals = sub[col].values.astype(float)
    z = vals - vals.mean()
    s2 = np.sum(z**2)
    if s2 == 0:
        return {"I": 0.0, "E_I": 0.0, "z": 0.0, "p": 1.0, "n": n}

    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    _, idxs = tree.query(coords, k=min(k + 1, n))

    numerator = 0.0
    for i in range(n):
        neighbors = idxs[i, 1:]
        numerator += z[i] * np.sum(z[neighbors]) / len(neighbors)

    I = float(numerator / s2)
    E_I = -1.0 / (n - 1)
    var_I = 1.0 / (k * n)
    se = np.sqrt(var_I) if var_I > 0 else 1.0
    z_score = float((I - E_I) / se)
    p_val = float(2 * (1 - stats.norm.cdf(abs(z_score))))

    return {
        "I": round(I, 4),
        "E_I": round(E_I, 4),
        "z": round(z_score, 3),
        "p": float(f"{p_val:.3g}"),
        "n": n,
    }


def interpret_moran(m: Dict[str, float], name: str) -> str:
    i_val, z_val, p_val = m["I"], m["z"], m["p"]
    if p_val < 0.05 and i_val > 0:
        clus = "fuerte autocorrelación espacial positiva (agrupamiento o clustering en el territorio)"
        concl = ("los vecindarios con alta prevalencia tienden a estar contiguos a otros vecindarios con alta "
                 "prevalencia. Esto justifica cuantitativamente el diseño de un Gemelo Digital Geoespacial, "
                 "pues los determinantes sociales no se distribuyen al azar en el espacio.")
    elif p_val < 0.05 and i_val < 0:
        clus = "autocorrelación espacial negativa (dispersión en damero)"
        concl = "los valores altos y bajos se alternan entre tracts vecinos."
    else:
        clus = "ausencia de autocorrelación espacial (distribución espacial aleatoria)"
        concl = "no se aprecia dependencia geográfica significativa al nivel del área analizada."
    return f"«{name}»: I de Moran = {i_val:.3f} (E[I] = {m['E_I']:.4f}, z = {z_val:.2f}, p = {p_val:.3g}). Evidencia {clus}: {concl}"


def fig_spatial_distribution(df: pd.DataFrame, col: str, name: str):
    """Mapa dual de Cook County (Chicago) y New York County (Manhattan)."""
    sub = df.dropna(subset=[col, "lon", "lat"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    vmin, vmax = float(sub[col].quantile(0.02)), float(sub[col].quantile(0.98))
    cmap = "YlOrRd"

    cook = sub[sub["county"] == "Cook County"]
    if not cook.empty:
        axes[0].scatter(cook["lon"], cook["lat"], c=cook[col], cmap=cmap, s=9,
                        vmin=vmin, vmax=vmax, alpha=0.85)
    axes[0].set_title(f"Cook County, IL · {short(name, 28)}", color=INK)
    axes[0].set_xlabel("Longitud")
    axes[0].set_ylabel("Latitud")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].tick_params(labelsize=7)

    ny = sub[sub["county"] == "New York County"]
    if not ny.empty:
        axes[1].scatter(ny["lon"], ny["lat"], c=ny[col], cmap=cmap, s=18,
                        vmin=vmin, vmax=vmax, alpha=0.85)
    axes[1].set_title(f"New York County, NY · {short(name, 28)}", color=INK)
    axes[1].set_xlabel("Longitud")
    axes[1].set_ylabel("Latitud")
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].tick_params(labelsize=7)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, cax=cbar_ax, label=f"{short(name, 25)} (%)")
    return fig


def fig_bivariate_scatter(df: pd.DataFrame, x_col: str, y_col: str, names: Dict[str, str]):
    """Diagrama de dispersión bivariado X vs Y con recta de ajuste y puntos por condado."""
    sub = df.dropna(subset=[x_col, y_col])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for county, color, m in (("Cook County", "#2563eb", "o"), ("New York County", "#ea580c", "s")):
        c_sub = sub[sub["county"] == county]
        if not c_sub.empty:
            ax.scatter(c_sub[x_col], c_sub[y_col], s=12, color=color, alpha=0.55, marker=m, label=county)

    xs = sub[x_col].values.astype(float)
    ys = sub[y_col].values.astype(float)
    slope, inter, r_val, p_val, _ = stats.linregress(xs, ys)
    grid_x = np.linspace(xs.min(), xs.max(), 100)
    ax.plot(grid_x, slope * grid_x + inter, color=INK, linewidth=1.8,
            label=f"Ajuste lineal (R² = {r_val**2:.2f}, p < {p_val:.2g})")

    ax.set_xlabel(names.get(x_col, x_col) + " (%)")
    ax.set_ylabel(names.get(y_col, y_col) + " (%)")
    ax.set_title(f"Asociación: {short(names.get(x_col, x_col), 24)} vs. {short(names.get(y_col, y_col), 24)}", color=INK)
    ax.legend(fontsize=8, frameon=True, loc="best")
    return fig


HOW_SPATIAL = (
    "Mapa térmico / cloroplético sobre los centros de cada census tract en Cook County (Chicago) "
    "y New York County (Manhattan). Colores más cálidos (rojo/ámbar) representan mayores concentraciones. "
    "El I de Moran cuantifica si los valores altos tienden a agruparse en clústeres espaciales contiguos."
)

HOW_BIVARIATE = (
    "Diagrama de dispersión bivariado entre un determinante social (eje X) y el resultado sanitario (eje Y), "
    "diferenciando tracts de Cook County (azul) y New York County (naranja). La recta muestra la tendencia "
    "promedio global y su coeficiente de determinación R²."
)

