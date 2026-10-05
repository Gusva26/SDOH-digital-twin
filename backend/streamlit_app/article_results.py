"""Genera todos los resultados, tablas y figuras del artículo SP-5 a partir de los
datos reales de la base (CDC PLACES) con el mismo motor de la app Streamlit.

    docker compose exec streamlit python streamlit_app/article_results.py

Salida: backend/article_outputs/results.json y figuras PNG a 300 dpi.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sqlalchemy import func  # noqa: E402

from app.core.database import SessionLocal  # noqa: E402
from app.models.geo import CensusTract  # noqa: E402
from app.services import crispdm_service, ml_evaluation, ml_service  # noqa: E402
from engine import data, eda, inference as inf, modeling  # noqa: E402
from engine.style import BLUE, GREEN, INK, LEVEL_COLORS, LEVEL_ES, MUTED, RED, plt, short  # noqa: E402

import warnings
warnings.filterwarnings("ignore")

try:
    from multiprocessing import resource_tracker
    if hasattr(resource_tracker, "ResourceTracker") and hasattr(resource_tracker.ResourceTracker, "_stop"):
        _orig_stop = resource_tracker.ResourceTracker._stop
        def _quiet_stop(self):
            try:
                _orig_stop(self)
            except Exception:
                pass
        resource_tracker.ResourceTracker._stop = _quiet_stop
except Exception:
    pass

OUT = os.path.join(os.path.dirname(HERE), "article_outputs")
os.makedirs(OUT, exist_ok=True)
TARGET = "pct_obesity"
DPI = 300


def save(fig, name):
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(os.path.join(OUT, name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    return str(o)


t0 = time.time()
R = {}
ds = data.load_from_db()
df = ds.df
if "lon" not in df.columns:
    db = SessionLocal()
    try:
        pts = pd.DataFrame(
            db.query(CensusTract.geoid, func.st_x(CensusTract.center), func.st_y(CensusTract.center)).all(),
            columns=["geoid", "lon", "lat"]
        )
        df = df.merge(pts, on="geoid", how="left")
    finally:
        db.close()
R["dataset"] = {"tracts": len(df), "counties": df["county"].value_counts().to_dict(),
                "features": ds.features, "names": {c: ds.label(c) for c in ds.features + ds.targets}}
print("dataset", len(df))

# ---------------------------------------------------------------- EDA
cols = ds.features + [TARGET]
desc = eda.describe(df, cols, ds.names)
out = eda.outliers(df, cols, ds.names)
norm = eda.normality(df, cols, ds.names)
corr = eda.correlation(df, cols)
moran = eda.moran_i(df, TARGET) if "lon" in df.columns else {"I": 0.0, "E_I": 0.0, "z": 0.0, "p": 1.0, "n": 0}
R["eda"] = {"describe": desc, "outliers": out, "normality": norm,
            "moran": moran,
            "interp_describe": eda.interpret_describe(desc), "interp_outliers": eda.interpret_outliers(out),
            "interp_normality": eda.interpret_normality(norm, len(df)),
            "interp_corr": eda.interpret_correlation(corr, ds.names, TARGET),
            "interp_moran": eda.interpret_moran(moran, ds.label(TARGET)),
            "missing_cells": int(df[cols].isna().sum().sum())}

# ---------------------------------------------------------------- Modelado
p = modeling.prepare(df, TARGET, ds.features)
r = modeling.train_all(p, 5, 2, progress=lambda f, s: print(f"  {f:.2f} {s}"))
R["prep"] = {"cuts": p["cuts"], "n_train": len(p["X_train"]), "n_test": len(p["X_test"]),
             "table": modeling.prep_table(p), "n_used": len(p["data"])}
R["training"] = {"table": modeling.training_table(r), "selection": modeling.selection_table(r),
                 "cv": modeling.cv_table(r), "best": r["models"][r["best"]]["name"],
                 "best_params": {k: r["models"][k]["best_params"] for k in r["models"]},
                 "grids": {r["models"][k]["name"]: r["models"][k]["grid"] for k in r["ranking"]},
                 "class_report": modeling.class_report_table(r), "confusion": r["models"][r["best"]]["confusion"],
                 "importance": r["importance"].assign(Variable=lambda d: d["Código"].map(ds.label)),
                 "interp": {"training": modeling.interpret_training(r), "cv": modeling.interpret_cv(r),
                            "learning": modeling.interpret_learning_curve(r),
                            "confusion": modeling.interpret_confusion(r), "roc": modeling.interpret_roc(r, p["y_test"]),
                            "importance": modeling.interpret_importance(r, ds.names),
                            "selection": modeling.interpret_selection(r),
                            "hp": {r["models"][k]["name"]: modeling.interpret_hyperparams(r["models"][k]) for k in r["ranking"]}},
                 "learning_curve": r["learning_curve"], "seconds": r["seconds"]}

# ---------------------------------------------------------------- Inferencia
f = inf.friedman(r)
nt = len(p["X_train"]) // r["n_splits"]
pw = inf.pairwise(r, len(p["X_train"]) - nt, nt)
mc = inf.mcnemar(r, p["y_test"])
bs = inf.bootstrap_f1(r, p["y_test"], 2000)
pt = inf.permutation_test(r, p, 200)
sp = inf.spearman_table(p, ds.names)
kw = inf.kruskal_levels(p, ds.names)
kc = inf.kruskal_counties(p, ds.names)
R["inference"] = {
    "friedman": {k: v for k, v in f.items()}, "friedman_table": inf.friedman_table(f), "pairwise": pw,
    "mcnemar": mc, "bootstrap": bs["table"], "bootstrap_diff": bs["diff"],
    "permutation": {"score": pt["score"], "p": pt["p"], "n": pt["n_perm"], "perm_mean": float(np.mean(pt["perm"])),
                    "perm_max": float(np.max(pt["perm"]))},
    "spearman": sp, "kruskal": kw, "counties": {k: v for k, v in kc.items()},
    "interp": {"friedman": inf.interpret_friedman(f), "cd": inf.interpret_cd(f),
               "pairwise": inf.interpret_pairwise(pw, r["models"][r["best"]]["name"]),
               "mcnemar": inf.interpret_mcnemar(mc), "bootstrap": inf.interpret_bootstrap(bs),
               "permutation": inf.interpret_permutation(pt), "spearman": inf.interpret_spearman(sp, ds.label(TARGET)),
               "kruskal": inf.interpret_kruskal(kw), "counties": inf.interpret_counties(kc, ds.label(TARGET))},
}
print("inferencia ok", round(time.time() - t0))

# ---------------------------------------------------------------- Generalización a los 7 resultados
gen = []
for tg in ds.targets:
    if tg == TARGET:
        m = r["models"][r["best"]]
        gen.append({"code": tg, "name": ds.label(tg), "best": m["name"], "cv": m["cv_mean"], "sd": m["cv_std"],
                    "test_f1": m["test_f1"], "auc": m["test_auc"], "kappa": m["test_kappa"]})
        continue
    pg = modeling.prepare(df, tg, ds.features)
    rg = modeling.train_all(pg, 5, 1)
    m = rg["models"][rg["best"]]
    gen.append({"code": tg, "name": ds.label(tg), "best": m["name"], "cv": m["cv_mean"], "sd": m["cv_std"],
                "test_f1": m["test_f1"], "auc": m["test_auc"], "kappa": m["test_kappa"],
                "baseline": rg["models"][modeling.BASELINE]["cv_mean"]})
    print("gen", tg, m["name"], round(m["cv_mean"], 3))
R["generalization"] = gen

# ---------------------------------------------------------------- H1–H3 y banco de latencia
db = SessionLocal()
try:
    if not ml_service.load_bundle(TARGET):
        print(f"Entrenando y guardando modelo activo para {TARGET}...")
        ml_service.train(db, TARGET)
    R["bench"] = crispdm_service.run_evaluation(db, repeats=30)
finally:
    db.close()
R["hypotheses"] = ml_evaluation.run_full(TARGET)
print("hipótesis ok", round(time.time() - t0))

# ================================================================ FIGURAS
names = ds.names
# Figura 1: arquitectura
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.axis("off")
ax.set_xlim(0, 10.6)
ax.set_ylim(0.8, 6.5)
layers = [
    (4.75, "#dbeafe", "Capa 1 · Sensado y fuentes",
     "CDC PLACES 2025 (Socrata API, 27 medidas, 1.632 tracts) · Census TIGERweb (geometrías)\n"
     "Hospitales y áreas de captación (radio 15 km) · Carga CSV · Eventos de reglas de alerta"),
    (2.85, "#dcfce7", "Capa 2 · Integración digital",
     "ETL: validación, armonización de medidas, unicidad (tract, indicador, año)\n"
     "PostgreSQL 16 + PostGIS 3.4 (SRID 4326) · FastAPI + JWT/RBAC · auditoría"),
    (0.95, "#fef3c7", "Capa 3 · Simulación y analítica",
     "Índice compuesto de equidad O(n·k) · Motor CRISP-DM (Python/Streamlit): EDA, 4 modelos,\n"
     "GridSearch + CV 5×2, pruebas inferenciales · Gemelo 3D (React/Three.js) · LangChain / LangFlow"),
]
for y, color, title, body in layers:
    ax.add_patch(FancyBboxPatch((0.3, y), 9.4, 1.5, boxstyle="round,pad=0.05,rounding_size=0.15",
                                facecolor=color, edgecolor=INK, linewidth=1))
    ax.text(0.55, y + 1.12, title, fontsize=11, fontweight="bold", color=INK, va="center")
    ax.text(0.55, y + 0.5, body, fontsize=9.2, color=INK, va="center", linespacing=1.5)
for y in (4.75, 2.85):
    ax.annotate("", xy=(5, y - 0.38), xytext=(5, y - 0.02), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
ax.annotate("", xy=(9.9, 5.5), xytext=(9.9, 1.7),
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4, connectionstyle="arc3,rad=0"))
ax.text(9.95, 3.6, "retroalimentación:\nalertas, escenarios,\nreportes", fontsize=7.5, color=RED, rotation=90,
        va="center", ha="left")
ax.text(5, 6.35, "Arquitectura SP-5 implementada", fontsize=12, fontweight="bold", ha="center", color=INK)
save(fig, "fig1_arquitectura.png")

# Figura 2: EDA (a) distribución del objetivo (b) mapa Cook (c) mapa Manhattan (d) Spearman
fig = plt.figure(figsize=(15.5, 4.6))
gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.9, 0.75, 1.25], wspace=0.32)
ax_hist = fig.add_subplot(gs[0])
ax_cook = fig.add_subplot(gs[1])
ax_ny = fig.add_subplot(gs[2])
ax_corr = fig.add_subplot(gs[3])

s = df[TARGET].dropna()
ax_hist.hist(s, bins=40, color="#bfdbfe", edgecolor="white", density=True)
from scipy import stats as _st  # noqa: E402
xs = np.linspace(s.min(), s.max(), 300)
ax_hist.plot(xs, _st.gaussian_kde(s)(xs), color=BLUE, lw=1.6, label="KDE")
row = desc.set_index("Código").loc[TARGET]
ax_hist.axvline(row["Media"], color=RED, lw=1.8, label=f"Media {row['Media']:.2f}")
ax_hist.axvline(row["Mediana"], color=GREEN, lw=1.8, ls="--", label=f"Mediana {row['Mediana']:.2f}")
ax_hist.axvline(row["Moda"], color="#f59e0b", lw=1.8, ls=":", label=f"Moda {row['Moda']:.1f}")
for c in p["cuts"]:
    ax_hist.axvline(c, color=MUTED, lw=0.8, ls="-.")
ax_hist.set_xlabel("Obesidad en adultos (%)")
ax_hist.set_ylabel("Densidad")
ax_hist.legend(fontsize=7.5, frameon=False)
ax_hist.set_title("(a) Distribución y cortes Q1–Q3", color=INK)

vmin, vmax = float(s.quantile(0.02)), float(s.quantile(0.98))
cmap = "YlOrRd"
if "lon" in df.columns:
    sub_c = df[df["county"] == "Cook County"].dropna(subset=["lon", "lat", TARGET])
    if not sub_c.empty:
        ax_cook.scatter(sub_c["lon"], sub_c["lat"], c=sub_c[TARGET], cmap=cmap, s=7, vmin=vmin, vmax=vmax, alpha=0.85)
    sub_ny = df[df["county"] == "New York County"].dropna(subset=["lon", "lat", TARGET])
    if not sub_ny.empty:
        ax_ny.scatter(sub_ny["lon"], sub_ny["lat"], c=sub_ny[TARGET], cmap=cmap, s=15, vmin=vmin, vmax=vmax, alpha=0.85)

ax_cook.set_aspect("equal", adjustable="datalim")
ax_cook.set_title("(b) Cook County, IL", color=INK)
ax_cook.set_xlabel("Longitud")
ax_cook.set_ylabel("Latitud")
ax_cook.tick_params(labelsize=6.8)

ax_ny.set_aspect("equal", adjustable="datalim")
ax_ny.set_title("(c) New York, NY", color=INK)
ax_ny.set_xlabel("Longitud")
ax_ny.tick_params(labelsize=6.8)

n = len(corr)
im = ax_corr.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
lab = [short(names.get(c, c), 18) for c in corr.columns]
ax_corr.set_xticks(range(n))
ax_corr.set_xticklabels(lab, rotation=65, ha="right", fontsize=6.8)
ax_corr.set_yticks(range(n))
ax_corr.set_yticklabels(lab, fontsize=6.8)
ax_corr.grid(False)
for i in range(n):
    for j in range(n):
        ax_corr.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=4.8,
                     color="white" if abs(corr.values[i, j]) > 0.6 else INK)
fig.colorbar(im, ax=ax_corr, fraction=0.04, label="ρ de Spearman")
ax_corr.set_title("(d) Correlación de Spearman", color=INK)
save(fig, "fig2_eda.png")

# Figura 3: (a) CV por fold (b) diferencia crítica
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
keys = r["ranking"] + [modeling.BASELINE]
bp = axes[0].boxplot([r["models"][k]["fold_scores"] for k in keys], patch_artist=True, widths=0.55,
                     medianprops=dict(color=RED))
for patch, k in zip(bp["boxes"], keys):
    patch.set_facecolor("#bbf7d0" if k == r["best"] else ("#e2e8f0" if k == modeling.BASELINE else "#dbeafe"))
for i, k in enumerate(keys, 1):
    v = r["models"][k]["fold_scores"]
    axes[0].scatter(np.random.default_rng(i).normal(i, 0.05, len(v)), v, s=9, color=INK, alpha=0.6, zorder=3)
axes[0].set_xticks(range(1, len(keys) + 1))
import textwrap  # noqa: E402
axes[0].set_xticklabels([textwrap.fill(r["models"][k]["name"], 13) for k in keys], fontsize=7.5)
axes[0].set_ylabel("F1-macro por fold")
axes[0].set_title("(a) Validación cruzada estratificada 5×2", color=INK)
order = np.argsort(f["avg_ranks"])
rk = f["avg_ranks"][order]
yy = np.arange(len(order))
axes[1].axvspan(rk[0], rk[0] + f["cd"], color=GREEN, alpha=0.15, label=f"CD de Nemenyi = {f['cd']:.2f}")
axes[1].scatter(rk, yy, color=BLUE, s=45, zorder=3)
for yi, v in zip(yy, rk):
    axes[1].text(v, yi - 0.22, f"{v:.2f}", ha="center", fontsize=8)
axes[1].set_yticks(yy)
axes[1].set_yticklabels([f["names"][i] for i in order], fontsize=8)
axes[1].invert_yaxis()
axes[1].set_xlim(0.8, f["k"] + 0.2)
axes[1].set_xlabel("Rango medio (1 = mejor)")
axes[1].legend(fontsize=8, frameon=True, loc="lower center")
axes[1].set_title(f"(b) Friedman χ²={f['stat']:.1f}, p={f['p']:.2g}, W={f['kendall_w']:.2f}", color=INK)
save(fig, "fig3_cv_friedman.png")

# Figura 4: (a) matriz de confusión (b) ROC
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from sklearn.preprocessing import label_binarize  # noqa: E402

best = r["models"][r["best"]]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
cm = best["confusion"]
pct = cm / cm.sum(axis=1, keepdims=True)
axes[0].imshow(pct, cmap="Blues", vmin=0, vmax=1)
lbl = [LEVEL_ES[c] for c in modeling.CLASSES]
axes[0].set_xticks(range(4))
axes[0].set_xticklabels(lbl)
axes[0].set_yticks(range(4))
axes[0].set_yticklabels(lbl)
axes[0].grid(False)
for i in range(4):
    for j in range(4):
        axes[0].text(j, i, f"{cm[i, j]}\n{pct[i, j]:.0%}", ha="center", va="center", fontsize=8.5,
                     color="white" if pct[i, j] > 0.5 else INK)
axes[0].set_xlabel("Nivel predicho")
axes[0].set_ylabel("Nivel observado")
axes[0].set_title(f"(a) Matriz de confusión en prueba · {best['name']}", color=INK)
yb = label_binarize(p["y_test"], classes=best["classes"])
for i, c in enumerate(best["classes"]):
    fpr, tpr, _ = roc_curve(yb[:, i], best["proba"][:, i])
    axes[1].plot(fpr, tpr, color=LEVEL_COLORS[c], lw=1.7,
                 label=f"{LEVEL_ES[c]} (AUC {roc_auc_score(yb[:, i], best['proba'][:, i]):.3f})")
axes[1].plot([0, 1], [0, 1], ls="--", color=MUTED, lw=0.8)
axes[1].set_xlabel("Tasa de falsos positivos")
axes[1].set_ylabel("Tasa de verdaderos positivos")
axes[1].legend(fontsize=8, frameon=True, loc="lower center")
axes[1].set_title("(b) Curvas ROC uno-contra-resto", color=INK)
save(fig, "fig4_confusion_roc.png")

# Figura 5: (a) importancia por permutación (b) Spearman IC95
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
imp = r["importance"].sort_values("Importancia")
axes[0].barh([short(names.get(c, c), 30) for c in imp["Código"]], imp["Importancia"], xerr=imp["Desv."],
             color=[BLUE if v > 0 else "#94a3b8" for v in imp["Importancia"]], capsize=2)
axes[0].axvline(0, color=MUTED, lw=0.8)
axes[0].set_xlabel("Caída del F1-macro al permutar")
axes[0].tick_params(axis="y", labelsize=8)
axes[0].set_title("(a) Importancia por permutación (10 repeticiones)", color=INK)
t = sp.sort_values("ρ Spearman")
yy = np.arange(len(t))
axes[1].errorbar(t["ρ Spearman"], yy, xerr=[t["ρ Spearman"] - t["IC95% inf."], t["IC95% sup."] - t["ρ Spearman"]],
                 fmt="none", ecolor=MUTED, capsize=3)
axes[1].scatter(t["ρ Spearman"], yy, color=[RED if v > 0 else BLUE for v in t["ρ Spearman"]], s=30, zorder=3)
axes[1].axvline(0, color=MUTED, lw=0.8)
axes[1].set_yticks(yy)
axes[1].set_yticklabels([short(v, 30) for v in t["Variable"]], fontsize=8)
axes[1].set_xlabel("ρ de Spearman con la obesidad (IC 95 %)")
axes[1].set_title("(b) Asociación bivariada determinante–resultado", color=INK)
save(fig, "fig5_explicabilidad.png")

# Figura 6: (a) mapa del gemelo: nivel estimado por tract (b) H2 barrido de radio
db = SessionLocal()
try:
    pts = pd.DataFrame(db.query(CensusTract.geoid, func.st_x(CensusTract.center), func.st_y(CensusTract.center)).all(),
                       columns=["geoid", "lon", "lat"])
finally:
    db.close()
full = p["data"].copy()
if "lon" not in full.columns:
    full = full.merge(pts, on="geoid", how="left")
full["pred"] = best["estimator"].predict(full[p["features"]])
fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.1]})
for ax, fips, title in ((axes[0], "Cook County", "(a) Cook County, IL"), (axes[1], "New York County", "(b) New York County, NY")):
    sub = full[full["county"] == fips]
    for c in modeling.CLASSES:
        ss = sub[sub["pred"] == c]
        ax.scatter(ss["lon"], ss["lat"], s=6 if fips == "Cook County" else 12, color=LEVEL_COLORS[c], label=LEVEL_ES[c])
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title + " · nivel predicho", color=INK)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.tick_params(labelsize=7)
axes[0].legend(fontsize=7.5, frameon=False, markerscale=2)
h2 = next(h for h in R["hypotheses"]["hypotheses"]["hypotheses"] if h["code"] == "H2")["detail"]
if h2.get("applicable"):
    sw = pd.DataFrame(h2["radius_sweep"])
    axes[2].plot(sw["radius_km"], sw["rho"], "o-", color=BLUE, label="Compuesto + decaimiento")
    axes[2].axhline(h2["rho"]["cobertura_binaria"], color="#f59e0b", ls="--", label="Compuesto + cobertura binaria")
    axes[2].axhline(h2["rho"]["sin_accesibilidad"], color=GREEN, ls=":", label="Compuesto sin accesibilidad")
    axes[2].set_xscale("log")
    axes[2].set_xticks(sw["radius_km"])
    axes[2].set_xticklabels([f"{int(v)}" for v in sw["radius_km"]])
    axes[2].set_xlabel("Radio de decaimiento exponencial (km)")
    axes[2].set_ylabel("|ρ| con obesidad observada")
    axes[2].legend(fontsize=7.5, frameon=True, loc="center right")
    axes[2].set_title("(c) H2: validez según el radio de decaimiento", color=INK)
save(fig, "fig6_gemelo_h2.png")

with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as fh:
    json.dump(R, fh, ensure_ascii=False, indent=1, default=jsonable)
print("LISTO", round(time.time() - t0), "s")
