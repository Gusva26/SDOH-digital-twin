import numpy as np
import pandas as pd
import streamlit as st

import ui
from engine.modeling import CLASSES
from engine.style import BLUE, INK, LEVEL_COLORS, LEVEL_ES, MUTED, RED, plt, short

st.title("🚀 Fase VI · Despliegue: predicción de escenarios")
r = ui.need("train", "Primero entrene los modelos en «IV. Entrenamiento».")
p = st.session_state.prep
ds = st.session_state.ds
best = r["models"][r["best"]]
est = best["estimator"]
X = p["X_train"]
med = X.median()

st.markdown(
    f"Simule un census tract ajustando sus determinantes sociales. Modelo: **{best['name']}** "
    f"(F1 CV {best['cv_mean']:.3f}). Los valores por defecto son la mediana de entrenamiento."
)
tract = st.selectbox("Partir de un tract real (opcional)", [None] + p["data"]["geoid"].tolist(),
                     format_func=lambda g: "Tract mediano" if g is None else
                     f"{g} · {p['data'].set_index('geoid').loc[g, 'name']}")
base = med if tract is None else p["data"].set_index("geoid").loc[tract, p["features"]].fillna(med)

vals = {}
cols = st.columns(3)
for i, f in enumerate(p["features"]):
    lo, hi = float(X[f].min()), float(X[f].max())
    vals[f] = cols[i % 3].slider(ds.label(f), lo, hi, float(np.clip(base[f], lo, hi)), 0.1, key=f"sc_{f}_{tract}")
row = pd.DataFrame([vals], columns=p["features"])
proba = est.predict_proba(row)[0]
classes = list(est.classes_)
probs = {c: float(proba[classes.index(c)]) for c in CLASSES}
level = max(probs, key=probs.get)

fig, ax = plt.subplots(figsize=(6.4, 2.4))
ax.barh([LEVEL_ES[c] for c in CLASSES], [probs[c] for c in CLASSES], color=[LEVEL_COLORS[c] for c in CLASSES])
for i, c in enumerate(CLASSES):
    ax.text(probs[c] + 0.01, i, f"{probs[c]:.0%}", va="center", fontsize=8)
ax.set_xlim(0, 1.1)
ax.invert_yaxis()
ax.set_xlabel("Probabilidad")
ax.set_title(f"Nivel predicho: {LEVEL_ES[level]}", color=INK)

# Explicabilidad local: cambio en la probabilidad del nivel predicho si cada variable vuelve a su mediana.
effects = []
for f in p["features"]:
    alt = row.copy()
    alt[f] = med[f]
    pa = est.predict_proba(alt)[0][classes.index(level)]
    effects.append({"Código": f, "Variable": ds.label(f), "Valor escenario": vals[f], "Mediana": float(med[f]),
                    "Efecto en P(nivel) (pp)": 100 * (probs[level] - pa)})
eff = pd.DataFrame(effects).sort_values("Efecto en P(nivel) (pp)", key=np.abs, ascending=False).round(3)
cuts = p["cuts"]
band = {"low": f"< {cuts[0]:.1f} %", "moderate": f"{cuts[0]:.1f}–{cuts[1]:.1f} %",
        "high": f"{cuts[1]:.1f}–{cuts[2]:.1f} %", "critical": f"≥ {cuts[2]:.1f} %"}[level]
second = sorted(probs, key=probs.get)[-2]
up = eff[eff["Efecto en P(nivel) (pp)"] > 0.5].head(3)
text = (
    f"El escenario se clasifica como {LEVEL_ES[level].upper()} ({band} de {ui.label(p['target'])}) con "
    f"probabilidad {probs[level]:.0%}; la segunda opción es {LEVEL_ES[second]} ({probs[second]:.0%}). "
    + ("La decisión es clara." if probs[level] - probs[second] > 0.3 else
       "Las dos opciones están próximas: el tract está cerca de una frontera entre niveles. ")
)
if len(up):
    text += " Lo que más empuja hacia este nivel: " + ", ".join(
        f"{v} ({s:.1f} % vs. mediana {m:.1f} %, +{e:.1f} pp)"
        for v, s, m, e in zip(up["Variable"], up["Valor escenario"], up["Mediana"], up["Efecto en P(nivel) (pp)"])
    ) + "."
ui.render("deploy_scenario", ui.S_DEPLOY, "Predicción del escenario simulado",
          "Probabilidad que el modelo seleccionado asigna a cada nivel de riesgo para el tract simulado. El nivel "
          "predicho es el de mayor probabilidad; la diferencia con el segundo indica la seguridad de la predicción.",
          text, fig=fig, order=1)

top = eff.head(10).iloc[::-1]
fig2, ax2 = plt.subplots(figsize=(7, max(2.4, 0.3 * len(top) + 0.8)))
ax2.barh([short(v, 34) for v in top["Variable"]], top["Efecto en P(nivel) (pp)"],
         color=[RED if v > 0 else BLUE for v in top["Efecto en P(nivel) (pp)"]])
ax2.axvline(0, color=MUTED, linewidth=0.8)
ax2.set_xlabel(f"Cambio en P({LEVEL_ES[level]}) atribuible a la variable (pp)")
ax2.set_title("Explicabilidad local del escenario", color=INK)
ui.render("deploy_local", ui.S_DEPLOY, "Explicabilidad local: ¿qué determina esta predicción?",
          "Para cada determinante se reemplaza su valor por la mediana del área (manteniendo el resto) y se "
          "mide cuánto cambia la probabilidad del nivel predicho. Rojo = la variable, en su valor actual, "
          "aumenta esa probabilidad; azul = la reduce. Es una explicación contrafactual local (tipo ceteris paribus).",
          ("Los factores con mayor efecto local son: " + ", ".join(
              f"«{v}» ({e:+.1f} pp)" for v, e in zip(eff["Variable"].head(3), eff["Efecto en P(nivel) (pp)"].head(3)))
           + ". Intervenir sobre los determinantes en rojo (acercarlos a la mediana) es lo que más reduciría la "
           "probabilidad de este nivel según el modelo; la relación es predictiva, no necesariamente causal."),
          fig=fig2, table=eff.drop(columns=["Código"]), order=2)
