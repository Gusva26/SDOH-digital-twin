import streamlit as st

import ui
from engine import inference as inf

st.title("📐 Fase V · Pruebas estadísticas inferenciales robustas")
r = ui.need("train", "Primero entrene los modelos en «IV. Entrenamiento».")
p = st.session_state.prep
ds = st.session_state.ds
cache = st.session_state.inf
tname = ds.label(p["target"])
st.caption(
    "Todas las pruebas son no paramétricas o de remuestreo (el EDA rechazó la normalidad) y los "
    "contrastes múltiples se corrigen con Holm. α = 0,05."
)

t1, t2, t3 = st.tabs(["Comparación de modelos", "Robustez del modelo elegido", "Determinantes vs. resultado"])

with t1:
    if "friedman" not in cache:
        cache["friedman"] = inf.friedman(r)
        n_fold_test = len(p["X_train"]) // r["n_splits"]
        cache["pairwise"] = inf.pairwise(r, len(p["X_train"]) - n_fold_test, n_fold_test)
    f = cache["friedman"]
    ui.render("inf_friedman", ui.S_EVAL, "Prueba de Friedman y post-hoc de Nemenyi", inf.HOW_FRIEDMAN,
              inf.interpret_friedman(f), table=inf.friedman_table(f), order=1)
    ui.render("inf_cd", ui.S_EVAL, "Diagrama de diferencia crítica", inf.HOW_CD, inf.interpret_cd(f),
              fig=inf.fig_cd(f), order=2)
    ui.render("inf_pairwise", ui.S_EVAL, "Comparaciones por pares: Wilcoxon y t corregida de Nadeau-Bengio (Holm)",
              inf.HOW_PAIRWISE, inf.interpret_pairwise(cache["pairwise"], r["models"][r["best"]]["name"]),
              table=cache["pairwise"], order=3)

with t2:
    if "mcnemar" not in cache:
        cache["mcnemar"] = inf.mcnemar(r, p["y_test"])
        cache["boot"] = inf.bootstrap_f1(r, p["y_test"])
    m = cache["mcnemar"]
    ui.render("inf_mcnemar", ui.S_EVAL, f"Prueba de McNemar: {m['a']} vs. {m['b']}", inf.HOW_MCNEMAR,
              inf.interpret_mcnemar(m), table=inf.mcnemar_table(m), order=4)
    bs = cache["boot"]
    ui.render("inf_bootstrap", ui.S_EVAL, "Intervalos de confianza bootstrap del F1 de prueba", inf.HOW_BOOTSTRAP,
              inf.interpret_bootstrap(bs), fig=inf.fig_bootstrap(bs, r), table=bs["table"], order=5)
    n_perm = st.select_slider("Permutaciones", [50, 100, 200, 500], value=100)
    if st.button("Ejecutar prueba de permutación del modelo", type="primary"):
        with st.spinner(f"Reentrenando {r['models'][r['best']]['name']} con {n_perm} permutaciones…"):
            cache["perm"] = inf.permutation_test(r, p, n_perm)
    if "perm" in cache:
        pt = cache["perm"]
        ui.render("inf_permutation", ui.S_EVAL, "Prueba de permutación: ¿supera el modelo al azar?",
                  inf.HOW_PERMUTATION, inf.interpret_permutation(pt), fig=inf.fig_permutation(pt), order=6)

with t3:
    if "spearman" not in cache:
        cache["spearman"] = inf.spearman_table(p, ds.names)
        cache["kruskal"] = inf.kruskal_levels(p, ds.names)
        cache["counties"] = inf.kruskal_counties(p, ds.names)
    sp = cache["spearman"]
    ui.render("inf_spearman", ui.S_EVAL, f"Correlación de Spearman con «{tname}» (IC 95 %, Holm)", inf.HOW_SPEARMAN,
              inf.interpret_spearman(sp, tname), fig=inf.fig_spearman(sp), table=sp.drop(columns=["Código"]), order=7)
    kw = cache["kruskal"]
    ui.render("inf_kruskal", ui.S_EVAL, "Kruskal-Wallis: determinantes por nivel de riesgo (ε²)", inf.HOW_KRUSKAL,
              inf.interpret_kruskal(kw), table=kw.drop(columns=["Código"]), order=8)
    kc = cache["counties"]
    if kc.get("applicable"):
        ui.render("inf_counties", ui.S_EVAL, f"Kruskal-Wallis: «{tname}» entre condados", inf.HOW_COUNTIES,
                  inf.interpret_counties(kc, tname), table=kc["table"], order=9)
