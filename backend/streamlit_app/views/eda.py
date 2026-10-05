import pandas as pd
import streamlit as st

import ui
from engine import eda

st.title("🔍 Fase II · Análisis exploratorio de datos (EDA)")
ds = ui.need("ds", "Primero cargue el dataset en «II. Carga del dataset».")
all_cols = ds.features + ds.targets
cols = st.multiselect("Variables a analizar", all_cols, default=all_cols, format_func=ds.label)
if not cols:
    st.stop()
df = ds.df

t1, t2, t3, t4, t5, t6 = st.tabs(
    ["Tendencia central y forma", "Distribución por variable", "Atípicos", "Normalidad", "Correlación", "Distribución espacial y calor"]
)

with t1:
    d = eda.describe(df, cols, ds.names)
    ui.render("eda_describe", ui.S_DATA, "Estadística descriptiva: media, mediana, moda, dispersión y forma",
              eda.HOW_DESCRIBE, eda.interpret_describe(d), table=d.drop(columns=["Código"]), order=10)
    per_var = pd.DataFrame({"Variable": d["Variable"], "Lectura": [eda.interpret_variable(r) for _, r in d.iterrows()]})
    ui.render("eda_describe_vars", ui.S_DATA, "Lectura de media, mediana, moda, asimetría y curtosis por variable",
              "Traducción automática de la tabla anterior para cada variable: compara media con mediana (sesgo), "
              "sitúa la moda, clasifica la asimetría (|g| < 0,5 simétrica; 0,5–1 moderada; > 1 fuerte) y la "
              "curtosis de exceso (> 1 leptocúrtica, < −1 platicúrtica) y valora la dispersión por el CV.",
              f"Las {len(per_var)} lecturas permiten decidir qué resumen usar en cada variable: la mediana "
              "cuando la distribución está sesgada y la media cuando es simétrica.",
              table=per_var, order=11)
    ui.render("eda_boxplots", ui.S_DATA, "Diagramas de caja comparados", eda.HOW_BOXPLOTS,
              eda.interpret_boxplots(df, cols, ds.names),
              fig=eda.fig_boxplots(df, cols, ds.names), order=12)

with t2:
    var = st.selectbox("Variable", cols, format_func=ds.label)
    row = eda.describe(df, [var], ds.names).iloc[0]
    ui.render(f"eda_hist_{var}", ui.S_DATA, f"Distribución de «{ds.label(var)}» con media, mediana y moda",
              eda.HOW_HISTOGRAM, eda.interpret_variable(row), fig=eda.fig_histogram(df[var], ds.label(var)), order=13)
    st.caption("Cada variable que consulte aquí se añade al reporte.")

with t3:
    o = eda.outliers(df, cols, ds.names)
    ui.render("eda_outliers", ui.S_DATA, "Detección de valores atípicos (Tukey, z-robusto, z clásico)",
              eda.HOW_OUTLIERS, eda.interpret_outliers(o), fig=eda.fig_outliers(o),
              table=o.drop(columns=["Código"]), order=14)
    var_o = st.selectbox("Ver los tracts atípicos de", cols, format_func=ds.label, key="var_o")
    lim = o.set_index("Código").loc[var_o]
    ext = df[(df[var_o] > lim["Límite sup. (Tukey)"]) | (df[var_o] < lim["Límite inf. (Tukey)"])]
    ext = ext[["geoid", "name", "county", var_o]].sort_values(var_o, ascending=False)
    ui.render(f"eda_outlier_list_{var_o}", ui.S_DATA, f"Tracts atípicos en «{ds.label(var_o)}»",
              "Listado de los census tracts que quedan fuera de las vallas de Tukey para la variable elegida, "
              "ordenados de mayor a menor valor.",
              (f"{len(ext)} tracts atípicos; el más extremo es {ext.iloc[0]['name']} ({ext.iloc[0]['county']}) con "
               f"{ext.iloc[0][var_o]:.1f} %, frente a una valla superior de {lim['Límite sup. (Tukey)']:.1f} %. "
               "Son candidatos naturales a intervención prioritaria si la variable es de riesgo."
               if len(ext) else "No hay tracts atípicos en esta variable."),
              table=ext, order=15)

with t4:
    n = eda.normality(df, cols, ds.names)
    ui.render("eda_normality", ui.S_DATA, "Pruebas de normalidad (Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling)",
              eda.HOW_NORMALITY, eda.interpret_normality(n, len(df)), table=n.drop(columns=["Código"]), order=16)
    var_q = st.selectbox("Gráfico Q-Q de", cols, format_func=ds.label, key="var_q")
    ui.render(f"eda_qq_{var_q}", ui.S_DATA, f"Gráfico Q-Q normal de «{ds.label(var_q)}»", eda.HOW_QQ,
              eda.interpret_qq(df[var_q], ds.label(var_q)), fig=eda.fig_qq(df[var_q], ds.label(var_q)), order=17)

with t5:
    tgt = st.selectbox("Resaltar asociación con el objetivo", [None] + ds.targets,
                       format_func=lambda c: "—" if c is None else ds.label(c))
    corr = eda.correlation(df, cols)
    ui.render("eda_corr", ui.S_DATA, "Matriz de correlación de Spearman", eda.HOW_CORRELATION,
              eda.interpret_correlation(corr, ds.names, tgt), fig=eda.fig_correlation(corr, ds.names), order=18)

with t6:
    st.subheader("🗺️ Distribución espacial y autocorrelación de Moran")
    var_map = st.selectbox("Variable para el mapa térmico", cols, format_func=ds.label, key="var_map")
    if "lon" in df.columns and "lat" in df.columns and df["lon"].notna().any():
        m = eda.moran_i(df, var_map)
        ui.render(f"eda_spatial_{var_map}", ui.S_DATA, f"Distribución espacial de «{ds.label(var_map)}»",
                  eda.HOW_SPATIAL, eda.interpret_moran(m, ds.label(var_map)),
                  fig=eda.fig_spatial_distribution(df, var_map, ds.label(var_map)),
                  table=pd.DataFrame([{
                      "Variable": ds.label(var_map),
                      "I de Moran": m["I"],
                      "E[I]": m["E_I"],
                      "Z-score": m["z"],
                      "p-valor": m["p"],
                      "Tracts analizados": m["n"],
                  }]), order=19)
    else:
        st.info("Coordenadas espaciales no disponibles en el dataset actual.")

    st.divider()
    st.subheader("📈 Asociación bivariada: Determinante vs. Objetivo")
    c_bx, c_by = st.columns(2)
    with c_bx:
        bx = st.selectbox("Determinante social (Eje X)", ds.features, format_func=ds.label, key="bx")
    with c_by:
        by = st.selectbox("Resultado de salud (Eje Y)", ds.targets, format_func=ds.label, key="by")
    ui.render(f"eda_biv_{bx}_{by}", ui.S_DATA, f"Asociación bivariada: «{ds.label(bx)}» vs. «{ds.label(by)}»",
              eda.HOW_BIVARIATE,
              f"Dispersión y ajuste lineal entre «{ds.label(bx)}» y «{ds.label(by)}» para los census tracts estudiados.",
              fig=eda.fig_bivariate_scatter(df, bx, by, ds.names), order=20)

