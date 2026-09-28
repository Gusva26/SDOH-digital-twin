import pandas as pd
import streamlit as st

import ui
from engine import data
from engine.style import BLUE, INK, plt, short

st.title("📥 Fase II · Carga del dataset (datos reales)")

src = st.radio("Fuente", ["Base de datos PostGIS (CDC PLACES cargado por el ETL)", "Subir CSV"], horizontal=True)
if src.startswith("Base"):
    try:
        years = data.available_years()
    except Exception as exc:
        st.error(f"No se pudo conectar a la base de datos: {exc}")
        st.stop()
    year = st.selectbox("Año preferente", ["Más reciente"] + years)
    if st.button("Cargar dataset", type="primary"):
        with st.spinner("Consultando PostGIS…"):
            st.session_state.ds = data.load_from_db(None if year == "Más reciente" else int(year))
        ui.reset_from(ui.S_DATA)
else:
    up = st.file_uploader(
        "CSV de CDC PLACES (LocationID, Measure, Data_Value, Year) o tabla ancha con 'geoid' + códigos",
        type="csv",
    )
    if up is not None and st.button("Cargar CSV", type="primary"):
        try:
            st.session_state.ds = data.load_csv(up.getvalue(), up.name)
            ui.reset_from(ui.S_DATA)
        except Exception as exc:
            st.error(str(exc))

ds = ui.need("ds", "Cargue un dataset para continuar.")
st.success(f"**{ds.source}** · {len(ds.df)} census tracts · {len(ds.features)} determinantes · {len(ds.targets)} resultados de salud")
for n in ds.notes:
    st.caption("• " + n)

preview = ds.df.head(15)
ui.render(
    "data_preview", ui.S_DATA, "Vista previa del dataset",
    "Cada fila es un census tract (unidad de análisis, identificado por su GEOID de 11 dígitos) y cada "
    "columna un indicador de CDC PLACES expresado como prevalencia en adultos (%). Se muestran las "
    "primeras 15 filas.",
    f"El dataset tiene {len(ds.df)} filas y {ds.df.shape[1]} columnas, con "
    f"{ds.df['county'].nunique()} condados. Los valores son estimaciones de área pequeña publicadas por "
    "el CDC (modelo con BRFSS + Census ACS), no conteos de pacientes: son datos agregados, públicos y sin "
    "información identificable.",
    table=preview,
)

q = data.quality_table(ds)
fig, ax = plt.subplots(figsize=(7.2, max(3, 0.3 * len(q) + 0.8)))
qs = q.sort_values("Completitud %")
ax.barh([short(v, 36) for v in qs["Variable"]], qs["Completitud %"],
        color=[BLUE if r == "Entrada (X)" else "#f59e0b" for r in qs["Rol"]])
ax.set_xlim(0, 105)
ax.set_xlabel("Completitud (%)  · azul = determinante (X), ámbar = resultado (y)")
ax.set_title("Completitud por variable", color=INK)
ax.tick_params(axis="y", labelsize=7.5)
ui.render(
    "data_quality", ui.S_DATA, "Calidad de datos: completitud por variable",
    "Porcentaje de tracts con valor observado en cada variable. Una completitud baja reduce la muestra "
    "útil o exige imputar; las variables sin ningún dato no pueden usarse.",
    data.interpret_quality(q, len(ds.df)),
    fig=fig, table=q,
)

dup = int(ds.df["geoid"].duplicated().sum())
nums = ds.df[ds.features + ds.targets]
out_range = int(((nums < 0) | (nums > 100)).sum().sum())
missing = int(nums.isna().sum().sum())
checks = pd.DataFrame(
    [
        ["Unicidad del tract (GEOID duplicados)", dup, "OK" if dup == 0 else "Revisar"],
        ["Plausibilidad (valores fuera de [0, 100] %)", out_range, "OK" if out_range == 0 else "Revisar"],
        ["Celdas faltantes en variables del modelo", missing, "Se imputan" if missing else "OK"],
    ],
    columns=["Verificación", "Casos", "Estado"],
)
ui.render(
    "data_checks", ui.S_DATA, "Verificaciones de integridad",
    "Reglas de calidad de CRISP-DM aplicadas al dataset: cada tract debe aparecer una sola vez, las "
    "prevalencias deben estar entre 0 y 100 % y se cuentan los huecos que tendrá que imputar la fase III.",
    ("Sin duplicados de tract" if dup == 0 else f"{dup} tracts duplicados (se conservará el primero)")
    + " y "
    + ("todas las prevalencias son porcentajes válidos. " if out_range == 0
       else f"{out_range} valores fuera de rango que deben revisarse. ")
    + (f"Hay {missing} celdas faltantes ({100 * missing / nums.size:.2f} % del total), un volumen bajo "
       "que la imputación por mediana absorbe sin distorsionar las distribuciones." if missing
       else "No hay valores faltantes."),
    table=checks,
)
