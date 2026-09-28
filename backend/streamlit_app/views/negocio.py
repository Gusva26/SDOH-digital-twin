import pandas as pd
import streamlit as st

import ui
from engine.data import TARGET_INFO

st.title("🎯 Fase I · Comprensión del negocio")
st.markdown(
    "**Problema.** Los gestores hospitalarios necesitan identificar, dentro de su área de captación, "
    "los census tracts con mayor carga de enfermedad asociada a los determinantes sociales de la "
    "salud (SDOH) para priorizar intervenciones con evidencia trazable.\n\n"
    "**Objetivo analítico.** Predecir el nivel de riesgo (Bajo / Moderado / Alto / Crítico) de un "
    "resultado de salud por tract a partir de 13 determinantes sociales, de acceso y de conducta "
    "(datos reales CDC PLACES), y explicar qué determinantes pesan más."
)

phases = pd.DataFrame(
    [
        ["I. Negocio", "Objetivos, criterios de éxito", "Esta página"],
        ["II. Datos", "Carga del dataset real + EDA (media, mediana, moda, curtosis, atípicos, normalidad)", "Carga · EDA"],
        ["III. Preparación", "Split estratificado, niveles por cuartiles, imputación y escalado", "Preparación"],
        ["IV. Modelado", "Entrenamiento de 4 algoritmos + línea base, GridSearch, CV repetida, selección", "Entrenamiento · Hiperparámetros · Validación · Selección"],
        ["V. Evaluación", "Friedman/Nemenyi, Wilcoxon-Holm, Nadeau-Bengio, McNemar, bootstrap, permutación, Spearman, Kruskal-Wallis", "Pruebas inferenciales"],
        ["VI. Despliegue", "Predicción de escenarios, reporte PDF, asistentes LangChain y LangFlow", "Despliegue · Reporte · IA"],
    ],
    columns=["Fase", "Qué se hace", "Módulo de la app"],
)
ui.render(
    "business_phases", ui.S_BUSINESS, "Mapa de la metodología CRISP-DM en la aplicación",
    "Cada fila es una de las seis fases iterativas de CRISP-DM (Chapman et al., 2000) y el módulo de "
    "esta app que la implementa. Las fases se recorren en orden; cada resultado alimenta la siguiente "
    "y todos se acumulan en el reporte PDF.",
    "El flujo cubre el ciclo completo: desde datos públicos reales hasta un modelo validado "
    "estadísticamente y desplegado con explicaciones. Volver a una fase anterior (p. ej. cambiar el "
    "objetivo) invalida automáticamente los resultados de las fases posteriores para mantener la "
    "trazabilidad.",
    table=phases,
)

criteria = pd.DataFrame(
    [
        ["Negocio", "Priorizar tracts críticos de forma explicable", "Cada predicción incluye su interpretación y factores"],
        ["Minería de datos", "F1-macro en CV claramente superior al azar (≈ 0,25)", "F1 CV ≥ 0,60 y prueba de permutación p < 0,05"],
        ["Estadístico", "Selección del modelo respaldada por pruebas", "Friedman + post-hoc con corrección de comparaciones múltiples"],
        ["Ético", "Sin datos individuales; conclusiones ecológicas", "Solo agregados por census tract"],
    ],
    columns=["Tipo", "Criterio", "Umbral / evidencia"],
)
ui.render(
    "business_criteria", ui.S_BUSINESS, "Criterios de éxito",
    "Criterios definidos antes de modelar, para evaluar el resultado sin sesgo de confirmación. "
    "Los de minería de datos y estadísticos se verifican en las fases IV y V.",
    "Con 4 clases equilibradas, un clasificador al azar obtiene F1 ≈ 0,25; el umbral de 0,60 exige "
    "más del doble. Además se exige significación frente a una distribución nula por permutación, "
    "de modo que el éxito no dependa de una partición afortunada.",
    table=criteria,
)

targets = pd.DataFrame([{"Código": k, "Resultado de salud (y)": v} for k, v in TARGET_INFO.items()])
ui.render(
    "business_targets", ui.S_BUSINESS, "Resultados de salud que se pueden modelar",
    "Variables objetivo disponibles en CDC PLACES. Son prevalencias en adultos (%) por census tract; "
    "se elige una en la fase de Preparación.",
    "Los determinantes sociales (entradas) nunca incluyen resultados de salud, para evitar fuga de "
    "información: el modelo aprende la relación contexto social → salud, no la fórmula de otro índice.",
    table=targets,
)
