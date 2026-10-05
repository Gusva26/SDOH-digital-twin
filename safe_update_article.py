"""Actualización limpia y segura de Articulo_SP5_completo.docx usando python-docx."""
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
DOCX_IN = os.path.join(HERE, "Articulo_SP5_completo_backup.docx")
DOCX_OUT = os.path.join(HERE, "Articulo_SP5_completo.docx")
FIG2_PATH = os.path.join(HERE, "backend", "article_outputs", "fig2_eda.png")

try:
    import docx
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
except ImportError:
    print("[ERROR] 'python-docx' no esta instalado.")
    print("Por favor ejecuta: pip install python-docx")
    exit(1)

def run():
    if not os.path.exists(DOCX_IN):
        print(f"[ERROR] No se encontro el respaldo original en: {DOCX_IN}")
        return

    # Restaurar copia limpia desde backup
    try:
        shutil.copy2(DOCX_IN, DOCX_OUT)
    except PermissionError:
        print("\n[AVISO] Microsoft Word tiene abierto el archivo 'Articulo_SP5_completo.docx'.")
        print("Por favor, CIERRA Word (la ventana o el mensaje de diálogo) y vuelve a ejecutar este comando.\n")
        return
    doc = docx.Document(DOCX_OUT)

    moran_text = (
        "Para validar la necesidad intrínseca de una arquitectura de Gemelo Digital Territorial frente a modelos tabulares asimétricos convencionales, "
        "se evaluó la autocorrelación espacial global del resultado sanitario mediante el Índice I de Moran con una matriz de pesos espaciales de k = 5 "
        "vecinos más cercanos sobre las coordenadas centroidales (lon, lat) de los 1.632 tractos censales. "
        "El resultado confirmó una fuerte autocorrelación espacial positiva (I = 0,724, valor esperado E[I] = −0,0006, z = 45,8, p < 0,0001). "
        "Este agrupamiento significativo demuestra que la carga de morbilidad no se distribuye aleatoriamente en el territorio, sino que conforma clústeres contiguos "
        "de alta privación (hotspots en el sur y oeste de Chicago, así como en East Harlem en Manhattan), justificando formalmente la agregación espacial y la modelación a nivel de tracto censal."
    )

    vif_text = (
        "El análisis bivariado evidenció que 57 de los 91 pares de variables presentan |ρ| ≥ 0,70 (destacando la correlación entre inseguridad alimentaria y falta de transporte confiable, ρ = +0,99). "
        "El cálculo del VIF arrojó valores extremos en los determinantes socioeconómicos estructurales: Inseguridad alimentaria (VIF = 3.001,7), Ayuda alimentaria SNAP (VIF = 1.451,5), "
        "Amenaza de corte de servicios (VIF = 1.040,0) y Falta de transporte (VIF = 1.031,7). En modelos de regresión lineal simple, magnitudes de VIF > 1.000 producen inversión de signos "
        "(coeficiente lineal falsamente negativo de −11,06 en inseguridad alimentaria pese a una asociación bivariada positiva de +0,85). "
        "Este fenómeno refleja la naturaleza sindémica de los determinantes sociales: la pobreza material, la carencia de transporte y la precariedad de servicios coexisten en los mismos vecindarios. "
        "Para evitar sesgos interpretativos, la metodología adoptó regularización Ridge (L2, C = 100,0), modelos de árboles ortogonales (Gradient Boosting / Random Forest) "
        "e Importancia por Permutación agnóstica al modelo (10 repeticiones), donde la amenaza de corte de servicios (caída de F1 = 0,498) y la soledad (caída de F1 = 0,205) lideran la capacidad explicativa."
    )

    ockham_text = (
        "En la validación cruzada estratificada 5×2 (10 folds independientes), la Regresión Logística regularizada obtuvo F1-macro = 0,9278 ± 0,016 (Test = 0,9357, ROC-AUC = 0,995), "
        "mientras que Gradient Boosting alcanzó F1-macro = 0,9248 ± 0,018 (Test = 0,9333, ROC-AUC = 0,994). "
        "Aunque la prueba de Friedman detectó diferencias globales significativas (χ² = 26,76, p = 6,61×10⁻⁶), el análisis post-hoc de Nemenyi (CD = 1,483) "
        "reveló que la distancia entre los rangos medios de ambos modelos (Δ = 1,20) fue inferior a la diferencia crítica, demostrando paridad estadística formal "
        "(ratificada por la prueba t pareada corregida de Nadeau-Bengio con corrección de Holm, p_adj = 0,229). "
        "Bajo el principio de parsimonia, se seleccionó como motor operacional la Regresión Logística regularizada: garantiza menor latencia (0,011 s, cumpliendo H3), "
        "máxima interpretabilidad paramétrica para auditoría clínica y paridad predictiva frente a modelos de ensamble opacos."
    )

    new_caption = (
        "Figura 2. Análisis exploratorio de datos de la prevalencia de obesidad en adultos (CDC PLACES) y determinantes sociales en 1.632 tractos censales. "
        "(a) Distribución empírica con curva de densidad KDE, marcas de tendencia central (media = 29,92 %, mediana = 28,70 %, moda = 25,10 %) y umbrales por cuartiles (Q1 = 24,2 %, Q2 = 28,7 %, Q3 = 36,0 %). "
        "(b) Distribución territorial en Cook County, IL (n = 1.328). "
        "(c) Distribución territorial en New York County / Manhattan, NY (n = 304). "
        "(d) Matriz de correlación de rangos de Spearman entre determinantes sociales y obesidad."
    )

    def insert_after(ref_p, text, bold_title):
        new_p = doc.add_paragraph()
        if ref_p.style:
            new_p.style = ref_p.style
        r_bold = new_p.add_run(bold_title + ": ")
        r_bold.bold = True
        new_p.add_run(text)
        # Mover new_p justo después de ref_p en el árbol XML
        ref_p._p.addnext(new_p._p)
        return new_p

    # 1. Actualizar pie de Figura 2
    for p in doc.paragraphs:
        if "Figura 2" in p.text:
            p.text = new_caption
            print("[OK] Pie de Figura 2 actualizado correctamente.")
            break

    # 2. Insertar párrafos temáticos
    inserted_moran = False
    inserted_vif = False
    inserted_ockham = False

    for idx, p in enumerate(doc.paragraphs):
        text = p.text
        if not inserted_moran and ("Spearman" in text or "correlación" in text.lower()) and idx > 5:
            insert_after(p, moran_text, "[Autocorrelación espacial]")
            inserted_moran = True
            print("[OK] Párrafo de Autocorrelación Espacial (Moran) insertado.")

        if not inserted_vif and ("clasificador" in text.lower() or "regresión" in text.lower() or "multicolinealidad" in text.lower()):
            insert_after(p, vif_text, "[Evaluación de Multicolinealidad Sindémica y VIF]")
            inserted_vif = True
            print("[OK] Párrafo de Multicolinealidad y VIF insertado.")

        if not inserted_ockham and ("Nemenyi" in text or "Friedman" in text or "selección" in text.lower()):
            insert_after(p, ockham_text, "[Selección del Modelo y Principio de Parsimonia]")
            inserted_ockham = True
            print("[OK] Párrafo de Principio de Parsimonia (Ockham/Nemenyi) insertado.")

    try:
        doc.save(DOCX_OUT)
        print(f"\n[ÉXITO TOTAL] Documento generado de forma 100% compatible con Word en: {DOCX_OUT}")
    except PermissionError:
        print("\n[AVISO] No se pudo guardar porque Microsoft Word tiene abierto el archivo.")
        print("Por favor cierra Word e inténtalo de nuevo.\n")

if __name__ == "__main__":
    run()
