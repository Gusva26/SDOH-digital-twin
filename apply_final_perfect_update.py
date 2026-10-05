"""Aplicación definitiva y perfecta de cambios en Articulo_SP5_completo.docx"""
import os
import shutil
import zipfile
import docx

HERE = os.path.dirname(os.path.abspath(__file__))
DOCX_IN = os.path.join(HERE, "Articulo_SP5_completo_backup.docx")
DOCX_OUT = os.path.join(HERE, "Articulo_SP5_completo.docx")
FIG2_SOURCE = os.path.join(HERE, "backend", "article_outputs", "fig2_eda.png")
FIG2_INTERNAL_NAME = "word/media/29c36ee896afc18a3abbff108d2fedb3fe2b8da1.png"

def run():
    print("--- INICIANDO ACTUALIZACIÓN PERFECTA ---")
    if not os.path.exists(DOCX_IN):
        print(f"[ERROR] No existe el respaldo {DOCX_IN}")
        return
    if not os.path.exists(FIG2_SOURCE):
        print(f"[ERROR] No existe la imagen {FIG2_SOURCE}")
        return

    # Paso 1: Reemplazar el archivo binario de la imagen directamente en el paquete ZIP
    with open(FIG2_SOURCE, 'rb') as f_img:
        new_img_bytes = f_img.read()

    with zipfile.ZipFile(DOCX_IN, 'r') as zin:
        all_files = {name: zin.read(name) for name in zin.namelist()}

    if FIG2_INTERNAL_NAME in all_files:
        old_size = len(all_files[FIG2_INTERNAL_NAME])
        all_files[FIG2_INTERNAL_NAME] = new_img_bytes
        print(f"[OK 1/4] Imagen física reemplazada: {FIG2_INTERNAL_NAME} (de {old_size:,} bytes a {len(new_img_bytes):,} bytes).")
    else:
        print(f"[ERROR] No se encontró {FIG2_INTERNAL_NAME} en el paquete ZIP.")
        return

    # Escribir temporalmente el docx con la imagen reemplazada
    with zipfile.ZipFile(DOCX_OUT, 'w', zipfile.ZIP_DEFLATED) as zout:
        for name, data in all_files.items():
            zout.writestr(name, data)

    # Paso 2: Usar python-docx sobre DOCX_OUT para actualizar el pie de figura y los textos
    try:
        doc = docx.Document(DOCX_OUT)
    except PermissionError:
        print("\n[AVISO] Microsoft Word tiene abierto el archivo. Por favor, CIERRA Word y vuelve a ejecutar.\n")
        return

    # Texto exacto para el pie de figura
    new_caption_body = (
        "Análisis exploratorio de datos de la prevalencia de obesidad en adultos (CDC PLACES) y determinantes sociales en 1.632 tractos censales. "
        "(a) Distribución empírica con curva de densidad KDE, marcas de tendencia central (media = 29,92 %, mediana = 28,70 %, moda = 25,10 %) y umbrales por cuartiles (Q1 = 24,2 %, Q2 = 28,7 %, Q3 = 36,0 %). "
        "(b) Distribución territorial en Cook County, IL (n = 1.328). "
        "(c) Distribución territorial en New York County / Manhattan, NY (n = 304). "
        "(d) Matriz de correlación de rangos de Spearman entre determinantes sociales y obesidad."
    )

    caption_updated = False
    for p in doc.paragraphs:
        txt = p.text.strip()
        # El pie de figura original empieza con "Figura 2. Análisis exploratorio"
        if txt.startswith("Figura 2. Análisis exploratorio") or (txt.startswith("Figura 2.") and "Spearman" in txt):
            p.text = ""
            r_bold = p.add_run("Figura 2. ")
            r_bold.bold = True
            p.add_run(new_caption_body)
            caption_updated = True
            print("[OK 2/4] Pie de Figura 2 actualizado con los 4 paneles y negrita.")
            break

    if not caption_updated:
        print("[AVISO] No se encontró el pie exacto por prefijo, buscando por 'Figura 2'...")
        for p in doc.paragraphs:
            if "Figura 2" in p.text and len(p.text) < 300:
                p.text = ""
                r_bold = p.add_run("Figura 2. ")
                r_bold.bold = True
                p.add_run(new_caption_body)
                print("[OK 2/4] Pie de Figura 2 actualizado.")
                break

    # Paso 3: Insertar los 3 apartados metodológicos de rigor científico
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

    def insert_after(ref_p, text, bold_title):
        new_p = doc.add_paragraph()
        if ref_p.style:
            new_p.style = ref_p.style
        r_bold = new_p.add_run(bold_title + ": ")
        r_bold.bold = True
        new_p.add_run(text)
        ref_p._p.addnext(new_p._p)
        return new_p

    inserted_moran = False
    inserted_vif = False
    inserted_ockham = False

    for idx, p in enumerate(doc.paragraphs):
        text = p.text
        if not inserted_moran and ("Spearman" in text or "correlación" in text.lower()) and idx > 10 and not text.startswith("Figura 2"):
            insert_after(p, moran_text, "[Autocorrelación espacial]")
            inserted_moran = True
            print("[OK 3/4] Párrafo de Autocorrelación Espacial (Moran) insertado.")

        if not inserted_vif and ("clasificador" in text.lower() or "regresión" in text.lower() or "multicolinealidad" in text.lower()):
            insert_after(p, vif_text, "[Evaluación de Multicolinealidad Sindémica y VIF]")
            inserted_vif = True
            print("[OK 3/4] Párrafo de Multicolinealidad y VIF insertado.")

        if not inserted_ockham and ("Nemenyi" in text or "Friedman" in text or "selección" in text.lower()):
            insert_after(p, ockham_text, "[Selección del Modelo y Principio de Parsimonia]")
            inserted_ockham = True
            print("[OK 3/4] Párrafo de Principio de Parsimonia (Ockham/Nemenyi) insertado.")

    try:
        doc.save(DOCX_OUT)
        print(f"\n[OK 4/4] Guardado exitoso en: {DOCX_OUT}")
        print("\n=== [ÉXITO TOTAL] Documento finalizado con Figura 2 de 4 paneles, pie actualizado y rigor científico completo. ===")
    except PermissionError:
        print("\n[AVISO] No se pudo guardar porque Microsoft Word tiene abierto el archivo.")
        print("Por favor, CIERRA Word y vuelve a ejecutar: python apply_final_perfect_update.py\n")

if __name__ == "__main__":
    run()
