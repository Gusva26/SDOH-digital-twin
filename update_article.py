"""Script automatizado para actualizar Articulo_SP5_completo.docx

Aplica las siguientes mejoras:
1. Reemplaza la imagen de la Figura 2 por la nueva versión de 4 paneles (backend/article_outputs/fig2_eda.png).
2. Actualiza el pie de la Figura 2 con la descripción de los 4 paneles y los mapas territoriales.
3. Inserta el reporte e interpretación formal del Índice I de Moran (I = 0.724, p < 0.0001) en la sección de EDA.
4. Inserta la tabla de VIF y la justificación de multicolinealidad sindémica en la sección de modelado/discusión.
5. Agrega la fundamentación del principio de parsimonia (Navaja de Ockham / Nemenyi / Nadeau-Bengio) en la selección del modelo.
6. Realiza un respaldo previo: Articulo_SP5_completo_backup.docx.
"""

import os
import shutil
import zipfile
import copy
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
DOCX_IN = os.path.join(HERE, "Articulo_SP5_completo.docx")
DOCX_BACKUP = os.path.join(HERE, "Articulo_SP5_completo_backup.docx")
DOCX_OUT = os.path.join(HERE, "Articulo_SP5_completo.docx")
FIG2_PATH = os.path.join(HERE, "backend", "article_outputs", "fig2_eda.png")

NS = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'rel': 'http://schemas.openxmlformats.org/package/2006/relationships',
}

for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)

def backup():
    if not os.path.exists(DOCX_BACKUP):
        shutil.copy2(DOCX_IN, DOCX_BACKUP)
        print(f"[OK] Respaldo creado en: {DOCX_BACKUP}")
    else:
        print(f"[INFO] Respaldo existente en: {DOCX_BACKUP}")

def update_docx():
    backup()
    
    with zipfile.ZipFile(DOCX_IN, 'r') as zin:
        files = {name: zin.read(name) for name in zin.namelist()}

    # 1. Identificar la imagen de la Figura 2 a través de document.xml y document.xml.rels
    doc_tree = ET.fromstring(files['word/document.xml'])
    rels_tree = ET.fromstring(files['word/_rels/document.xml.rels'])
    
    # Mapeo de rId -> target
    rid_map = {}
    for rel in rels_tree.iter('{http://schemas.openxmlformats.org/package/2006/relationships}Relationship'):
        rid_map[rel.attrib.get('Id')] = rel.attrib.get('Target')

    # Buscar el párrafo con "Figura 2" o "fig2"
    fig2_media_name = None
    target_p = None
    for p in doc_tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
        p_text = ''.join(t.text for t in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if t.text)
        if "Figura 2" in p_text or "fig2" in p_text.lower():
            # Buscar blip en este párrafo o en el anterior/siguiente
            for blip in p.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}blip'):
                embed = blip.attrib.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                if embed and embed in rid_map:
                    fig2_media_name = "word/" + rid_map[embed].lstrip('/')
                    target_p = p
                    break

    # Si no se encontró directamente en el mismo párrafo, buscar la imagen previa más cercana
    if not fig2_media_name:
        all_p = list(doc_tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'))
        for idx, p in enumerate(all_p):
            p_text = ''.join(t.text for t in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if t.text)
            if "Figura 2" in p_text:
                target_p = p
                # Buscar en párrafos adyacentes (idx-2 a idx+2)
                for near_idx in range(max(0, idx - 3), min(len(all_p), idx + 3)):
                    for blip in all_p[near_idx].iter('{http://schemas.openxmlformats.org/drawingml/2006/main}blip'):
                        embed = blip.attrib.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                        if embed and embed in rid_map:
                            target_file = rid_map[embed].replace('\\', '/').split('/')[-1]
                            fig2_media_name = "word/media/" + target_file
                            break
                    if fig2_media_name:
                        break
                break

    media_files = [k for k in files.keys() if 'media' in k.lower()]
    print(f"[DEBUG] Archivos multimedia encontrados en docx: {media_files}")

    # 2. Reemplazar la imagen física en word/media/
    replaced_img = False
    if fig2_media_name and fig2_media_name in files and os.path.exists(FIG2_PATH):
        with open(FIG2_PATH, 'rb') as f_img:
            files[fig2_media_name] = f_img.read()
        print(f"[OK] Reemplazada imagen de la Figura 2 en {fig2_media_name} con {FIG2_PATH}")
        replaced_img = True
    else:
        # Si no se halló por nombre exacto, buscar en media
        for name in media_files:
            if any(term in name.lower() for term in ["image2", "fig2", "image_2"]):
                with open(FIG2_PATH, 'rb') as f_img:
                    files[name] = f_img.read()
                print(f"[OK] Reemplazada imagen en {name} con {FIG2_PATH}")
                replaced_img = True
                break
    
    if not replaced_img:
        print("[AVISO] No se identificó automáticamente la imagen de Figura 2 en word/media/.")

    # 3. Actualizar textos en document.xml
    body = doc_tree.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}body')
    
    # Texto de pie de Figura 2 actualizado
    new_caption = (
        "Figura 2. Análisis exploratorio de datos de la prevalencia de obesidad en adultos (CDC PLACES) y determinantes sociales en 1.632 tractos censales. "
        "(a) Distribución empírica con curva de densidad KDE, marcas de tendencia central (media = 29,92 %, mediana = 28,70 %, moda = 25,10 %) y umbrales por cuartiles (Q1 = 24,2 %, Q2 = 28,7 %, Q3 = 36,0 %). "
        "(b) Distribución territorial en Cook County, IL (n = 1.328). "
        "(c) Distribución territorial en New York County / Manhattan, NY (n = 304). "
        "(d) Matriz de correlación de rangos de Spearman entre determinantes sociales y obesidad."
    )

    # Texto de Moran's I
    moran_text = (
        "Para validar la necesidad intrínseca de una arquitectura de Gemelo Digital Territorial frente a modelos tabulares asimétricos convencionales, "
        "se evaluó la autocorrelación espacial global del resultado sanitario mediante el Índice I de Moran con una matriz de pesos espaciales de k = 5 "
        "vecinos más cercanos sobre las coordenadas centroidales (lon, lat) de los 1.632 tractos censales. "
        "El resultado confirmó una fuerte autocorrelación espacial positiva (I = 0,724, valor esperado E[I] = −0,0006, z = 45,8, p < 0,0001). "
        "Este agrupamiento significativo demuestra que la carga de morbilidad no se distribuye aleatoriamente en el territorio, sino que conforma clústeres contiguos "
        "de alta privación (hotspots en el sur y oeste de Chicago, así como en East Harlem en Manhattan), justificando formalmente la agregación espacial y la modelación a nivel de tracto censal."
    )

    # Texto de VIF y Multicolinealidad
    vif_text = (
        "Evaluación de Multicolinealidad Sindémica y Factor de Inflación de la Varianza (VIF): "
        "El análisis bivariado evidenció que 57 de los 91 pares de variables presentan |ρ| ≥ 0,70 (destacando la correlación entre inseguridad alimentaria y falta de transporte confiable, ρ = +0,99). "
        "El cálculo del VIF arrojó valores extremos en los determinantes socioeconómicos estructurales: Inseguridad alimentaria (VIF = 3.001,7), Ayuda alimentaria SNAP (VIF = 1.451,5), "
        "Amenaza de corte de servicios (VIF = 1.040,0) y Falta de transporte (VIF = 1.031,7). En modelos de regresión lineal simple, magnitudes de VIF > 1.000 producen inversión de signos "
        "(coeficiente lineal falsamente negativo de −11,06 en inseguridad alimentaria pese a una asociación bivariada positiva de +0,85). "
        "Este fenómeno refleja la naturaleza sindémica de los determinantes sociales: la pobreza material, la carencia de transporte y la precariedad de servicios coexisten en los mismos vecindarios. "
        "Para evitar sesgos interpretativos, la metodología adoptó regularización Ridge (L2, C = 100,0), modelos de árboles ortogonales (Gradient Boosting / Random Forest) "
        "e Importancia por Permutación agnóstica al modelo (10 repeticiones), donde la amenaza de corte de servicios (caída de F1 = 0,498) y la soledad (caída de F1 = 0,205) lideran la capacidad explicativa."
    )

    # Texto de Selección de Modelos (Ockham / Nemenyi)
    ockham_text = (
        "Selección del Modelo y Principio de Parsimonia (Navaja de Ockham): "
        "En la validación cruzada estratificada 5×2 (10 folds independientes), la Regresión Logística regularizada obtuvo F1-macro = 0,9278 ± 0,016 (Test = 0,9357, ROC-AUC = 0,995), "
        "mientras que Gradient Boosting alcanzó F1-macro = 0,9248 ± 0,018 (Test = 0,9333, ROC-AUC = 0,994). "
        "Aunque la prueba de Friedman detectó diferencias globales significativas (χ² = 26,76, p = 6,61×10⁻⁶), el análisis post-hoc de Nemenyi (CD = 1,483) "
        "reveló que la distancia entre los rangos medios de ambos modelos (Δ = 1,20) fue inferior a la diferencia crítica, demostrando paridad estadística formal "
        "(ratificada por la prueba t pareada corregida de Nadeau-Bengio con corrección de Holm, p_adj = 0,229). "
        "Bajo el principio de parsimonia, se seleccionó como motor operacional la Regresión Logística regularizada: garantiza menor latencia (0,011 s, cumpliendo H3), "
        "máxima interpretabilidad paramétrica para auditoría clínica y paridad predictiva frente a modelos de ensamble opacos."
    )

    def create_p(text, bold_prefix=None, template_p=None):
        p = ET.Element('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p')
        if template_p is not None:
            pPr = template_p.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr')
            if pPr is not None:
                p.append(copy.deepcopy(pPr))
        if bold_prefix:
            r_bold = ET.SubElement(p, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r')
            rPr = ET.SubElement(r_bold, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rPr')
            ET.SubElement(rPr, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}b')
            t_bold = ET.SubElement(r_bold, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t')
            t_bold.text = bold_prefix + " "
        r = ET.SubElement(p, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r')
        t = ET.SubElement(r, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t')
        t.text = text
        return p

    # Modificar el pie de Figura 2 si se encontró
    caption_updated = False
    for p in doc_tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
        p_text = ''.join(t.text for t in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if t.text)
        if "Figura 2" in p_text:
            # Reemplazar el texto del párrafo
            for t in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
                t.text = ""
            r = p.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r')
            if r is not None:
                t = r.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t')
                if t is not None:
                    t.text = new_caption
                else:
                    new_t = ET.SubElement(r, '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t')
                    new_t.text = new_caption
            caption_updated = True
            print("[OK] Pie de Figura 2 actualizado con los 4 paneles.")
            break

    # Verificar si los textos ya fueron previamente insertados para evitar duplicación
    doc_full_text = ''.join(doc_tree.itertext())
    inserted_moran = "Autocorrelación espacial" in doc_full_text
    inserted_vif = "Multicolinealidad y VIF" in doc_full_text
    inserted_ockham = "Principio de Parsimonia" in doc_full_text

    if inserted_moran:
        print("[INFO] Párrafo de Índice I de Moran ya presente en el documento.")
    if inserted_vif:
        print("[INFO] Párrafo de VIF ya presente en el documento.")
    if inserted_ockham:
        print("[INFO] Párrafo de Ockham/Nemenyi ya presente en el documento.")

    all_p = list(body.findall('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'))
    for idx, p in enumerate(all_p):
        p_text = ''.join(t.text for t in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if t.text)
        
        # Insertar Moran después de mencionar correlación de Spearman o EDA
        if not inserted_moran and ("Spearman" in p_text or "correlación" in p_text.lower()) and idx > 5:
            pos = list(body).index(p)
            body.insert(pos + 1, create_p(moran_text, "[Autocorrelación espacial]", template_p=p))
            inserted_moran = True
            print("[OK] Párrafo de Índice I de Moran insertado.")

        # Insertar VIF en sección de modelado / determinantes
        if not inserted_vif and ("clasificador" in p_text.lower() or "regresión" in p_text.lower() or "multicolinealidad" in p_text.lower()):
            pos = list(body).index(p)
            body.insert(pos + 1, create_p(vif_text, "[Análisis de Multicolinealidad y VIF]", template_p=p))
            inserted_vif = True
            print("[OK] Párrafo y diagnóstico de VIF insertado.")

        # Insertar Ockham cerca de la discusión de selección de modelos / Nemenyi
        if not inserted_ockham and ("Nemenyi" in p_text or "Friedman" in p_text or "selección" in p_text.lower()):
            pos = list(body).index(p)
            body.insert(pos + 1, create_p(ockham_text, "[Principio de Parsimonia y Paridad Estadística]", template_p=p))
            inserted_ockham = True
            print("[OK] Párrafo de Navaja de Ockham y Nemenyi insertado.")

    # Guardar document.xml actualizado
    files['word/document.xml'] = ET.tostring(doc_tree, encoding='utf-8', xml_declaration=True)

    # Escribir de vuelta el archivo .docx
    with zipfile.ZipFile(DOCX_OUT, 'w', zipfile.ZIP_DEFLATED) as zout:
        for name, content in files.items():
            zout.writestr(name, content)

    print(f"\n[ÉXITO] Manuscrito actualizado y guardado en: {DOCX_OUT}")

if __name__ == "__main__":
    update_docx()
