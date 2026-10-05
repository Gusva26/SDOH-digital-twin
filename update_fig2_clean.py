"""Reemplazo exacto de la imagen y pie de Figura 2 en Articulo_SP5_completo.docx usando python-docx."""
import os
import struct
import docx

HERE = os.path.dirname(os.path.abspath(__file__))
DOCX_PATH = os.path.join(HERE, "Articulo_SP5_completo.docx")
FIG2_PATH = os.path.join(HERE, "backend", "article_outputs", "fig2_eda.png")

def get_png_dimensions(path):
    with open(path, 'rb') as f:
        f.seek(16)
        w, h = struct.unpack('>II', f.read(8))
    return w, h

def run():
    if not os.path.exists(DOCX_PATH):
        print(f"[ERROR] No existe {DOCX_PATH}")
        return
    if not os.path.exists(FIG2_PATH):
        print(f"[ERROR] No existe {FIG2_PATH}")
        return

    try:
        doc = docx.Document(DOCX_PATH)
    except PermissionError:
        print("\n[AVISO] Microsoft Word tiene abierto el archivo. Por favor, CIERRA Word y vuelve a ejecutar.\n")
        return

    # Leer dimensiones de la nueva imagen
    w_px, h_px = get_png_dimensions(FIG2_PATH)
    aspect = h_px / w_px
    # Ancho deseado: 6.2 pulgadas (ajustado a los márgenes estándar de Word)
    width_emu = int(6.2 * 914400)
    height_emu = int(width_emu * aspect)

    with open(FIG2_PATH, 'rb') as f:
        new_img_bytes = f.read()

    # 1. Buscar el párrafo que contiene el pie de Figura 2 (empieza con "Figura 2")
    caption_p = None
    caption_idx = -1
    for idx, p in enumerate(doc.paragraphs):
        txt = p.text.strip()
        if txt.startswith("Figura 2") or txt.startswith("Fig. 2") or "Figura 2. Análisis exploratorio" in txt:
            caption_p = p
            caption_idx = idx
            break

    if caption_p is None:
        print("[ERROR] No se encontró el párrafo del pie de Figura 2.")
        return

    print(f"[OK] Encontrado pie de Figura 2 en párrafo #{caption_idx}: '{caption_p.text[:60]}...'")

    # 2. Buscar la imagen en los párrafos adyacentes anteriores (idx-1, idx-2, etc.)
    image_replaced = False
    for look_idx in range(max(0, caption_idx - 3), caption_idx + 1):
        p = doc.paragraphs[look_idx]
        for blip in p._p.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}blip'):
            rId = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
            if rId and rId in doc.part.related_parts:
                img_part = doc.part.related_parts[rId]
                img_part._blob = new_img_bytes
                image_replaced = True
                print(f"[OK] Imagen reemplazada con éxito (relId: {rId}, parte: {img_part.partname}).")

                # Ajustar las dimensiones de escala para los 4 paneles
                for extent in p._p.iter('{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent'):
                    extent.set('cx', str(width_emu))
                    extent.set('cy', str(height_emu))
                for ext in p._p.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}ext'):
                    ext.set('cx', str(width_emu))
                    ext.set('cy', str(height_emu))
                break
        if image_replaced:
            break

    if not image_replaced:
        print("[AVISO] No se halló el elemento de dibujo en el párrafo previo. Buscando en todo el documento...")
        for p in doc.paragraphs:
            for blip in p._p.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}blip'):
                rId = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                if rId and rId in doc.part.related_parts:
                    img_part = doc.part.related_parts[rId]
                    # Si es una imagen PNG o JPEG
                    if "image2" in img_part.partname.lower() or "fig2" in img_part.partname.lower():
                        img_part._blob = new_img_bytes
                        image_replaced = True
                        print(f"[OK] Reemplazada imagen en {img_part.partname}")
                        break
            if image_replaced:
                break

    # 3. Actualizar el texto del pie de figura con formato enriquecido (Figura 2. en negrita)
    new_caption_body = (
        "Análisis exploratorio de datos de la prevalencia de obesidad en adultos (CDC PLACES) y determinantes sociales en 1.632 tractos censales. "
        "(a) Distribución empírica con curva de densidad KDE, marcas de tendencia central (media = 29,92 %, mediana = 28,70 %, moda = 25,10 %) y umbrales por cuartiles (Q1 = 24,2 %, Q2 = 28,7 %, Q3 = 36,0 %). "
        "(b) Distribución territorial en Cook County, IL (n = 1.328). "
        "(c) Distribución territorial en New York County / Manhattan, NY (n = 304). "
        "(d) Matriz de correlación de rangos de Spearman entre determinantes sociales y obesidad."
    )

    caption_p.text = ""
    r_bold = caption_p.add_run("Figura 2. ")
    r_bold.bold = True
    caption_p.add_run(new_caption_body)
    print("[OK] Pie de Figura 2 actualizado con los 4 paneles y negrita.")

    try:
        doc.save(DOCX_PATH)
        print(f"\n[ÉXITO COMPLETO] Figura 2 actualizada con 4 paneles en: {DOCX_PATH}")
    except PermissionError:
        print("\n[AVISO] No se pudo guardar porque Microsoft Word tiene abierto el archivo.")
        print("Por favor, CIERRA Word y vuelve a ejecutar: python update_fig2_clean.py\n")

if __name__ == "__main__":
    run()
