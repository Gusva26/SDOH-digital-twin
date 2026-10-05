import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
DOCX_FILE = os.path.join(HERE, "Articulo_SP5_completo.docx")
BACKUP_FILE = os.path.join(HERE, "Articulo_SP5_completo_backup.docx")

if os.path.exists(BACKUP_FILE):
    shutil.copy2(BACKUP_FILE, DOCX_FILE)
    print(f"[OK] Documento restaurado exitosamente desde: {BACKUP_FILE}")
else:
    print("[ERROR] No se encontro el archivo de respaldo.")
