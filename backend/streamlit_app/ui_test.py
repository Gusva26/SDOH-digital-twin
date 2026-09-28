"""Recorre todas las vistas con AppTest y reporta excepciones.

STREAMLIT_REQUIRE_LOGIN=false python streamlit_app/ui_test.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
os.environ["STREAMLIT_REQUIRE_LOGIN"] = "false"

from streamlit.testing.v1 import AppTest  # noqa: E402

from engine import data, modeling  # noqa: E402

ds = data.load_from_db()
prep = modeling.prepare(ds.df, "pct_obesity", ds.features)
train = modeling.train_all(prep, 3, 1)

views = ["negocio", "carga", "eda", "preparacion", "entrenamiento", "hiperparametros", "validacion",
         "seleccion", "inferencia", "despliegue", "reporte", "langchain_view", "langflow_view"]
at = AppTest.from_file(os.path.join(HERE, "crispdm_app.py"), default_timeout=600)
at.run()
at.session_state.ds = ds
at.session_state.prep = prep
at.session_state.train = train
failed = 0
for v in views:
    at.switch_page(os.path.join(HERE, "views", f"{v}.py"))
    at.run()
    errs = [e.value for e in at.exception]
    failed += bool(errs)
    print(f"{'FAIL' if errs else 'ok  '} {v}: {len(at.session_state.report)} items en reporte", errs[:1])
# Botón «Generar reporte PDF»
at.switch_page(os.path.join(HERE, "views", "reporte.py"))
at.run()
btn = [b for b in at.button if b.label == "Generar reporte PDF"][0]
btn.click().run()
print("PDF:", len(at.session_state.pdf) if "pdf" in at.session_state else None, [e.value for e in at.exception][:1])
sys.exit(1 if failed else 0)
