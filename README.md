# 🏥 SDOH Digital Twin 3D

**A Digital Twin Framework for Real-Time Monitoring of Social Determinants of Health (SDOH) and Health Equity in Metropolitan Hospital Catchment Areas.**

Aplicación full-stack de gemelos digitales 3D para monitorear determinantes sociales de la salud y equidad en salud dentro de las áreas de captación de hospitales metropolitanos, basada en **datasets públicos** (CDC PLACES y Census ACS).

---

## ✨ Módulos incluidos

| Módulo | Descripción |
|--------|-------------|
| **Dashboard 3D** | Gemelo digital 3D (Three.js / React-Three-Fiber) del área hospitalaria; cada census tract es una voxel coloreada y con altura según su nivel de riesgo o índice de equidad. |
| **Mapa SDOH** | Exploración de indicadores de determinantes sociales (ingreso, educación, vivienda, empleo, transporte, alimentación, acceso a salud, etc.) agrupados por dominio. |
| **Equidad en Salud** | Cálculo de índices compuestos de equidad por census tract, percentiles de vulnerabilidad y niveles de riesgo (low/moderate/high/critical). |
| **Hospitales & Catchments** | Registro de hospitales y sus áreas de captación (catchment areas) con asignación de census tracts. |
| **Alertas** | Reglas de alerta configurables y generación de alertas por umbrales de indicadores. |
| **Reportes** | Generación de reportes en **PDF, Word (.docx), Excel (.xlsx) y CSV**. |
| **Usuarios, Roles y Perfiles** | Autenticación JWT, roles con permisos granulares (admin, clínico, analista, salud pública, visor), perfiles y auditoría. |
| **Metodología CRISP-DM** | Las 6 fases del proceso (comprensión del negocio, comprensión de los datos, preparación, modelado, evaluación y despliegue) navegables desde el sidebar, cada una consultando el estado real del sistema. Incluye banco de pruebas de latencia y sensibilidad de pesos. Ver [`docs/CRISP-DM.md`](docs/CRISP-DM.md). |
| **Modelos predictivos (ML)** | Entrena y compara 4 clasificadores (Regresión Logística, Random Forest, Gradient Boosting, Red Neuronal MLP) con los datos reales de CDC PLACES para predecir el nivel de riesgo de un resultado de salud por census tract a partir de sus determinantes sociales. El mejor (F1-macro en validación cruzada k=5) se guarda en `backend/ml_artifacts/best_model.joblib` y se sirve en `/api/ml/*`; la app lo consume en el Dashboard 3D (modo *Predicción ML*), en la página *Modelos ML* y en las fases IV–VI de CRISP-DM. *Ejecutar pipeline* en CRISP-DM recorre III → IV (índice + ML) → V → VI. Entrenar: `make train` o el botón de la página. |

---

## 🧱 Stack tecnológico

- **Backend:** Python 3.12 · FastAPI · SQLAlchemy · GeoAlchemy2 (PostGIS)
- **Frontend:** React 18 · TypeScript · Vite · Three.js / React-Three-Fiber / Drei · Recharts
- **Base de datos:** PostgreSQL 16 + PostGIS 3.4
- **ETL:** descarga automática del dataset público **CDC PLACES** (Socrata, sin API key) + geometrías TIGERweb; script `app.scripts.etl_pipeline`

---

## 📁 Estructura del proyecto

```
.
├── docker-compose.yml          # PostGIS + backend + frontend
├── .env.example                # Variables de entorno (copiar a .env)
├── Makefile                    # atajos (up, down, seed, etl, etc.)
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py             # App FastAPI + seed de roles/admin
│       ├── core/               # config, database, security
│       ├── models/             # SQLAlchemy + PostGIS (geo, sdoh, users, reports)
│       ├── schemas/            # Pydantic
│       ├── api/                # Routers (auth, users, geo, hospitals, sdoh, reports)
│       ├── services/           # equity, reportes (PDF/Word/Excel/CSV), auditoría
│       └── scripts/            # seed, migrate, etl_pipeline
└── frontend/
    ├── Dockerfile
    ├── package.json
    └── src/
        ├── api/                # Cliente axios + interceptor JWT
        ├── components/         # Layout, Twin3D, StatCard, ProtectedRoute
        ├── pages/              # Dashboard, SdohMap, Equity, Hospitals, Alerts, Reports, Users, Profile
        ├── store/              # zustand (auth)
        └── types/
```

---

## 🚀 Puesta en marcha (Docker Compose)

Requisito: **Docker** con plugin Compose.

```bash
# 1. Configurar variables de entorno
cp .env.example .env

# 2. Levantar la infraestructura (PostGIS + backend + frontend)
docker compose up -d --build

# 3. Cargar el dataset (recomendado: público CDC PLACES)
docker compose exec backend python -m app.scripts.etl_pipeline --public

#    Alternativa demo/sintética: docker compose exec backend python -m app.scripts.seed

# 4. Acceder
#    Frontend:  http://localhost:5173
#    API docs:  http://localhost:8000/docs
```

> El backend crea tablas y seed de **roles, permisos y usuario admin** automáticamente al arrancar.

### Credenciales por defecto

| Usuario | Contraseña | Rol |
|---------|-----------|-----|
| `admin` | `admin123` | Administrador (superusuario) |

---

## 📦 Importar datasets públicos reales

El sistema usa **CDC PLACES** (Local Data for Better Health, Census Tract Data **2025 release**) como
dataset público por defecto, distribuido en dominio público y accesible vía la Socrata Open Data API
**sin necesidad de API key**. El ETL descarga automáticamente las estimaciones por census tract
(valor + IC 95% + población) para New York County (36061) y Cook County (17031), descarga las
geometrías reales de los tractos desde **TIGERweb** (Census Bureau), construye el catálogo SDOH,
persiste condados/tractos/indicadores, asigna los tractos a los catchment areas de los hospitales,
calcula los **índices de equidad reales** y genera alertas por reglas:

```bash
make seed-public          # ETL completo con CDC PLACES (año 2022)
# o explícitamente:
docker compose exec backend python -m app.scripts.etl_pipeline --public --year 2022
```

Opciones del ETL:

```bash
python -m app.scripts.etl_pipeline --public --year 2023                 # otro año
python -m app.scripts.etl_pipeline --public --counties 17031,36061 --radius 12
python -m app.scripts.etl_pipeline --public --offline                   # reusar CSV cacheado

# Importar un CSV de CDC PLACES descargado a mano (compat)
python -m app.scripts.etl_pipeline --cdc backend/data/raw/places.csv

# Dataset demo/sintético (solo para pruebas rápidas)
python -m app.scripts.etl_pipeline --demo    # o: make seed
```

Los CSV crudos se cachean en `backend/data/raw/places_cdc_<año>.csv`. El detalle de fuentes,
mapeo de medidas y funciones está en `backend/app/scripts/etl_pipeline.py`.

---

## 🛠️ Comandos útiles (Makefile)

```bash
make up          # levantar servicios
make down        # detener
make logs        # ver logs
make seed        # cargar dataset demo (reinicia tablas)
make seed-etl    # ejecutar ETL
make migrate     # seed roles/admin + migraciones ligeras (seguro de repetir)
make psql        # consola SQL
```

---

## 🔌 API principal (resumen)

| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/api/auth/login` | Iniciar sesión (JWT) |
| GET | `/api/auth/me` | Usuario actual |
| GET/POST | `/api/users` | Listar / crear usuarios (admin) |
| GET | `/api/users/roles/all` | Roles y permisos |
| GET | `/api/geo/health-system/stats` | Estadísticas del sistema |
| GET | `/api/geo/catchments` | Áreas de captación |
| GET/POST | `/api/hospitals` | Hospitales |
| GET | `/api/sdoh/catalog` | Catálogo de indicadores |
| GET | `/api/sdoh/values` | Valores SDOH |
| GET | `/api/sdoh/equity` | Índices de equidad |
| POST | `/api/sdoh/equity/compute` | Recalcular índices |
| GET | `/api/sdoh/alerts` | Alertas |
| POST | `/api/sdoh/alerts/generate` | Generar alertas por reglas |
| POST | `/api/reports/generate` | Generar reporte (pdf/word/excel/csv) |
| GET | `/api/reports/download/{id}` | Descargar reporte |
| GET | `/api/crispdm/phases` | Estado de las 6 fases CRISP-DM |
| GET | `/api/crispdm/{fase}` | Detalle de una fase (`business-understanding`, `data-understanding`, `data-preparation`, `modeling`, `evaluation`, `deployment`) |
| POST | `/api/crispdm/evaluation/run` | Banco de pruebas: latencia (H3) y sensibilidad de pesos |
| POST | `/api/crispdm/pipeline/run` | Ejecuta las fases III → IV → V encadenadas |

Documentación interactiva completa en **http://localhost:8000/docs** (Swagger UI).

---

## ⚠️ Notas

- **Python 3.14 local:** algunas librerías geoespaciales (`shapely`, `psycopg2-binary`) aún no publican
  wheels para Python 3.14 en Windows. El proyecto está fijado a **Python 3.12** dentro de Docker, donde
  todas las dependencias se instalan sin problemas. Para desarrollo local use Python 3.12.
- Los reportes generados se guardan en `backend/generated_reports/`.

---

## 🗂️ Modelo de datos clave

- **counties / census_tracts** — geometrías PostGIS (MultiPolygon) de unidades censales.
- **hospitals / hospital_catchments / catchment_memberships** — hospitales, áreas de captación y su
  relación con los census tracts (con fracción de población).
- **indicator_catalog / sdoh_indicators** — catálogo de indicadores SDOH y sus valores por tract/año.
- **equity_indexes** — índices de equidad calculados por tract.
- **alert_rules / alerts** — reglas y alertas generadas.
- **users / roles / permissions / profiles / audit_logs** — seguridad y auditoría.
```
