from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth, ai, crispdm, geo, hospitals, ml, reports, sdoh, users
from app.core.config import settings

# Register models so metadata is complete
from app.models import domain  # noqa: F401
from app.models.user import Permission, Role
from app.core.database import Base, engine, SessionLocal
from app.core.security import hash_password


def init_default_data():
    """Create tables and seed system roles + admin user."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if db.query(Role).count() == 0:
            roles = {
                "admin": ("Administrador", True),
                "clinician": ("Clínico", False),
                "analyst": ("Analista de datos", False),
                "public_health": ("Salud Pública", False),
                "viewer": ("Visor", False),
            }
            perms = {
                "hospitals:read": "Ver hospitales",
                "hospitals:write": "Gestionar hospitales",
                "sdoh:read": "Ver indicadores SDOH",
                "sdoh:write": "Editar indicadores SDOH",
                "sdoh:compute": "Calcular índices de equidad",
                "alerts:read": "Ver alertas",
                "alerts:write": "Gestionar alertas",
                "reports:read": "Ver reportes",
                "reports:generate": "Generar reportes",
                "users:manage": "Gestionar usuarios",
            }
            perm_objs = {}
            for code, name in perms.items():
                p = Permission(code=code, name=name)
                db.add(p)
                perm_objs[code] = p

            admin_role = Role(
                code="admin", name="Administrador", description="Acceso total", is_system=True
            )
            admin_role.permissions = [p for p in perm_objs.values()]

            analyst_role = Role(
                code="analyst", name="Analista de datos", description="Análisis y reportes", is_system=True
            )
            analyst_role.permissions = [
                perm_objs["hospitals:read"],
                perm_objs["sdoh:read"],
                perm_objs["sdoh:write"],
                perm_objs["sdoh:compute"],
                perm_objs["alerts:read"],
                perm_objs["alerts:write"],
                perm_objs["reports:read"],
                perm_objs["reports:generate"],
            ]

            clinician_role = Role(
                code="clinician", name="Clínico", description="Acceso a datos clínicos y de población", is_system=True
            )
            clinician_role.permissions = [
                perm_objs["hospitals:read"],
                perm_objs["sdoh:read"],
                perm_objs["alerts:read"],
                perm_objs["reports:read"],
            ]

            public_role = Role(
                code="public_health", name="Salud Pública",
                description="Monitoreo de equidad en salud", is_system=True,
            )
            public_role.permissions = [
                perm_objs["hospitals:read"],
                perm_objs["sdoh:read"],
                perm_objs["alerts:read"],
                perm_objs["reports:read"],
                perm_objs["reports:generate"],
            ]

            viewer_role = Role(
                code="viewer", name="Visor", description="Solo lectura", is_system=True
            )
            viewer_role.permissions = [
                perm_objs["hospitals:read"],
                perm_objs["sdoh:read"],
                perm_objs["alerts:read"],
            ]

            for r in [
                admin_role,
                analyst_role,
                clinician_role,
                public_role,
                viewer_role,
            ]:
                db.add(r)
            db.commit()

        from app.models.user import User as U

        role = db.query(Role).filter(Role.code == "admin").first()
        if role and db.query(U).count() == 0:
            admin_user = U(
                username="admin",
                email="admin@sdohtwin.com",
                hashed_password=hash_password("admin123"),
                is_superuser=True,
            )
            admin_user.roles = [role]
            db.add(admin_user)
            db.commit()
            print("Seeded admin user (admin / admin123)")
        else:
            print("DB already seeded.")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_default_data()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

import json

origins = json.loads(settings.BACKEND_CORS_ORIGINS or "[]")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_origin_regex=r"^https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api"
app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(users.router, prefix=API_PREFIX)
app.include_router(geo.router, prefix=API_PREFIX)
app.include_router(hospitals.router, prefix=API_PREFIX)
app.include_router(sdoh.router, prefix=API_PREFIX)
app.include_router(reports.router, prefix=API_PREFIX)
app.include_router(ai.router, prefix=API_PREFIX)
app.include_router(crispdm.router, prefix=API_PREFIX)
app.include_router(ml.router, prefix=API_PREFIX)


@app.get("/")
def root():
    return {"app": settings.APP_NAME, "version": settings.APP_VERSION, "status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}
