from dataguard.api.app import app
from dataguard.api.audit_routes import router as audit_router

app.include_router(audit_router)

__all__ = ["app"]
