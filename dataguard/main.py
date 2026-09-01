from fastapi import Request
from fastapi.responses import JSONResponse

from dataguard.api.app import app
from dataguard.api.audit_routes import router as audit_router
from dataguard.api.auth_routes import router as auth_router
import dataguard.audit.persistence  # noqa: F401
from dataguard.processing.validation import UnsafeDocumentError

app.include_router(audit_router)
app.include_router(auth_router)


async def _unsafe_document_handler(request: Request, exc: UnsafeDocumentError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


app.add_exception_handler(UnsafeDocumentError, _unsafe_document_handler)

__all__ = ["app"]
