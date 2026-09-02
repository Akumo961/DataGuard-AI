from fastapi import Request
from fastapi.responses import JSONResponse

from dataguard.api.app import app
from dataguard.api.audit_routes import router as audit_router
from dataguard.audit import persistence as _audit_persistence  # noqa: F401
from dataguard.processing.validation import UnsafeDocumentError

app.include_router(audit_router)

_document_route = next(
    route for route in app.routes if getattr(route, "path", None) == "/api/v1/analyze-document"
)
app.add_api_route(
    "/api/v1/documents",
    _document_route.endpoint,
    methods=["POST"],
    response_model=getattr(_document_route, "response_model", None),
    tags=["analysis"],
)


async def _unsafe_document_handler(request: Request, exc: UnsafeDocumentError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


app.add_exception_handler(UnsafeDocumentError, _unsafe_document_handler)

__all__ = ["app"]
