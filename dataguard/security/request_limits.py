"""Request-size guard applied before application handlers."""

from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from dataguard.core.config import get_settings


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request, call_next: Callable[[object], Awaitable[Response]]
    ) -> Response:
        declared = request.headers.get("content-length")
        if declared:
            try:
                if int(declared) > get_settings().max_request_body_bytes:
                    return JSONResponse({"detail": "Request body too large"}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
        return await call_next(request)
