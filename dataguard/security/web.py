"""HTTP security dependencies and middleware."""

from collections.abc import Awaitable, Callable
from secrets import token_hex

from fastapi import Header, HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from dataguard.core.config import get_settings
from dataguard.security.auth import AuthenticatedPrincipal, decode_access_token


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID")
        if not request_id or len(request_id) > 128 or any(ord(c) < 32 for c in request_id):
            request_id = token_hex(16)
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cache-Control"] = (
            "no-store" if request.url.path.startswith("/api/") else "no-cache"
        )
        if get_settings().security_headers_enabled:
            # The application serves its frontend assets from the same origin. Keep the
            # policy restrictive while explicitly allowing those static assets to execute.
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self'; "
                "style-src 'self'; "
                "img-src 'self' data:; "
                "object-src 'none'; "
                "base-uri 'none'; "
                "form-action 'self'; "
                "frame-ancestors 'none'"
            )
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


def get_current_principal(
    authorization: str | None = Header(default=None),
) -> AuthenticatedPrincipal:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )
    token = authorization[7:].strip()
    try:
        return decode_access_token(token)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid access token"
        ) from exc
