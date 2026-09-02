"""Rate-limit middleware with Redis-backed production enforcement."""

from collections.abc import Awaitable, Callable
from uuid import uuid4

from redis.asyncio import Redis
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from dataguard.core.config import get_settings
from dataguard.processing.validation import UnsafeDocumentError
from dataguard.security.audit_context import AuditRequestContext, reset_context, set_context
from dataguard.security.malware import MalwareScannerUnavailableError
from dataguard.security.metrics import metrics
from dataguard.security.rate_limit import InMemoryRateLimiter, RedisRateLimiter


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, redis: Redis | None = None) -> None:
        super().__init__(app)
        settings = get_settings()
        self._limit = settings.rate_limit_per_minute
        self._redis_limiter = RedisRateLimiter(redis) if redis is not None else None
        self._memory_limiter = InMemoryRateLimiter()

    @staticmethod
    def _request_id(request) -> str:
        candidate = request.headers.get("X-Request-ID", "")
        if candidate and len(candidate) <= 128 and all(ord(char) >= 32 for char in candidate):
            return candidate
        return str(uuid4())

    @staticmethod
    def _client_context(request) -> tuple[str, str]:
        ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("User-Agent", "")[:255]
        return ip[:64], user_agent

    async def dispatch(
        self, request, call_next: Callable[[object], Awaitable[Response]]
    ) -> Response:
        ip_address, client = self._client_context(request)
        token = set_context(
            AuditRequestContext(
                request_id=self._request_id(request),
                ip_address=ip_address,
                client=client or None,
            )
        )
        metrics.inc("dataguard_requests_total", method=request.method, path=request.url.path)
        try:
            if request.url.path in {"/health/live", "/health/ready"}:
                return await call_next(request)
            key = f"ratelimit:{ip_address}:{request.url.path}"
            redis = getattr(request.app.state, "redis", None)
            limiter = self._redis_limiter
            if limiter is None and redis is not None:
                limiter = RedisRateLimiter(redis)
            try:
                if limiter is not None:
                    allowed = await limiter.allow(key, self._limit)
                else:
                    settings = get_settings()
                    if settings.environment == "production":
                        metrics.inc("dataguard_rate_limit_errors_total", reason="unavailable")
                        return JSONResponse(
                            {"detail": "Rate limiting service unavailable"}, status_code=503
                        )
                    allowed = await self._memory_limiter.allow(key, self._limit)
            except Exception:
                metrics.inc("dataguard_rate_limit_errors_total", reason="backend_error")
                if get_settings().environment == "production":
                    return JSONResponse(
                        {"detail": "Rate limiting service unavailable"}, status_code=503
                    )
                allowed = await self._memory_limiter.allow(key, self._limit)
            if not allowed:
                metrics.inc("dataguard_rate_limited_total", path=request.url.path)
                return JSONResponse(
                    {"detail": "Rate limit exceeded"},
                    status_code=429,
                    headers={"Retry-After": "60"},
                )
            response = await call_next(request)
            metrics.inc(
                "dataguard_response_total",
                method=request.method,
                path=request.url.path,
                status=str(response.status_code),
            )
            return response
        except UnsafeDocumentError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        except MalwareScannerUnavailableError as exc:
            metrics.inc("dataguard_security_events_total", event="malware_scanner_unavailable")
            return JSONResponse({"detail": str(exc)}, status_code=503)
        finally:
            reset_context(token)
