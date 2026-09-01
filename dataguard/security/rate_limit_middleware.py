"""Rate-limit middleware with Redis-backed production enforcement."""

from collections.abc import Awaitable, Callable

from redis.asyncio import Redis
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from dataguard.core.config import get_settings
from dataguard.security.rate_limit import InMemoryRateLimiter, RedisRateLimiter


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, redis: Redis | None = None) -> None:
        super().__init__(app)
        settings = get_settings()
        self._limit = settings.rate_limit_per_minute
        self._redis_limiter = RedisRateLimiter(redis) if redis is not None else None
        self._memory_limiter = InMemoryRateLimiter()

    async def dispatch(self, request, call_next: Callable[[object], Awaitable[Response]]) -> Response:
        if request.url.path in {"/health/live", "/health/ready"}:
            return await call_next(request)
        client = request.client.host if request.client else "unknown"
        key = f"ratelimit:{client}:{request.url.path}"
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
                    return JSONResponse({"detail": "Rate limiting service unavailable"}, status_code=503)
                allowed = await self._memory_limiter.allow(key, self._limit)
        except Exception:
            if get_settings().environment == "production":
                return JSONResponse({"detail": "Rate limiting service unavailable"}, status_code=503)
            allowed = await self._memory_limiter.allow(key, self._limit)
        if not allowed:
            return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429, headers={"Retry-After": "60"})
        return await call_next(request)
