from __future__ import annotations

import time

from prometheus_client import Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

REQUESTS = Counter(
    "dataguard_http_requests_total",
    "HTTP requests handled by DataGuard",
    ("method", "route", "status"),
)
LATENCY = Histogram(
    "dataguard_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ("method", "route"),
)


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        route_path = getattr(route, "path", "unmatched")
        elapsed = time.perf_counter() - started
        REQUESTS.labels(request.method, route_path, str(response.status_code)).inc()
        LATENCY.labels(request.method, route_path).observe(elapsed)
        return response


def metrics_response() -> Response:
    return Response(generate_latest(), media_type="text/plain; version=0.0.4")
