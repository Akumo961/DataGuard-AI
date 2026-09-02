from __future__ import annotations

from collections import Counter
from threading import Lock


class MetricsRegistry:
    """Small dependency-free Prometheus-compatible counter registry.

    Counters are process-local; production deployments should scrape every replica
    and aggregate at the monitoring layer or replace this registry with OpenTelemetry.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: Counter[tuple[str, tuple[tuple[str, str], ...]]] = Counter()

    def inc(self, name: str, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._counters[key] += 1

    def render(self) -> str:
        with self._lock:
            items = list(self._counters.items())
        lines = [
            "# HELP dataguard_requests_total HTTP requests handled by this process.",
            "# TYPE dataguard_requests_total counter",
        ]
        for (name, labels), value in sorted(items):
            rendered = ",".join(f'{key}="{val.replace(chr(34), chr(92) + chr(34))}"' for key, val in labels)
            metric_name = name
            lines.append(f"{metric_name}{{{rendered}}} {value}" if rendered else f"{metric_name} {value}")
        return "\n".join(lines) + "\n"


metrics = MetricsRegistry()
