from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class AuditRequestContext:
    request_id: str | None = None
    ip_address: str | None = None
    client: str | None = None


_current: ContextVar[AuditRequestContext] = ContextVar(
    "dataguard_audit_context", default=AuditRequestContext()
)


def set_context(context: AuditRequestContext):
    return _current.set(context)


def reset_context(token) -> None:
    _current.reset(token)


def get_context() -> AuditRequestContext:
    return _current.get()
