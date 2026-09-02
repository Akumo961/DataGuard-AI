from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AuditRequestContext:
    request_id: str | None = None
    ip_address: str | None = None
    client: str | None = None
    classification_policy: dict[str, Any] | None = None


_current: ContextVar[AuditRequestContext] = ContextVar(
    "dataguard_audit_context", default=AuditRequestContext()
)


def set_context(context: AuditRequestContext):
    return _current.set(context)


def reset_context(token) -> None:
    _current.reset(token)


def get_context() -> AuditRequestContext:
    return _current.get()


def set_classification_policy(policy: dict[str, Any] | None) -> None:
    current = _current.get()
    _current.set(
        AuditRequestContext(
            request_id=current.request_id,
            ip_address=current.ip_address,
            client=current.client,
            classification_policy=policy,
        )
    )


def get_classification_policy() -> dict[str, Any] | None:
    return _current.get().classification_policy
