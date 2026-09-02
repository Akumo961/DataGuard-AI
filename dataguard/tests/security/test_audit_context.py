from __future__ import annotations

from dataguard.security.audit_context import AuditRequestContext, get_context, reset_context, set_context


def test_audit_context_round_trip() -> None:
    token = set_context(
        AuditRequestContext(
            request_id="req-123",
            ip_address="192.0.2.10",
            client="test-agent",
        )
    )
    try:
        context = get_context()
        assert context.request_id == "req-123"
        assert context.ip_address == "192.0.2.10"
        assert context.client == "test-agent"
    finally:
        reset_context(token)
    assert get_context().request_id is None
