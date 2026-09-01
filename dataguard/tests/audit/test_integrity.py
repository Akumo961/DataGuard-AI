from dataclasses import replace

from dataguard.audit import AuditService
from dataguard.audit.integrity import verify_chain


def _record(event_id: str, previous_hash: str = ""):
    return AuditService().create_record(
        event_id=event_id,
        timestamp=f"2026-09-01T00:00:0{event_id[-1]}Z",
        user_id="u1",
        organization_id="org1",
        action="ANALYSIS_COMPLETED",
        object_type="analysis",
        object_id=event_id,
        previous_state=None,
        new_state={"status": "COMPLETED"},
        ip_address=None,
        request_id=None,
        result="success",
        previous_hash=previous_hash,
    )


def test_hash_chain_detects_middle_record_tampering() -> None:
    first = _record("evt-1")
    second = _record("evt-2", first.integrity_hash)
    third = _record("evt-3", second.integrity_hash)
    assert verify_chain([first, second, third])
    assert not verify_chain([first, replace(second, result="failed"), third])


def test_hash_chain_detects_reordering() -> None:
    first = _record("evt-1")
    second = _record("evt-2", first.integrity_hash)
    assert not verify_chain([second, first])
