from dataclasses import replace

from dataguard.audit import AuditService


def test_audit_record_integrity_verifies() -> None:
    service = AuditService()
    record = service.create_record(
        event_id="evt-1", timestamp="2026-08-26T00:00:00Z", user_id="u1", organization_id="org1",
        action="PIA_UPDATED", object_type="pia", object_id="pia1", previous_state={"status": "DRAFT"},
        new_state={"status": "IN_REVIEW"}, ip_address="127.0.0.1", request_id="req1", result="success",
    )
    assert service.verify(record)
    assert not service.verify(record, previous_hash="tampered")


def test_audit_mutation_is_detected() -> None:
    service = AuditService()
    record = service.create_record(
        event_id="evt-2", timestamp="2026-08-26T00:00:00Z", user_id="u1", organization_id="org1",
        action="PIA_UPDATED", object_type="pia", object_id="pia1", previous_state=None,
        new_state={"status": "APPROVED"}, ip_address=None, request_id="req2", result="success",
    )
    tampered = replace(record, result="failed")
    assert not service.verify(tampered)
