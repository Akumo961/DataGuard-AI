import pytest

from dataguard.pia import PIA, PIAStatus, PIAWorkflow


def test_full_pia_lifecycle() -> None:
    workflow = PIAWorkflow()
    pia = PIA("pia-1", "org-1", "Citizen portal")
    history = []
    for target in (PIAStatus.IN_REVIEW, PIAStatus.REQUIRES_REMEDIATION, PIAStatus.IN_REVIEW, PIAStatus.APPROVED, PIAStatus.ARCHIVED):
        pia, entry = workflow.transition(pia, target, "user-1", "reviewed")
        history.append(entry)
    assert pia.status is PIAStatus.ARCHIVED
    assert pia.version == 6
    assert len(history) == 5


def test_invalid_transition_is_rejected() -> None:
    pia = PIA("pia-1", "org-1", "System")
    with pytest.raises(ValueError):
        PIAWorkflow().transition(pia, PIAStatus.APPROVED, "user-1")
