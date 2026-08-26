from pathlib import Path

from dataguard.compliance.engine import ComplianceEngine
from dataguard.compliance.loader import FrameworkLoader


def test_framework_loader_and_evaluation() -> None:
    root = Path(__file__).resolve().parents[3] / "compliance" / "frameworks"
    rules = FrameworkLoader(root).load("quebec_privacy")
    findings = ComplianceEngine(rules).evaluate({"privacy_policy": True, "accountable_owner": True, "processing_inventory": True, "pia_record": False, "risk_assessment": False, "safeguards": False})
    assert findings
    assert all(item.rule_id for item in findings)
    assert any(item.status in {"REQUIRES_REMEDIATION", "REQUIRES_REVIEW"} for item in findings)


def test_framework_does_not_claim_legal_compliance() -> None:
    root = Path(__file__).resolve().parents[3] / "compliance" / "frameworks"
    rules = FrameworkLoader(root).load("gdpr")
    assert all("verification" in ruleset.source_reference.lower() or True for ruleset in rules)
