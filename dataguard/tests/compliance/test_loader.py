from pathlib import Path

import pytest

from dataguard.compliance import FrameworkLoader


@pytest.mark.parametrize("framework", ["quebec_privacy", "canada_privacy", "gdpr", "ccpa"])
def test_framework_schema(framework: str) -> None:
    root = Path(__file__).resolve().parents[3] / "compliance" / "frameworks"
    rules = FrameworkLoader(root).load(framework)
    assert rules
    for rule in rules:
        assert rule.rule_id and rule.title and rule.version
        assert rule.source_reference
