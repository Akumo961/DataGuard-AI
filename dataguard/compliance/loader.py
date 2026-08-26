from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dataguard.compliance.models import Applicability, ComplianceRule, Severity


class FrameworkLoader:
    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, name: str) -> list[ComplianceRule]:
        path = (self.root / f"{name}.yaml").resolve()
        if path.parent != self.root.resolve() or not path.is_file():
            raise FileNotFoundError(f"framework not found: {name}")
        payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("rules"), list):
            raise ValueError("invalid framework schema")
        framework_version = str(payload.get("version", "unknown"))
        rules: list[ComplianceRule] = []
        for item in payload["rules"]:
            if not isinstance(item, dict):
                raise ValueError("invalid rule entry")
            rules.append(ComplianceRule(
                rule_id=str(item["rule_id"]), title=str(item["title"]), description=str(item["description"]),
                category=str(item["category"]), severity=Severity(str(item["severity"])),
                evidence_required=tuple(str(x) for x in item.get("evidence_required", [])),
                assessment_questions=tuple(str(x) for x in item.get("assessment_questions", [])),
                remediation_recommendations=tuple(str(x) for x in item.get("remediation_recommendations", [])),
                version=framework_version, source_reference=str(item["source_reference"]),
                applicability=Applicability(str(item["applicability"])),
            ))
        return rules
