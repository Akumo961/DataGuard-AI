import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging


class RiskAssessor:
    """Assess document risk based on PII detection and context"""

    def __init__(self, compliance_rules: Dict[str, List[str]]):
        self.compliance_rules = compliance_rules
        self.logger = logging.getLogger(__name__)

        # Risk weights for different PII types
        self.pii_weights = {
            'ssn': 1.0, 'credit_card': 0.9, 'passport': 0.8,
            'medical_record_number': 0.9, 'health_plan_id': 0.7,
            'email': 0.3, 'phone': 0.4, 'address': 0.5,
            'name': 0.6, 'ip_address': 0.4
        }

        # Context modifiers
        self.context_modifiers = {
            'public_document': 0.5,
            'internal_memo': 0.8,
            'customer_data': 1.0,
            'employee_data': 0.9,
            'financial_data': 1.0,
            'medical_data': 1.2,
            'general': 0.7
        }

        # File type modifiers
        self.file_type_modifiers = {
            'txt': 1.0, 'pdf': 1.0, 'docx': 1.0,
            'csv': 1.2, 'xlsx': 1.2,  # Structured data higher risk
            'png': 0.8, 'jpg': 0.8, 'jpeg': 0.8,  # Images lower risk
            'wav': 0.6, 'mp3': 0.6, 'm4a': 0.6  # Audio lowest risk
        }

    def assess_document_risk(self, detected_pii: Dict[str, List[str]],
                             document_context: str = "general",
                             file_type: str = "txt") -> Dict[str, Any]:
        """Assess overall document risk"""

        # Calculate base risk from PII
        base_risk = self._calculate_base_risk(detected_pii)

        # Apply context modifier
        context_mod = self.context_modifiers.get(document_context, 1.0)

        # Apply file type modifier
        file_mod = self.file_type_modifiers.get(file_type.lower(), 1.0)

        # Calculate final risk score
        final_risk_score = base_risk * context_mod * file_mod
        final_risk_score = min(final_risk_score, 1.0)  # Cap at 1.0

        # Determine risk level
        risk_level = self._classify_risk_level(final_risk_score)

        # Compliance framework violations
        violations = self._check_compliance_violations(detected_pii)

        return {
            'base_risk_score': base_risk,
            'context_modifier': context_mod,
            'file_type_modifier': file_mod,
            'final_risk_score': final_risk_score,
            'risk_level': risk_level,
            'compliance_violations': violations,
            'pii_breakdown': self._get_pii_breakdown(detected_pii)
        }

    def _calculate_base_risk(self, detected_pii: Dict[str, List[str]]) -> float:
        """Calculate base risk score from detected PII"""
        total_risk = 0.0

        for pii_type, items in detected_pii.items():
            weight = self.pii_weights.get(pii_type, 0.2)
            count = len(items)
            # Logarithmic scaling to prevent domination by single PII type
            risk_contribution = weight * min(count, 10) * 0.1
            total_risk += risk_contribution

        return min(total_risk, 1.0)

    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify risk into levels"""
        if risk_score >= 0.8:
            return "CRITICAL"
        elif risk_score >= 0.6:
            return "HIGH"
        elif risk_score >= 0.3:
            return "MEDIUM"
        else:
            return "LOW"

    def _check_compliance_violations(self, detected_pii: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Check which compliance frameworks are violated"""
        violations = {}

        for framework, pii_types in self.compliance_rules.items():
            framework_violations = []
            for pii_type in pii_types:
                if pii_type in detected_pii and detected_pii[pii_type]:
                    framework_violations.append(pii_type)

            if framework_violations:
                violations[framework] = framework_violations

        return violations

    def _get_pii_breakdown(self, detected_pii: Dict[str, List[str]]) -> Dict[str, Any]:
        """Get detailed PII breakdown"""
        breakdown = {}
        total_items = 0

        for pii_type, items in detected_pii.items():
            count = len(items)
            weight = self.pii_weights.get(pii_type, 0.2)
            risk_contribution = weight * count

            breakdown[pii_type] = {
                'count': count,
                'weight': weight,
                'risk_contribution': risk_contribution
            }
            total_items += count

        breakdown['total_items'] = total_items
        return breakdown