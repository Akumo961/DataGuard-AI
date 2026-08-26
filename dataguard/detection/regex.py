from __future__ import annotations

import re
from dataclasses import dataclass

from dataguard.detection.base import DetectionEngine
from dataguard.domain.models import Detection, PIIType


@dataclass(frozen=True)
class PatternRule:
    pii_type: PIIType
    pattern: re.Pattern[str]
    confidence: float


class RegexPIIDetector(DetectionEngine):
    name = "regex"

    _rules = (
        PatternRule(PIIType.EMAIL, re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), 0.98),
        PatternRule(PIIType.PHONE, re.compile(r"(?<!\d)(?:\+?1[ .-]?)?(?:\(?[2-9]\d{2}\)?[ .-]?)[2-9]\d{2}[ .-]?\d{4}(?!\d)"), 0.90),
        PatternRule(PIIType.IP_ADDRESS, re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), 0.97),
        PatternRule(PIIType.CREDIT_CARD, re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"), 0.75),
        PatternRule(PIIType.SOCIAL_INSURANCE_NUMBER, re.compile(r"(?<!\d)\d{3}[ -]?\d{3}[ -]?\d{3}(?!\d)"), 0.72),
        PatternRule(PIIType.PASSPORT, re.compile(r"\b[A-Z]{1,2}\d{6,8}\b"), 0.55),
        PatternRule(PIIType.DATE_OF_BIRTH, re.compile(r"\b(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])\b"), 0.82),
    )

    def detect(self, text: str) -> list[Detection]:
        if not text:
            return []
        detections: list[Detection] = []
        for rule in self._rules:
            for match in rule.pattern.finditer(text):
                value = match.group(0)
                confidence = rule.confidence
                if rule.pii_type is PIIType.CREDIT_CARD:
                    digits = re.sub(r"\D", "", value)
                    confidence = 0.99 if self._luhn(digits) else 0.20
                    if confidence < 0.5:
                        continue
                if rule.pii_type is PIIType.SOCIAL_INSURANCE_NUMBER:
                    digits = re.sub(r"\D", "", value)
                    if digits[:3] == "000" or digits[0] in "89":
                        continue
                detections.append(Detection(rule.pii_type, match.start(), match.end(), confidence, self.name, value))
        return _deduplicate(detections)

    @staticmethod
    def _luhn(number: str) -> bool:
        if not 13 <= len(number) <= 19:
            return False
        total = 0
        parity = len(number) % 2
        for index, char in enumerate(number):
            digit = int(char)
            if index % 2 == parity:
                digit *= 2
                if digit > 9:
                    digit -= 9
            total += digit
        return total % 10 == 0


def _deduplicate(detections: list[Detection]) -> list[Detection]:
    ordered = sorted(detections, key=lambda item: (item.start, -(item.end - item.start), -item.confidence))
    result: list[Detection] = []
    for detection in ordered:
        if any(detection.start >= existing.start and detection.end <= existing.end for existing in result):
            continue
        result.append(detection)
    return result
