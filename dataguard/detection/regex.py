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
    group: int = 0


_H = r"[ \t]"
_NAME = r"[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'’-]+"
_STREET_TYPES = (
    r"(?:street|st\.?|avenue|ave\.?|road|rd\.?|boulevard|blvd\.?|drive|dr\.?|lane|ln\.?|"
    r"route|chemin|rue|av\.?|boul\.?|montée|place|pl\.?|rang)"
)
_CITY = r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ .'-]{1,60}"


class RegexPIIDetector(DetectionEngine):
    name = "regex"

    _rules = (
        PatternRule(
            PIIType.EMAIL,
            re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
            0.98,
        ),
        PatternRule(
            PIIType.PHONE,
            re.compile(
                r"(?<!\d)(?:\+?1[ .-]?)?(?:\(?[2-9]\d{2}\)?[ .-]?)[2-9]\d{2}[ .-]?\d{4}(?!\d)"
            ),
            0.90,
        ),
        PatternRule(
            PIIType.IP_ADDRESS,
            re.compile(
                r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
                r"(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
            ),
            0.97,
        ),
        PatternRule(
            PIIType.CREDIT_CARD,
            re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
            0.75,
        ),
        PatternRule(
            PIIType.SOCIAL_INSURANCE_NUMBER,
            re.compile(r"(?<!\d)\d{3}[ -]?\d{3}[ -]?\d{3}(?!\d)"),
            0.72,
        ),
        PatternRule(
            PIIType.HEALTH_INSURANCE_ID,
            re.compile(r"(?<![A-Z0-9])[A-Z]{4}[ -]?\d{8}(?![A-Z0-9])", re.I),
            0.88,
        ),
        PatternRule(
            PIIType.PASSPORT,
            re.compile(r"\b[A-Z]{1,2}\d{6,8}\b"),
            0.55,
        ),
        PatternRule(
            PIIType.DATE_OF_BIRTH,
            re.compile(
                r"\b(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.]"
                r"(?:0[1-9]|[12]\d|3[01])\b"
            ),
            0.82,
        ),
        PatternRule(
            PIIType.PERSON,
            re.compile(
                rf"(?i)(?:nom(?:{_H}+complet)?|prénom|name){_H}*[:\-]{_H}*"
                rf"({_NAME}(?:{_H}+{_NAME}){{1,3}})"
            ),
            0.93,
            1,
        ),
        PatternRule(
            PIIType.ADDRESS,
            re.compile(
                rf"(?i)(?:adresse|address|domicile){_H}*[:\-]{_H}*"
                rf"(\d{{1,6}}{_H}+[^\n,]+(?:,{_H}*[A-Za-zÀ-ÖØ-öø-ÿ .'-]+)?)"
            ),
            0.91,
            1,
        ),
        PatternRule(
            PIIType.HEALTH_INFORMATION,
            re.compile(
                r"(?i)(?:diagnostic|diagnosis|condition médicale|medical condition|"
                r"dossier médical|medical record)\s*[:\-]\s*([^\n]{2,120})"
            ),
            0.90,
            1,
        ),
    )

    _contextual_rules = (
        PatternRule(
            PIIType.PERSON,
            re.compile(
                rf"\b(?:M(?:me|me\.)|Mme|M(?:r|r\.)|Mr|Mrs|Ms|Dr|Docteure?|"
                rf"Monsieur|Madame){_H}+({_NAME}(?:{_H}+{_NAME}){{1,2}})\b"
            ),
            0.86,
            1,
        ),
        PatternRule(
            PIIType.PERSON,
            re.compile(
                rf"(?i)\b(?:first name|last name|given name|family name|"
                rf"nom de famille|prénom)(?:{_H}+(?:is|est){_H}+|{_H}*:{_H}*)?"
                rf"({_NAME}(?:{_H}+{_NAME}){{0,2}})\b"
            ),
            0.88,
            1,
        ),
        PatternRule(
            PIIType.ADDRESS,
            re.compile(
                rf"\b(\d{{1,6}}{_H}+(?:{_STREET_TYPES}{_H}+[A-Za-zÀ-ÖØ-öø-ÿ0-9'’.-]+"
                rf"(?:{_H}+[A-Za-zÀ-ÖØ-öø-ÿ0-9'’.-]+){{0,3}}|"
                rf"[A-Za-zÀ-ÖØ-öø-ÿ0-9'’.-]+{_H}+{_STREET_TYPES}"
                rf"(?:{_H}+[A-Za-zÀ-ÖØ-öø-ÿ0-9'’.-]+){{0,4}})"
                rf"(?:,{_H}*{_CITY})?)\b",
                re.I,
            ),
            0.87,
            1,
        ),
        PatternRule(
            PIIType.ADDRESS,
            re.compile(
                rf"\b(\d{{1,6}}{_H}+[^\n,]{{3,80}},{_H}*"
                rf"[A-Za-zÀ-ÖØ-öø-ÿ .'-]{{2,60}}{_H}+[A-Z]{{2}}{_H}+"
                rf"\d[A-Z]\d[ -]?\d[A-Z]\d)\b",
                re.I,
            ),
            0.94,
            1,
        ),
    )

    def detect(self, text: str) -> list[Detection]:
        if not text:
            return []
        detections: list[Detection] = []
        for rule in (*self._rules, *self._contextual_rules):
            for match in rule.pattern.finditer(text):
                value = match.group(rule.group)
                start, end = match.span(rule.group)
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
                detections.append(
                    Detection(rule.pii_type, start, end, confidence, self.name, value)
                )
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
    ordered = sorted(
        detections, key=lambda item: (item.start, -(item.end - item.start), -item.confidence)
    )
    result: list[Detection] = []
    for detection in ordered:
        if any(
            detection.start >= existing.start and detection.end <= existing.end
            for existing in result
        ):
            continue
        result.append(detection)
    return result
