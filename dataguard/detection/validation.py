from __future__ import annotations

import re

from dataguard.domain.models import Detection, PIIType


class DetectionValidator:
    """Reduce obvious false positives without making legal identity claims."""

    def validate(self, text: str, detections: list[Detection]) -> list[Detection]:
        result: list[Detection] = []
        for detection in detections:
            value = detection.value or text[detection.start : detection.end]
            if detection.pii_type is PIIType.EMAIL and "@" not in value:
                continue
            if detection.pii_type is PIIType.PHONE and len(re.sub(r"\D", "", value)) < 10:
                continue
            if detection.pii_type is PIIType.IP_ADDRESS:
                octets = value.split(".")
                if len(octets) != 4 or any(int(octet) > 255 for octet in octets):
                    continue
            result.append(detection)
        return result
