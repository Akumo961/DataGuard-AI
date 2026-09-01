from __future__ import annotations

from typing import Any

from dataguard.detection.base import DetectionEngine
from dataguard.domain.models import Detection, PIIType


class NERDetector(DetectionEngine):
    """Adapter boundary for a separately evaluated multilingual NER model."""

    name = "ner"

    def __init__(self, model: Any, label_map: dict[str, PIIType]) -> None:
        self.model = model
        self.label_map = label_map

    @classmethod
    def from_spacy(cls, model_name: str = "xx_ent_wiki_sm") -> NERDetector:
        """Load the pinned lightweight multilingual spaCy NER model lazily."""
        try:
            import spacy
        except ImportError as exc:
            raise RuntimeError("spaCy NER support is not installed") from exc
        model = spacy.load(model_name)
        return cls(
            model,
            {
                "PER": PIIType.PERSON,
                "PERSON": PIIType.PERSON,
                "ORG": PIIType.ORGANIZATION,
                "GPE": PIIType.LOCATION,
                "LOC": PIIType.LOCATION,
            },
        )

    def detect(self, text: str) -> list[Detection]:
        entities = self.model(text)
        result: list[Detection] = []
        for entity in entities:
            label = getattr(entity, "label_", getattr(entity, "label", None))
            pii_type = self.label_map.get(str(label))
            if pii_type is None:
                continue
            start = int(entity.start_char if hasattr(entity, "start_char") else entity.start)
            end = int(entity.end_char if hasattr(entity, "end_char") else entity.end)
            score = float(getattr(entity, "score", 0.5))
            result.append(
                Detection(
                    pii_type,
                    start,
                    end,
                    max(0.0, min(score, 1.0)),
                    self.name,
                    text[start:end],
                    {"model_label": str(label)},
                )
            )
        return result
