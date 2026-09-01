from __future__ import annotations

import argparse
from collections import defaultdict
import json
from dataclasses import dataclass

from dataguard.detection.ner import NERDetector


@dataclass(frozen=True)
class Case:
    text: str
    expected: tuple[tuple[str, str], ...]


# Synthetic-only Québec/English corpus. Every value is fictional.
CASES = (
    Case(
        "Jean Tremblay travaille à Montréal, au 123 Rue Exemple.",
        (("PERSON", "Jean Tremblay"), ("LOCATION", "Montréal")),
    ),
    Case(
        "Marie-Claude Gagnon vit près de Québec au 456 avenue Fictive.",
        (("PERSON", "Marie-Claude Gagnon"), ("LOCATION", "Québec")),
    ),
    Case(
        "François Bouchard prépare le dossier pour Gatineau.",
        (("PERSON", "François Bouchard"), ("LOCATION", "Gatineau")),
    ),
    Case(
        "Émilie Roy sent the report to Sarah Johnson in Montréal.",
        (("PERSON", "Émilie Roy"), ("PERSON", "Sarah Johnson"), ("LOCATION", "Montréal")),
    ),
    Case(
        "John Smith reviewed the fictional case in Toronto.",
        (("PERSON", "John Smith"), ("LOCATION", "Toronto")),
    ),
    Case(
        "Le dossier contient 789 Boulevard Fictif et aucun nom de personne.",
        (),
    ),
)


def _spans(text: str, detector: NERDetector) -> set[tuple[str, str]]:
    return {(item.pii_type.value, text[item.start : item.end]) for item in detector.detect(text)}


def evaluate(model_name: str = "xx_ent_wiki_sm") -> dict[str, object]:
    detector = NERDetector.from_spacy(model_name)
    labels = sorted({label for case in CASES for label, _ in case.expected})
    per_class: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = fp = fn = 0
        for case in CASES:
            expected = set(case.expected)
            predicted = _spans(case.text, detector)
            tp += int(bool({item for item in predicted if item[0] == label} & expected))
            fp += len({item for item in predicted if item[0] == label} - expected)
            fn += len({item for item in expected if item[0] == label} - predicted)
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    macro_f1 = (
        sum(float(item["f1"]) for item in per_class.values()) / len(per_class)
        if per_class
        else 0.0
    )
    return {
        "model": model_name,
        "cases": len(CASES),
        "synthetic_only": True,
        "per_class": per_class,
        "macro_f1": macro_f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the configured multilingual NER model")
    parser.add_argument("--model", default="xx_ent_wiki_sm")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.model), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
