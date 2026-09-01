from __future__ import annotations

from dataclasses import dataclass

from dataguard.detection.regex import RegexPIIDetector


@dataclass(frozen=True)
class Case:
    text: str
    expected: frozenset[str]


# Synthetic-only benchmark. Values are deliberately fictional and must never be
# replaced with production personal information.
CASES = (
    Case("Nom: Alice Tremblay\nEmail: alice@example.com", frozenset({"person", "email"})),
    Case("Téléphone: +1 514-555-0199", frozenset({"phone"})),
    Case("NAS: 123-456-789", frozenset({"social_insurance_number"})),
    Case("RAMQ: ABCE 12345678", frozenset({"health_insurance_id"})),
    Case("Adresse: 123 Rue Exemple, Montréal", frozenset({"address"})),
    Case("IP: 192.0.2.10", frozenset({"ip_address"})),
    Case("Diagnostic: condition fictive", frozenset({"health_information"})),
    Case("Date de naissance: 1985-04-12", frozenset({"date_of_birth"})),
    Case("Réunion à Montréal demain à 10 h.", frozenset()),
    Case("Le dossier contient trois lignes sans identifiant.", frozenset()),
)


def evaluate(cases=CASES) -> dict[str, object]:
    detector = RegexPIIDetector()
    labels = sorted({label for case in cases for label in case.expected} | {d.pii_type.value for case in cases for d in detector.detect(case.text)})
    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = fp = fn = 0
        for case in cases:
            predicted = {d.pii_type.value for d in detector.detect(case.text)}
            tp += int(label in predicted and label in case.expected)
            fp += int(label in predicted and label not in case.expected)
            fn += int(label not in predicted and label in case.expected)
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    macro_f1 = sum(item["f1"] for item in per_class.values()) / len(per_class) if per_class else 0.0
    return {"cases": len(cases), "macro_f1": macro_f1, "per_class": per_class}


if __name__ == "__main__":
    import json
    print(json.dumps(evaluate(), indent=2, sort_keys=True))
