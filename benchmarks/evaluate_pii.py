from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataguard.detection.regex import RegexPIIDetector


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate DataGuard regex PII detector on a fixed corpus")
    parser.add_argument("--corpus", default="benchmarks/pii_corpus.jsonl")
    args = parser.parse_args()

    detector = RegexPIIDetector()
    tp = fp = fn = 0
    cases = 0
    for line in Path(args.corpus).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        expected = set(case["expected"])
        predicted = {item.pii_type.value for item in detector.detect(case["text"])}
        tp += len(expected & predicted)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
        cases += 1

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(
        json.dumps(
            {
                "cases": cases,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "corpus": str(args.corpus),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
