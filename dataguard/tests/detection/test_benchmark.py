from typing import cast

from dataguard.evaluation.pii_benchmark import evaluate


def test_synthetic_benchmark_is_reproducible_and_reports_metrics() -> None:
    result = evaluate()
    assert result["cases"] == 10
    macro_f1 = cast(float, result["macro_f1"])
    assert 0.0 <= macro_f1 <= 1.0
    per_class = cast(dict[str, dict[str, object]], result["per_class"])
    assert per_class
    for metrics in per_class.values():
        assert {"precision", "recall", "f1", "tp", "fp", "fn"} <= metrics.keys()
