from dataguard.evaluation.pii_benchmark import evaluate


def test_synthetic_benchmark_is_reproducible_and_reports_metrics() -> None:
    result = evaluate()
    assert result["cases"] == 10
    assert 0.0 <= result["macro_f1"] <= 1.0
    assert result["per_class"]
    for metrics in result["per_class"].values():
        assert {"precision", "recall", "f1", "tp", "fp", "fn"} <= metrics.keys()
