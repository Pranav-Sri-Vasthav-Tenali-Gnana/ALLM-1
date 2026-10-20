import pytest

from core.eval.metrics import classification_metrics, primary_metric, rouge_metrics


def test_classification_metrics():
    m = classification_metrics([0, 1, 1, 0], [0, 1, 0, 0])
    assert m["accuracy"] == pytest.approx(0.75)
    assert 0 < m["macro_f1"] <= 1


def test_rouge_identical_is_100():
    m = rouge_metrics(["The cat sat on the mat."], ["The cat sat on the mat."])
    assert m["rouge1"] == pytest.approx(100.0)
    assert m["rougeL"] == pytest.approx(100.0)


def test_rouge_disjoint_is_0():
    assert rouge_metrics(["apples"], ["zebra"])["rouge1"] == 0.0


def test_primary_metric():
    assert primary_metric("classification") == "accuracy"
    assert primary_metric("summarization") == "rougeL"
