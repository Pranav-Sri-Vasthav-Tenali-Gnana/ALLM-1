"""Task metrics. Pure functions -- no model or I/O -- so they're easy to unit test."""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(preds: Sequence[int], labels: Sequence[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def _sentences_per_line(text: str) -> str:
    # rougeLsum scores summary-level LCS over sentences separated by newlines.
    return "\n".join(s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s)


def rouge_metrics(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """Mean F1 of ROUGE-1/2/L/Lsum, scaled to 0-100 like most papers report."""
    from rouge_score import rouge_scorer

    keys = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    scorer = rouge_scorer.RougeScorer(keys, use_stemmer=True)
    totals = {k: [] for k in keys}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(_sentences_per_line(ref), _sentences_per_line(pred))
        for k in keys:
            totals[k].append(scores[k].fmeasure)
    return {k: round(float(np.mean(v)) * 100, 2) if v else 0.0 for k, v in totals.items()}


def primary_metric(task: str) -> str:
    return "accuracy" if task == "classification" else "rougeL"
