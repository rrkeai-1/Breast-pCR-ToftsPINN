"""
Metrics utilities for binary pCR prediction.

All metrics here operate on the standard ``y_true`` (0/1 labels) and
``y_prob`` (predicted positive-class probabilities) pair, plus a chosen
decision threshold for the hard-label metrics.

This module is intentionally simple. Bootstrap confidence intervals are
deliberately omitted from the public release; if a reviewer requests them,
they can be added later without altering the surface.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        v = float(value)
        if not np.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _coerce_arrays(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(pd.to_numeric(pd.Series(y_true), errors="coerce"), dtype=float)
    y_prob = np.asarray(pd.to_numeric(pd.Series(y_prob), errors="coerce"), dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"y_true and y_prob have incompatible shapes: {y_true.shape} vs {y_prob.shape}"
        )
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    return y_true[mask], y_prob[mask]


def compute_binary_metrics(
    y_true,
    y_prob,
    decision_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Compute the standard panel of binary-classification metrics used in this
    project: AUROC, AUPRC (a.k.a. average precision), accuracy, balanced
    accuracy, precision (PPV), recall (sensitivity), specificity, F1, log loss,
    plus the confusion-matrix counts and basic sample sizes.

    NaNs and inf values in either input are dropped before scoring.

    Returns a JSON-serialisable dict.
    """
    y_true_clean, y_prob_clean = _coerce_arrays(y_true, y_prob)
    threshold = float(decision_threshold)

    n_total = int(y_true_clean.size)
    n_positive = int((y_true_clean > 0.5).sum()) if n_total > 0 else 0
    n_negative = int(n_total - n_positive)

    out: dict[str, Any] = {
        "n_samples": n_total,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "decision_threshold": threshold,
        "roc_auc": float("nan"),
        "average_precision": float("nan"),
        "accuracy": float("nan"),
        "balanced_accuracy": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "specificity": float("nan"),
        "f1": float("nan"),
        "log_loss": float("nan"),
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 0,
    }

    if n_total == 0 or n_positive == 0 or n_negative == 0:
        return out

    y_int = (y_true_clean > 0.5).astype(int)
    y_pred = (y_prob_clean >= threshold).astype(int)

    try:
        out["roc_auc"] = _safe_float(roc_auc_score(y_int, y_prob_clean))
    except Exception:
        pass
    try:
        out["average_precision"] = _safe_float(average_precision_score(y_int, y_prob_clean))
    except Exception:
        pass
    try:
        out["accuracy"] = _safe_float(accuracy_score(y_int, y_pred))
        out["balanced_accuracy"] = _safe_float(balanced_accuracy_score(y_int, y_pred))
        out["precision"] = _safe_float(precision_score(y_int, y_pred, zero_division=0))
        out["recall"] = _safe_float(recall_score(y_int, y_pred, zero_division=0))
        out["f1"] = _safe_float(f1_score(y_int, y_pred, zero_division=0))
    except Exception:
        pass

    try:
        cm = confusion_matrix(y_int, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        out["true_negative"] = int(tn)
        out["false_positive"] = int(fp)
        out["false_negative"] = int(fn)
        out["true_positive"] = int(tp)
        denom_spec = tn + fp
        out["specificity"] = _safe_float(tn / denom_spec) if denom_spec > 0 else float("nan")
    except Exception:
        pass

    try:
        # log_loss requires both classes present in y_true; we already checked.
        clipped = np.clip(y_prob_clean, 1e-7, 1.0 - 1e-7)
        out["log_loss"] = _safe_float(log_loss(y_int, clipped, labels=[0, 1]))
    except Exception:
        pass

    return out


def select_best_threshold(
    y_true,
    y_prob,
    metric: str = "f1",
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_step: float = 0.01,
) -> dict[str, Any]:
    """
    Sweep a decision threshold over [threshold_min, threshold_max] in steps of
    ``threshold_step`` and select the one that maximises ``metric`` on the
    supplied (y_true, y_prob) pair.

    ``metric`` must be one of: ``"f1"``, ``"balanced_accuracy"``.

    Returns a dict with the chosen threshold, the chosen metric value, and the
    full panel of metrics at that threshold.
    """
    metric = str(metric).strip().lower()
    if metric not in {"f1", "balanced_accuracy"}:
        raise ValueError(f"Unsupported threshold-selection metric: {metric!r}")

    if not (threshold_min < threshold_max):
        raise ValueError("threshold_min must be < threshold_max")
    if threshold_step <= 0:
        raise ValueError("threshold_step must be > 0")

    grid = np.arange(threshold_min, threshold_max + 0.5 * threshold_step, threshold_step)
    grid = np.clip(grid, 1e-3, 1.0 - 1e-3)

    best_score = -np.inf
    best_threshold = 0.5
    best_metrics: dict[str, Any] | None = None

    for t in grid:
        m = compute_binary_metrics(y_true, y_prob, decision_threshold=float(t))
        score = m.get(metric, float("nan"))
        if not np.isfinite(score):
            continue
        if score > best_score:
            best_score = float(score)
            best_threshold = float(t)
            best_metrics = m

    if best_metrics is None:
        # Fallback: nothing scored finite; return metrics at 0.5 anyway.
        best_metrics = compute_binary_metrics(y_true, y_prob, decision_threshold=0.5)
        best_threshold = 0.5

    return {
        "threshold_metric": metric,
        "decision_threshold": float(best_threshold),
        "best_metric_value": float(best_metrics.get(metric, float("nan"))),
        "metrics_at_threshold": best_metrics,
    }


__all__ = [
    "compute_binary_metrics",
    "select_best_threshold",
]
