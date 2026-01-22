from __future__ import annotations

import numpy as np


def _safe_div(num: float, den: float) -> float:
    return num / den if den != 0 else 0.0


def roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute ROC AUC using the rank (Mann–Whitney U) formulation with tie handling.
    Returns 0.5 when only one class is present.
    """
    y = labels.astype(int).flatten()
    s = scores.astype(float).flatten()
    if y.size == 0:
        return 0.5

    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]

    ranks = np.arange(1, len(s_sorted) + 1, dtype=np.float64)
    i = 0
    while i < len(s_sorted):
        j = i + 1
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        if j - i > 1:
            ranks[i:j] = (ranks[i] + ranks[j - 1]) / 2.0
        i = j

    rank_sum_pos = ranks[y_sorted == 1].sum()
    auc = (rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def compute_binary_metrics(preds: np.ndarray, labels: np.ndarray, *, threshold: float = 0.5) -> dict[str, float]:
    y = labels.astype(int).flatten()
    y_hat = (preds.flatten() >= threshold).astype(int)

    tp = int(((y_hat == 1) & (y == 1)).sum())
    tn = int(((y_hat == 0) & (y == 0)).sum())
    fp = int(((y_hat == 1) & (y == 0)).sum())
    fn = int(((y_hat == 0) & (y == 1)).sum())

    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall) if (precision + recall) else 0.0
    auc = roc_auc_score(y, preds)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
    }

