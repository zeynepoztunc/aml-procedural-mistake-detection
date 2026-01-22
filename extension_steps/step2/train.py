from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn


def _safe_div(num: float, den: float) -> float:
    return num / den if den != 0 else 0.0


def _roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
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
    # Average ranks for ties.
    i = 0
    while i < len(s_sorted):
        j = i + 1
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        if j - i > 1:
            avg_rank = (ranks[i] + ranks[j - 1]) / 2.0
            ranks[i:j] = avg_rank
        i = j

    rank_sum_pos = ranks[y_sorted == 1].sum()
    auc = (rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    criterion,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.train()
    total_loss = 0.0
    all_preds: list[float] = []
    all_labels: list[float] = []

    for batch in dataloader:
        embeddings = batch["embeddings"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(embeddings, mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        preds = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.detach().cpu().numpy().flatten().tolist())

    return total_loss / max(1, len(dataloader)), np.asarray(all_preds), np.asarray(all_labels)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds: list[float] = []
    all_labels: list[float] = []

    for batch in dataloader:
        embeddings = batch["embeddings"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(embeddings, mask)
        loss = criterion(logits, labels)
        total_loss += float(loss.item())

        preds = torch.sigmoid(logits).cpu().numpy().flatten()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().flatten().tolist())

    return total_loss / max(1, len(dataloader)), np.asarray(all_preds), np.asarray(all_labels)


def compute_metrics(preds: np.ndarray, labels: np.ndarray, *, threshold: float = 0.5) -> dict[str, float]:
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
    auc = _roc_auc_score(y, preds)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
    }
