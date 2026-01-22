from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import Batch


@dataclass(frozen=True)
class TrainStats:
    train_loss: float
    val_loss: float
    val_precision: float
    val_recall: float
    val_f1: float


def _resize_like(labels: torch.Tensor, masks: torch.Tensor, target_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Pipeline helper: ActionFormer can downsample in time; resize labels/masks to match logits length.
    if labels.size(-1) == target_len:
        return labels, masks
    labels_i = F.interpolate(labels.unsqueeze(1), size=target_len, mode="nearest").squeeze(1)
    masks_i = F.interpolate(masks.unsqueeze(1), size=target_len, mode="nearest").squeeze(1)
    return labels_i, masks_i


def compute_loss(
    *,
    cls_logits: torch.Tensor,  # (B, T)
    labels: torch.Tensor,  # (B, T)
    masks: torch.Tensor,  # (B, T)
    pos_weight: float,
    loss_type: str = "bce",
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
) -> torch.Tensor:
    # Pipeline core (training): masked loss for boundary detection (handles padded timesteps).
    pos_w = torch.tensor(pos_weight, device=cls_logits.device, dtype=cls_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(
        cls_logits,
        labels,
        pos_weight=pos_w.expand_as(cls_logits),
        reduction="none",
    )
    if loss_type == "focal":
        # Binary focal loss built on top of BCE-with-logits:
        #   FL = alpha_t * (1 - p_t)^gamma * BCE
        # where p_t = p if y=1 else (1-p).
        p = torch.sigmoid(cls_logits)
        pt = p * labels + (1.0 - p) * (1.0 - labels)
        modulating = (1.0 - pt).clamp(min=0.0).pow(float(focal_gamma))
        loss = bce * modulating
        if focal_alpha is not None:
            a = float(focal_alpha)
            alpha_t = a * labels + (1.0 - a) * (1.0 - labels)
            loss = loss * alpha_t
    else:
        loss = bce
    return (loss * masks).sum() / (masks.sum() + 1e-8)


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_weight: float,
    optimizer,
    loss_type: str = "bce",
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
) -> float:
    # Pipeline core (training): one epoch over (features, boundary_labels) batches.
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        assert isinstance(batch, Batch)
        optimizer.zero_grad(set_to_none=True)
        features = batch.features.to(device)
        labels = batch.boundary_labels.to(device)
        masks = batch.masks.to(device)

        out_cls, _, _, _ = model(features)
        cls_logits = out_cls[0].squeeze(1)  # (B, T')
        labels_i, masks_i = _resize_like(labels, masks, cls_logits.size(-1))
        loss = compute_loss(
            cls_logits=cls_logits,
            labels=labels_i,
            masks=masks_i,
            pos_weight=pos_weight,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.item())
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate(*, model: torch.nn.Module, loader: DataLoader, device: torch.device, pos_weight: float, threshold: float = 0.5) -> TrainStats:
    # Pipeline core (validation): computes loss + precision/recall/F1 for boundary logits.
    model.eval()
    total_loss = 0.0
    all_preds: list[float] = []
    all_labels: list[float] = []

    for batch in loader:
        assert isinstance(batch, Batch)
        features = batch.features.to(device)
        labels = batch.boundary_labels.to(device)
        masks = batch.masks.to(device)

        out_cls, _, _, _ = model(features)
        cls_logits = out_cls[0].squeeze(1)
        labels_i, masks_i = _resize_like(labels, masks, cls_logits.size(-1))

        loss = compute_loss(cls_logits=cls_logits, labels=labels_i, masks=masks_i, pos_weight=pos_weight)
        total_loss += float(loss.item())

        probs = torch.sigmoid(cls_logits)
        for i in range(probs.size(0)):
            seq_len = int(masks_i[i].sum().item())
            all_preds.extend(probs[i, :seq_len].detach().cpu().numpy().tolist())
            all_labels.extend(labels_i[i, :seq_len].detach().cpu().numpy().tolist())

    if not all_preds:
        return TrainStats(train_loss=float("nan"), val_loss=total_loss / max(1, len(loader)), val_precision=0.0, val_recall=0.0, val_f1=0.0)

    preds = np.array(all_preds, dtype=np.float32)
    labs = np.array(all_labels, dtype=np.float32)
    bin_preds = (preds > float(threshold)).astype(np.float32)
    # For soft (Gaussian) labels, treat values >= 0.5 as "positive" for reporting.
    bin_labs = (labs >= 0.5).astype(np.float32)

    tp = float(((bin_preds == 1) & (bin_labs == 1)).sum())
    fp = float(((bin_preds == 1) & (bin_labs == 0)).sum())
    fn = float(((bin_preds == 0) & (bin_labs == 1)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return TrainStats(
        train_loss=float("nan"),
        val_loss=total_loss / max(1, len(loader)),
        val_precision=precision,
        val_recall=recall,
        val_f1=f1,
    )


def cosine_with_warmup(*, base_lr: float, epoch: int, num_epochs: int, warmup_epochs: int) -> float:
    # Pipeline helper: LR schedule used by `main.py` training loop.
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, (num_epochs - warmup_epochs))
    return base_lr * 0.5 * (1.0 + float(np.cos(np.pi * progress)))
