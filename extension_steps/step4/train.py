from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch_pooled(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.train()
    total_loss = 0.0
    all_preds: list[float] = []
    all_labels: list[float] = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item()) * int(x.shape[0])
        preds = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.detach().cpu().numpy().flatten().tolist())

    return total_loss / max(1, len(loader.dataset)), np.asarray(all_preds), np.asarray(all_labels)


@torch.no_grad()
def eval_epoch_pooled(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds: list[float] = []
    all_labels: list[float] = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * int(x.shape[0])
        preds = torch.sigmoid(logits).cpu().numpy().flatten()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.cpu().numpy().flatten().tolist())

    return total_loss / max(1, len(loader.dataset)), np.asarray(all_preds), np.asarray(all_labels)


def train_epoch_pyg(model: nn.Module, loader, optimizer, criterion, device: torch.device):
    model.train()
    total_loss = 0.0
    all_preds: list[float] = []
    all_labels: list[float] = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y.view(-1, 1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item()) * int(batch.num_graphs)
        preds = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch.y.detach().cpu().numpy().flatten().tolist())

    return total_loss / max(1, len(loader.dataset)), np.asarray(all_preds), np.asarray(all_labels)


@torch.no_grad()
def eval_epoch_pyg(model: nn.Module, loader, criterion, device: torch.device):
    model.eval()
    total_loss = 0.0
    all_preds: list[float] = []
    all_labels: list[float] = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y.view(-1, 1))

        total_loss += float(loss.item()) * int(batch.num_graphs)
        preds = torch.sigmoid(logits).cpu().numpy().flatten()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch.y.cpu().numpy().flatten().tolist())

    return total_loss / max(1, len(loader.dataset)), np.asarray(all_preds), np.asarray(all_labels)

