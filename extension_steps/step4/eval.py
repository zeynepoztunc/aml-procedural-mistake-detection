from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import EvalConfig, TrainConfig
from .metrics import compute_binary_metrics
from .pooling import PooledGraphDataset, collate_pooled
from .train import eval_epoch_pooled, eval_epoch_pyg, train_epoch_pooled, train_epoch_pyg


def group_by_recipe(realized_graphs: dict[str, dict[str, Any]]) -> dict[int, list[str]]:
    groups: dict[int, list[str]] = defaultdict(list)
    for recording_id, g in realized_graphs.items():
        groups[int(g["recipe_id"])].append(recording_id)
    return dict(groups)


def leave_one_recipe_out_pooled(
    *,
    realized_graphs: dict[str, dict[str, Any]],
    recipe_groups: dict[int, list[str]],
    model_factory: Callable[[int], nn.Module],
    input_dim: int,
    train_cfg: TrainConfig,
    eval_cfg: EvalConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for fold_idx, test_recipe in enumerate(recipe_groups.keys()):
        test_ids = recipe_groups[test_recipe]
        train_ids: list[str] = []
        for recipe_id, ids in recipe_groups.items():
            if recipe_id != test_recipe:
                train_ids.extend(ids)

        if len(test_ids) < eval_cfg.min_test_size or len(train_ids) < eval_cfg.min_train_size:
            continue

        train_ds = PooledGraphDataset(train_ids, realized_graphs)
        test_ds = PooledGraphDataset(test_ids, realized_graphs)
        train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate_pooled)
        test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate_pooled)

        model = model_factory(input_dim).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([train_cfg.pos_weight], device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.num_epochs)

        best_loss = float("inf")
        best_state = None
        for _epoch in range(train_cfg.num_epochs):
            train_loss, _, _ = train_epoch_pooled(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            if train_loss < best_loss:
                best_loss = train_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)

        test_loss, preds, labels = eval_epoch_pooled(model, test_loader, criterion, device)
        metrics = compute_binary_metrics(preds, labels, threshold=eval_cfg.threshold)
        results.append(
            {
                "fold": fold_idx,
                "test_recipe": int(test_recipe),
                "train_size": len(train_ids),
                "test_size": len(test_ids),
                "train_cfg": asdict(train_cfg),
                "eval_cfg": asdict(eval_cfg),
                "test_loss": float(test_loss),
                **metrics,
            }
        )
    return results


def leave_one_recipe_out_pyg(
    *,
    pyg_data: dict[str, Any],
    recipe_groups: dict[int, list[str]],
    model_factory: Callable[[], nn.Module],
    train_cfg: TrainConfig,
    eval_cfg: EvalConfig,
    device: torch.device,
    pyg_dataloader_cls: Any,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for fold_idx, test_recipe in enumerate(recipe_groups.keys()):
        test_ids = [rid for rid in recipe_groups[test_recipe] if rid in pyg_data]
        train_ids: list[str] = []
        for recipe_id, ids in recipe_groups.items():
            if recipe_id != test_recipe:
                train_ids.extend(ids)
        train_ids = [rid for rid in train_ids if rid in pyg_data]

        if len(test_ids) < eval_cfg.min_test_size or len(train_ids) < eval_cfg.min_train_size:
            continue

        train_graphs = [pyg_data[rid] for rid in train_ids]
        test_graphs = [pyg_data[rid] for rid in test_ids]
        train_loader = pyg_dataloader_cls(train_graphs, batch_size=train_cfg.batch_size, shuffle=True)
        test_loader = pyg_dataloader_cls(test_graphs, batch_size=train_cfg.batch_size, shuffle=False)

        model = model_factory().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([train_cfg.pos_weight], device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.num_epochs)

        best_loss = float("inf")
        best_state = None
        for _epoch in range(train_cfg.num_epochs):
            train_loss, _, _ = train_epoch_pyg(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            if train_loss < best_loss:
                best_loss = train_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)

        test_loss, preds, labels = eval_epoch_pyg(model, test_loader, criterion, device)
        metrics = compute_binary_metrics(preds, labels, threshold=eval_cfg.threshold)
        results.append(
            {
                "fold": fold_idx,
                "test_recipe": int(test_recipe),
                "train_size": len(train_ids),
                "test_size": len(test_ids),
                "train_cfg": asdict(train_cfg),
                "eval_cfg": asdict(eval_cfg),
                "test_loss": float(test_loss),
                **metrics,
            }
        )
    return results


def summarize_folds(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = ["accuracy", "precision", "recall", "f1", "auc"]
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(r[key]) for r in results if key in r]
        if not values:
            summary[key] = {"mean": float("nan"), "std": float("nan")}
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary[key] = {"mean": float(arr.mean()), "std": float(arr.std())}
    return summary

