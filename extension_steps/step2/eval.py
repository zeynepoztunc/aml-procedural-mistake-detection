from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import EvalConfig, TrainConfig
from .data import TaskVerificationDataset, collate_fn
from .train import compute_metrics, evaluate, train_epoch


def group_by_recipe(processed_data: dict[str, dict[str, Any]]) -> dict[int, list[str]]:
    recipe_groups: dict[int, list[str]] = defaultdict(list)
    for recording_id, rec in processed_data.items():
        recipe_groups[int(rec["recipe_id"])].append(recording_id)
    return dict(recipe_groups)


def leave_one_recipe_out_eval(
    *,
    model_factory: Callable[[], nn.Module],
    processed_data: dict[str, dict[str, Any]],
    recipe_groups: dict[int, list[str]],
    train_cfg: TrainConfig,
    eval_cfg: EvalConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    recipe_ids = list(recipe_groups.keys())
    results: list[dict[str, Any]] = []

    for fold_idx, test_recipe in enumerate(recipe_ids):
        test_ids = recipe_groups[test_recipe]
        train_ids: list[str] = []
        for recipe_id, recs in recipe_groups.items():
            if recipe_id != test_recipe:
                train_ids.extend(recs)

        if len(test_ids) < eval_cfg.min_test_size or len(train_ids) < eval_cfg.min_train_size:
            continue

        train_ds = TaskVerificationDataset(train_ids, processed_data, max_steps=train_cfg.max_steps)
        test_ds = TaskVerificationDataset(test_ids, processed_data, max_steps=train_cfg.max_steps)
        train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate_fn)

        model = model_factory().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([train_cfg.pos_weight], device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.num_epochs)

        best_loss = float("inf")
        best_state = None
        for _epoch in range(train_cfg.num_epochs):
            train_loss, _, _ = train_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            if train_loss < best_loss:
                best_loss = train_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        test_loss, preds, labels = evaluate(model, test_loader, criterion, device)
        metrics = compute_metrics(preds, labels, threshold=eval_cfg.threshold)

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

