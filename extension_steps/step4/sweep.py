from __future__ import annotations

import argparse
import csv
import itertools
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import EvalConfig, TrainConfig, repo_root_from_file
from .eval import group_by_recipe, summarize_folds
from .io import load_realized_task_graphs_pkl
from .metrics import compute_binary_metrics
from .models import PooledMLPClassifier, build_gnn_models
from .pooling import PooledGraphDataset, collate_pooled
from .pyg import create_pyg_data, try_import_pyg
from .train import eval_epoch_pooled, eval_epoch_pyg, set_seed, train_epoch_pooled, train_epoch_pyg


def _resolve_under_repo(repo_root: Path, maybe_rel: str) -> Path:
    p = Path(maybe_rel)
    return p if p.is_absolute() else (repo_root / p)


def _try_relative(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except Exception:
        return path


def _parse_floats(values: list[str]) -> list[float]:
    out: list[float] = []
    for v in values:
        for part in v.split(","):
            part = part.strip()
            if part:
                out.append(float(part))
    return out


def _parse_ints(values: list[str]) -> list[int]:
    out: list[int] = []
    for v in values:
        for part in v.split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep Step 4 hyperparameters and write a CSV summary.")
    p.add_argument(
        "--graphs-pkl",
        default=str(Path("extension_data") / "realized_task_graphs.pkl"),
        help="Path to Step-3 output realized_task_graphs.pkl (relative to repo root unless absolute).",
    )
    p.add_argument("--model", choices=["pooled", "gcn", "gat", "sage"], default="sage")

    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seeds", nargs="+", default=["1337"], help="Random seeds to sweep (space/comma separated).")
    p.add_argument("--no-progress", action="store_true", help="Disable progress bars.")

    # Training params to sweep (keep small to avoid huge runtimes).
    p.add_argument("--num-epochs", nargs="+", default=["50"], help="Epoch counts to sweep.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", nargs="+", default=["1e-3"], help="Learning rates to sweep.")
    p.add_argument("--weight-decay", nargs="+", default=["1e-4"], help="Weight decay values to sweep.")
    p.add_argument("--hidden-dim", nargs="+", default=["128"], help="Hidden dims to sweep.")
    p.add_argument("--num-layers", nargs="+", default=["2"], help="Number of GNN layers to sweep (ignored for pooled).")
    p.add_argument("--dropout", nargs="+", default=["0.3"], help="Dropout values to sweep.")
    p.add_argument("--pos-weight", nargs="+", default=["2.0"], help="Positive class weights to sweep.")

    # Eval thresholds: computed from the same fold predictions; no retraining.
    p.add_argument("--thresholds", nargs="+", default=["0.5"], help="Decision thresholds to sweep.")

    # Sorting: recall-first, precision second.
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument(
        "--out-csv",
        default=str(Path("extension_data") / "sweeps" / "step4_sweep.csv"),
        help="CSV output path.",
    )
    return p.parse_args()


def _select_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loro_eval_thresholds(
    *,
    realized_graphs: dict[str, dict[str, Any]],
    feature_dim: int,
    model_name: str,
    train_cfg: TrainConfig,
    base_eval_cfg: EvalConfig,
    thresholds: list[float],
    device: torch.device,
    show_progress: bool,
) -> dict[float, dict[str, dict[str, float]]]:
    """
    Run LORO once for a given training config and compute metrics for multiple thresholds
    from the same fold predictions.
    """
    recipe_groups = group_by_recipe(realized_graphs)
    recipe_ids = list(recipe_groups.keys())

    per_thr_results: dict[float, list[dict[str, Any]]] = {float(t): [] for t in thresholds}

    if model_name == "pooled":
        input_dim = feature_dim * 2

        def model_factory() -> nn.Module:
            return PooledMLPClassifier(input_dim, hidden_dim=train_cfg.hidden_dim, dropout=train_cfg.dropout)

        fold_iter = tqdm(recipe_ids, desc="LORO folds", leave=False, disable=not show_progress)
        for fold_idx, test_recipe in enumerate(fold_iter):
            test_ids = recipe_groups[test_recipe]
            train_ids: list[str] = []
            for recipe_id, ids in recipe_groups.items():
                if recipe_id != test_recipe:
                    train_ids.extend(ids)

            if len(test_ids) < base_eval_cfg.min_test_size or len(train_ids) < base_eval_cfg.min_train_size:
                continue

            train_ds = PooledGraphDataset(train_ids, realized_graphs)
            test_ds = PooledGraphDataset(test_ids, realized_graphs)
            train_loader = DataLoader(
                train_ds, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate_pooled
            )
            test_loader = DataLoader(
                test_ds, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate_pooled
            )

            model = model_factory().to(device)
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
            for thr in thresholds:
                metrics = compute_binary_metrics(preds, labels, threshold=float(thr))
                per_thr_results[float(thr)].append(
                    {
                        "fold": fold_idx,
                        "test_recipe": int(test_recipe),
                        "train_size": len(train_ids),
                        "test_size": len(test_ids),
                        "train_cfg": asdict(train_cfg),
                        "eval_cfg": asdict(EvalConfig(**{**asdict(base_eval_cfg), "threshold": float(thr)})),
                        "test_loss": float(test_loss),
                        **metrics,
                    }
                )
    else:
        pyg = try_import_pyg()
        if pyg is None:
            raise RuntimeError("Requested a GNN model sweep, but torch-geometric is not installed.")

        pyg_data = {rid: create_pyg_data(pyg, g, rid) for rid, g in realized_graphs.items()}
        recipe_groups = {k: [rid for rid in v if rid in pyg_data] for k, v in recipe_groups.items()}

        gnn = build_gnn_models(pyg)
        if model_name == "gcn":
            ModelClass = gnn.GCNClassifier
            model_kwargs: dict[str, Any] = {}
        elif model_name == "gat":
            ModelClass = gnn.GATClassifier
            model_kwargs = {"heads": 4}
        else:
            ModelClass = gnn.GraphSAGEClassifier
            model_kwargs = {}

        def model_factory() -> nn.Module:
            return ModelClass(
                feature_dim,
                hidden_dim=train_cfg.hidden_dim,
                num_layers=train_cfg.num_layers,
                dropout=train_cfg.dropout,
                **model_kwargs,
            )

        fold_iter = tqdm(recipe_ids, desc="LORO folds", leave=False, disable=not show_progress)
        for fold_idx, test_recipe in enumerate(fold_iter):
            test_ids = recipe_groups[test_recipe]
            train_ids: list[str] = []
            for recipe_id, ids in recipe_groups.items():
                if recipe_id != test_recipe:
                    train_ids.extend(ids)

            if len(test_ids) < base_eval_cfg.min_test_size or len(train_ids) < base_eval_cfg.min_train_size:
                continue

            train_graphs = [pyg_data[rid] for rid in train_ids]
            test_graphs = [pyg_data[rid] for rid in test_ids]
            train_loader = pyg.PyGDataLoader(train_graphs, batch_size=train_cfg.batch_size, shuffle=True)
            test_loader = pyg.PyGDataLoader(test_graphs, batch_size=train_cfg.batch_size, shuffle=False)

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
            for thr in thresholds:
                metrics = compute_binary_metrics(preds, labels, threshold=float(thr))
                per_thr_results[float(thr)].append(
                    {
                        "fold": fold_idx,
                        "test_recipe": int(test_recipe),
                        "train_size": len(train_ids),
                        "test_size": len(test_ids),
                        "train_cfg": asdict(train_cfg),
                        "eval_cfg": asdict(EvalConfig(**{**asdict(base_eval_cfg), "threshold": float(thr)})),
                        "test_loss": float(test_loss),
                        **metrics,
                    }
                )

    out: dict[float, dict[str, dict[str, float]]] = {}
    for thr, folds in per_thr_results.items():
        out[thr] = summarize_folds(folds)
        out[thr]["folds_used"] = {"mean": float(len(folds)), "std": 0.0}
    return out


def _sort_key_recall_then_precision(row: dict[str, Any]) -> tuple[float, float, float]:
    return (float(row["recall_mean"]), float(row["precision_mean"]), float(row["f1_mean"]))


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_file(Path(__file__))
    os.chdir(repo_root)

    device = _select_device(args.device)
    print(f"Device: {device.type}")

    seeds = _parse_ints(list(args.seeds))
    num_epochs_list = _parse_ints(list(args.num_epochs))
    lrs = _parse_floats(list(args.lr))
    weight_decays = _parse_floats(list(args.weight_decay))
    hidden_dims = _parse_ints(list(args.hidden_dim))
    num_layers_list = _parse_ints(list(args.num_layers))
    dropouts = _parse_floats(list(args.dropout))
    pos_weights = _parse_floats(list(args.pos_weight))
    thresholds = _parse_floats(list(args.thresholds))

    show_progress = not bool(args.no_progress)

    graphs_pkl = _resolve_under_repo(repo_root, str(args.graphs_pkl))
    loaded = load_realized_task_graphs_pkl(graphs_pkl)
    realized_graphs = loaded.realized_graphs
    feature_dim = loaded.feature_dim

    out_path = _resolve_under_repo(repo_root, str(args.out_csv))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "graphs_pkl",
        "model",
        "feature_dim",
        "seed",
        "num_epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "hidden_dim",
        "num_layers",
        "dropout",
        "pos_weight",
        "threshold",
        "folds_used",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "auc_mean",
        "auc_std",
    ]

    combos = list(
        itertools.product(
            seeds,
            num_epochs_list,
            lrs,
            weight_decays,
            hidden_dims,
            num_layers_list,
            dropouts,
            pos_weights,
        )
    )

    rows: list[dict[str, Any]] = []
    combo_iter = tqdm(combos, desc=f"sweep ({Path(args.graphs_pkl).name})", disable=not show_progress)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for seed, num_epochs, lr, weight_decay, hidden_dim, num_layers, dropout, pos_weight in combo_iter:
            set_seed(int(seed))
            train_cfg = TrainConfig(
                num_epochs=int(num_epochs),
                batch_size=int(args.batch_size),
                lr=float(lr),
                weight_decay=float(weight_decay),
                hidden_dim=int(hidden_dim),
                num_layers=int(num_layers),
                dropout=float(dropout),
                pos_weight=float(pos_weight),
                seed=int(seed),
            )
            base_eval_cfg = EvalConfig()

            per_thr_summary = _loro_eval_thresholds(
                realized_graphs=realized_graphs,
                feature_dim=int(feature_dim),
                model_name=str(args.model),
                train_cfg=train_cfg,
                base_eval_cfg=base_eval_cfg,
                thresholds=thresholds,
                device=device,
                show_progress=show_progress,
            )

            for thr, summary in per_thr_summary.items():
                row: dict[str, Any] = {
                    "graphs_pkl": str(_try_relative(graphs_pkl, repo_root)),
                    "model": str(args.model),
                    "feature_dim": int(feature_dim),
                    "seed": int(seed),
                    "num_epochs": int(num_epochs),
                    "batch_size": int(args.batch_size),
                    "lr": float(lr),
                    "weight_decay": float(weight_decay),
                    "hidden_dim": int(hidden_dim),
                    "num_layers": int(num_layers),
                    "dropout": float(dropout),
                    "pos_weight": float(pos_weight),
                    "threshold": float(thr),
                    "folds_used": float(summary["folds_used"]["mean"]),
                    "accuracy_mean": float(summary["accuracy"]["mean"]),
                    "accuracy_std": float(summary["accuracy"]["std"]),
                    "precision_mean": float(summary["precision"]["mean"]),
                    "precision_std": float(summary["precision"]["std"]),
                    "recall_mean": float(summary["recall"]["mean"]),
                    "recall_std": float(summary["recall"]["std"]),
                    "f1_mean": float(summary["f1"]["mean"]),
                    "f1_std": float(summary["f1"]["std"]),
                    "auc_mean": float(summary["auc"]["mean"]),
                    "auc_std": float(summary["auc"]["std"]),
                }
                writer.writerow(row)
                rows.append(row)

    print(f"Wrote: {_try_relative(out_path, repo_root)}")

    # Recall-first summary.
    rows_sorted = sorted(rows, key=_sort_key_recall_then_precision, reverse=True)
    print("\nTop by recall_mean (tie-break: precision_mean, then f1_mean):")
    for r in rows_sorted[: int(args.top_n)]:
        print(
            f"  {r['model']} thr={r['threshold']:.2f} pos_w={r['pos_weight']:.2f} "
            f"recall={r['recall_mean']:.4f} precision={r['precision_mean']:.4f} f1={r['f1_mean']:.4f} auc={r['auc_mean']:.4f} "
            f"(seed={r['seed']} lr={r['lr']:.1e} wd={r['weight_decay']:.1e} hd={r['hidden_dim']} l={r['num_layers']})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

