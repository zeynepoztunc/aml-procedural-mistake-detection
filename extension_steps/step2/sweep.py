from __future__ import annotations

import argparse
import csv
import itertools
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import EvalConfig, TrainConfig, repo_root_from_file
from .data import TaskVerificationDataset, collate_fn
from .eval import group_by_recipe, summarize_folds
from .io import load_step_embeddings_pkl
from .main import _make_model
from .train import compute_metrics, evaluate, set_seed, train_epoch


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
    p = argparse.ArgumentParser(description="Sweep Step 2 hyperparameters and write a CSV summary.")
    p.add_argument(
        "--embeddings-pkl",
        nargs="+",
        default=[str(Path("extension_data") / "step_embeddings_gt.pkl")],
        help="One or more Step-1 embedding pkls to evaluate.",
    )
    p.add_argument("--models", nargs="+", default=["mlp"], choices=["mlp", "transformer", "lstm"])
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seeds", nargs="+", default=["1337"], help="Random seeds to sweep (space/comma separated).")
    p.add_argument("--no-progress", action="store_true", help="Disable progress bars.")

    p.add_argument("--max-steps", nargs="+", default=["50"], help="Max steps (pad/truncate) values to sweep.")
    p.add_argument("--num-epochs", nargs="+", default=["50"], help="Epoch counts to sweep.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", nargs="+", default=["1e-4"], help="Learning rates to sweep.")
    p.add_argument("--hidden-dim", nargs="+", default=["256"], help="Hidden dims to sweep.")
    p.add_argument("--dropout", nargs="+", default=["0.2"], help="Dropout values to sweep.")
    p.add_argument("--pos-weight", nargs="+", default=["2.0"], help="Positive class weights to sweep.")

    p.add_argument(
        "--thresholds",
        nargs="+",
        default=["0.5"],
        help="Decision thresholds to sweep (computed from the same fold predictions; no retraining).",
    )
    p.add_argument("--sort-metric", default="f1_mean", help="CSV column to sort by for the Top-N summary.")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument(
        "--out-csv",
        default=str(Path("extension_data") / "sweeps" / "step2_sweep.csv"),
        help="CSV output path.",
    )
    return p.parse_args()


def _select_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def _loro_eval_thresholds(
    *,
    processed_data: dict[str, dict[str, Any]],
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
    recipe_groups = group_by_recipe(processed_data)
    recipe_ids = list(recipe_groups.keys())

    per_thr_results: dict[float, list[dict[str, Any]]] = {float(t): [] for t in thresholds}

    def model_factory() -> nn.Module:
        return _make_model(model_name, feature_dim=int(feature_dim), train_cfg=train_cfg)

    fold_iter = tqdm(
        recipe_ids,
        desc="LORO folds",
        leave=False,
        disable=not show_progress,
    )
    for fold_idx, test_recipe in enumerate(fold_iter):
        test_ids = recipe_groups[test_recipe]
        train_ids: list[str] = []
        for recipe_id, recs in recipe_groups.items():
            if recipe_id != test_recipe:
                train_ids.extend(recs)

        if len(test_ids) < base_eval_cfg.min_test_size or len(train_ids) < base_eval_cfg.min_train_size:
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

        for thr in thresholds:
            metrics = compute_metrics(preds, labels, threshold=float(thr))
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

    # Summarize each threshold.
    out: dict[float, dict[str, dict[str, float]]] = {}
    for thr, folds in per_thr_results.items():
        out[thr] = summarize_folds(folds)
        out[thr]["folds_used"] = {"mean": float(len(folds)), "std": 0.0}
    return out


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_file(Path(__file__))
    os.chdir(repo_root)

    device = _select_device(args.device)
    print(f"Device: {device.type}")

    models = list(args.models)
    seeds = _parse_ints(list(args.seeds))
    max_steps_list = _parse_ints(list(args.max_steps))
    num_epochs_list = _parse_ints(list(args.num_epochs))
    lrs = _parse_floats(list(args.lr))
    hidden_dims = _parse_ints(list(args.hidden_dim))
    dropouts = _parse_floats(list(args.dropout))
    pos_weights = _parse_floats(list(args.pos_weight))
    thresholds = _parse_floats(list(args.thresholds))

    show_progress = not bool(args.no_progress)

    out_path = _resolve_under_repo(repo_root, str(args.out_csv))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "embeddings_pkl",
        "method",
        "feature_dim",
        "model",
        "seed",
        "max_steps",
        "num_epochs",
        "batch_size",
        "lr",
        "hidden_dim",
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

    rows: list[dict[str, Any]] = []
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for pkl in args.embeddings_pkl:
            embeddings_pkl = _resolve_under_repo(repo_root, pkl)
            loaded = load_step_embeddings_pkl(embeddings_pkl)
            processed_data = loaded.data

            combos = itertools.product(models, seeds, max_steps_list, num_epochs_list, lrs, hidden_dims, dropouts, pos_weights)
            total = (
                len(models)
                * len(seeds)
                * len(max_steps_list)
                * len(num_epochs_list)
                * len(lrs)
                * len(hidden_dims)
                * len(dropouts)
                * len(pos_weights)
            )
            combo_iter = tqdm(
                combos,
                desc=f"sweep ({Path(pkl).name})",
                leave=True,
                total=total,
                disable=not show_progress,
            )

            for (model_name, seed, max_steps, num_epochs, lr, hidden_dim, dropout, pos_weight) in combo_iter:
                combo_iter.set_postfix(
                    {
                        "model": str(model_name),
                        "pos_w": float(pos_weight),
                        "thr": ",".join(f"{t:.2f}" for t in thresholds),
                    }
                )
                set_seed(int(seed))
                train_cfg = TrainConfig(
                    max_steps=int(max_steps),
                    num_epochs=int(num_epochs),
                    batch_size=int(args.batch_size),
                    lr=float(lr),
                    hidden_dim=int(hidden_dim),
                    dropout=float(dropout),
                    pos_weight=float(pos_weight),
                    seed=int(seed),
                )
                base_eval_cfg = EvalConfig(threshold=0.5)

                summaries_by_thr = _loro_eval_thresholds(
                    processed_data=processed_data,
                    feature_dim=int(loaded.feature_dim),
                    model_name=str(model_name),
                    train_cfg=train_cfg,
                    base_eval_cfg=base_eval_cfg,
                    thresholds=thresholds,
                    device=device,
                    show_progress=show_progress,
                )
                for thr, summary in summaries_by_thr.items():
                    row: dict[str, Any] = {
                        "embeddings_pkl": str(_try_relative(embeddings_pkl, repo_root)),
                        "method": str(loaded.method),
                        "feature_dim": int(loaded.feature_dim),
                        "model": str(model_name),
                        "seed": int(seed),
                        "max_steps": int(max_steps),
                        "num_epochs": int(num_epochs),
                        "batch_size": int(args.batch_size),
                        "lr": float(lr),
                        "hidden_dim": int(hidden_dim),
                        "dropout": float(dropout),
                        "pos_weight": float(pos_weight),
                        "threshold": float(thr),
                        "folds_used": int(summary["folds_used"]["mean"]),
                    }
                    for metric in ("accuracy", "precision", "recall", "f1", "auc"):
                        row[f"{metric}_mean"] = float(summary[metric]["mean"])
                        row[f"{metric}_std"] = float(summary[metric]["std"])
                    writer.writerow(row)
                    rows.append(row)

    print(f"Wrote: {out_path}")

    sort_key = str(args.sort_metric)
    if rows and sort_key in rows[0]:
        top_n = max(0, int(args.top_n))
        rows_sorted = sorted(rows, key=lambda r: float(r.get(sort_key, -1e9)), reverse=True)
        if top_n:
            print(f"\nTop {top_n} by {sort_key}:")
            for r in rows_sorted[:top_n]:
                print(
                    f"  {r['model']} thr={float(r['threshold']):.2f} pos_w={float(r['pos_weight']):.2f} "
                    f"f1={float(r['f1_mean']):.4f} auc={float(r['auc_mean']):.4f} "
                    f"p={float(r['precision_mean']):.4f} r={float(r['recall_mean']):.4f} "
                    f"({r['embeddings_pkl']})"
                )
    else:
        print(f"Note: sort metric '{sort_key}' not found in CSV columns.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
