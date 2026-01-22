from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch

from .config import EvalConfig, TrainConfig, repo_root_from_file
from .eval import (
    group_by_recipe,
    leave_one_recipe_out_pooled,
    leave_one_recipe_out_pyg,
    summarize_folds,
)
from .io import load_realized_task_graphs_pkl
from .models import PooledMLPClassifier, build_gnn_models
from .pyg import create_pyg_data, try_import_pyg
from .reporting import format_mean_std, write_json
from .train import set_seed


def _resolve_under_repo(repo_root: Path, maybe_rel: str) -> Path:
    p = Path(maybe_rel)
    return p if p.is_absolute() else (repo_root / p)


def _try_relative(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except Exception:
        return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 4: GNN classification (refactored).")
    parser.add_argument(
        "--graphs-pkl",
        default=str(Path("extension_data") / "realized_task_graphs.pkl"),
        help="Path to Step-3 output realized_task_graphs.pkl (relative to repo root unless absolute).",
    )
    parser.add_argument("--model", choices=["pooled", "gcn", "gat", "sage"], default="pooled")

    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pos-weight", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out-json", default=None, help="Optional path to write results JSON.")
    return parser.parse_args()


def _select_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_file(Path(__file__))
    os.chdir(repo_root)

    device = _select_device(args.device)
    print(f"Device: {device.type}")

    set_seed(args.seed)

    train_cfg = TrainConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pos_weight=args.pos_weight,
        seed=args.seed,
    )
    eval_cfg = EvalConfig(threshold=args.threshold)

    graphs_path = _resolve_under_repo(repo_root, args.graphs_pkl)
    try:
        loaded = load_realized_task_graphs_pkl(graphs_path)
    except FileNotFoundError:
        rel = _try_relative(graphs_path, repo_root)
        print(f"Missing graphs file: {rel}")
        print("Run Step 3 first to generate extension_data/realized_task_graphs.pkl.")
        return 2
    except Exception as e:
        rel = _try_relative(graphs_path, repo_root)
        print(f"Failed loading graphs file: {rel}")
        print(f"{type(e).__name__}: {e}")
        return 2

    realized_graphs = loaded.realized_graphs
    recipe_groups = group_by_recipe(realized_graphs)
    feature_dim = loaded.feature_dim

    payload: dict[str, Any]
    if args.model == "pooled":
        input_dim = feature_dim * 2

        def model_factory(inp: int) -> torch.nn.Module:
            return PooledMLPClassifier(inp, hidden_dim=train_cfg.hidden_dim, dropout=train_cfg.dropout)

        folds = leave_one_recipe_out_pooled(
            realized_graphs=realized_graphs,
            recipe_groups=recipe_groups,
            model_factory=model_factory,
            input_dim=input_dim,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            device=device,
        )
        summary = summarize_folds(folds)
        payload = {
            "graphs_pkl": str(graphs_path),
            "model": "pooled",
            "feature_dim": feature_dim,
            "train_cfg": train_cfg,
            "eval_cfg": eval_cfg,
            "folds": folds,
            "summary": summary,
        }
    else:
        pyg = try_import_pyg()
        if pyg is None:
            print("Requested a GNN model, but torch-geometric is not installed.")
            print("Install torch-geometric or run with --model pooled.")
            return 2

        pyg_data = {rid: create_pyg_data(pyg, g, rid) for rid, g in realized_graphs.items()}
        gnn = build_gnn_models(pyg)

        if args.model == "gcn":
            ModelClass = gnn.GCNClassifier
            model_kwargs = {}
        elif args.model == "gat":
            ModelClass = gnn.GATClassifier
            model_kwargs = {"heads": 4}
        else:
            ModelClass = gnn.GraphSAGEClassifier
            model_kwargs = {}

        def model_factory() -> torch.nn.Module:
            return ModelClass(
                feature_dim,
                hidden_dim=train_cfg.hidden_dim,
                num_layers=train_cfg.num_layers,
                dropout=train_cfg.dropout,
                **model_kwargs,
            )

        folds = leave_one_recipe_out_pyg(
            pyg_data=pyg_data,
            recipe_groups=recipe_groups,
            model_factory=model_factory,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            device=device,
            pyg_dataloader_cls=pyg.PyGDataLoader,
        )
        summary = summarize_folds(folds)
        payload = {
            "graphs_pkl": str(graphs_path),
            "model": args.model,
            "feature_dim": feature_dim,
            "train_cfg": train_cfg,
            "eval_cfg": eval_cfg,
            "folds": folds,
            "summary": summary,
        }

    rel = _try_relative(graphs_path, repo_root)
    print(f"\n=== Step 4: {payload['model']} on {rel} ===")
    print(f"Feature dim: {feature_dim} | Folds: {len(payload['folds'])}")
    for metric, stats in payload["summary"].items():
        print(f"{metric:>9s}: {format_mean_std(**stats)}")

    if args.out_json:
        out_path = _resolve_under_repo(repo_root, args.out_json)
        write_json(out_path, payload)
        print(f"\nWrote: {_try_relative(out_path, repo_root)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

