from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch

from .config import EvalConfig, TrainConfig, repo_root_from_file
from .eval import group_by_recipe, leave_one_recipe_out_eval, summarize_folds
from .io import load_step_embeddings_pkl
from .models import LSTMTaskVerifier, MLPTaskVerifier, TransformerTaskVerifier
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
    parser = argparse.ArgumentParser(description="Step 2: task verification baselines (refactored).")
    parser.add_argument(
        "--embeddings-pkl",
        nargs="+",
        default=[str(Path("extension_data") / "step_embeddings_gt.pkl")],
        help="One or more Step-1 embedding pkls to evaluate (relative to repo root unless absolute).",
    )
    parser.add_argument("--model", choices=["mlp", "transformer", "lstm"], default="mlp")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
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


def _make_model(model_name: str, *, feature_dim: int, train_cfg: TrainConfig) -> torch.nn.Module:
    if model_name == "mlp":
        return MLPTaskVerifier(feature_dim, hidden_dim=train_cfg.hidden_dim, dropout=train_cfg.dropout)
    if model_name == "transformer":
        return TransformerTaskVerifier(
            feature_dim,
            hidden_dim=max(256, train_cfg.hidden_dim * 2),
            dropout=max(0.1, train_cfg.dropout),
            max_steps=train_cfg.max_steps,
        )
    if model_name == "lstm":
        return LSTMTaskVerifier(
            feature_dim,
            hidden_dim=max(256, train_cfg.hidden_dim * 2),
            dropout=train_cfg.dropout,
        )
    raise ValueError(f"Unknown model: {model_name}")


def run_one(
    *,
    repo_root: Path,
    embeddings_pkl: Path,
    model_name: str,
    train_cfg: TrainConfig,
    eval_cfg: EvalConfig,
    device: torch.device,
) -> dict[str, Any]:
    loaded = load_step_embeddings_pkl(embeddings_pkl)
    recipe_groups = group_by_recipe(loaded.data)

    def model_factory() -> torch.nn.Module:
        return _make_model(model_name, feature_dim=loaded.feature_dim, train_cfg=train_cfg)

    folds = leave_one_recipe_out_eval(
        model_factory=model_factory,
        processed_data=loaded.data,
        recipe_groups=recipe_groups,
        train_cfg=train_cfg,
        eval_cfg=eval_cfg,
        device=device,
    )
    summary = summarize_folds(folds)

    rel = _try_relative(embeddings_pkl, repo_root)
    print(f"\n=== Step 2: {model_name.upper()} on {rel} ===")
    print(f"Method: {loaded.method} | Feature dim: {loaded.feature_dim} | Folds: {len(folds)}")
    for metric, stats in summary.items():
        print(f"{metric:>9s}: {format_mean_std(**stats)}")

    return {
        "embeddings_pkl": str(embeddings_pkl),
        "method": loaded.method,
        "feature_dim": loaded.feature_dim,
        "model": model_name,
        "train_cfg": train_cfg,
        "eval_cfg": eval_cfg,
        "folds": folds,
        "summary": summary,
    }


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_file(Path(__file__))
    os.chdir(repo_root)

    device = _select_device(args.device)
    print(f"Device: {device.type}")

    set_seed(args.seed)

    train_cfg = TrainConfig(
        max_steps=args.max_steps,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        pos_weight=args.pos_weight,
        seed=args.seed,
    )
    eval_cfg = EvalConfig(threshold=args.threshold)

    results = []
    for pkl in args.embeddings_pkl:
        embeddings_pkl = _resolve_under_repo(repo_root, pkl)
        try:
            results.append(
                run_one(
                    repo_root=repo_root,
                    embeddings_pkl=embeddings_pkl,
                    model_name=args.model,
                    train_cfg=train_cfg,
                    eval_cfg=eval_cfg,
                    device=device,
                )
            )
        except FileNotFoundError:
            rel = _try_relative(embeddings_pkl, repo_root)
            print(f"\nMissing embeddings file: {rel}")
            print("Run Step 1 first to generate step embeddings (e.g. extension_data/step_embeddings_gt.pkl).")
            return 2
        except Exception as e:
            rel = _try_relative(embeddings_pkl, repo_root)
            print(f"\nFailed loading/evaluating embeddings file: {rel}")
            print(f"{type(e).__name__}: {e}")
            return 2

    if args.out_json:
        out_path = _resolve_under_repo(repo_root, args.out_json)
        write_json(out_path, results if len(results) > 1 else results[0])
        rel_out = _try_relative(out_path, repo_root)
        print(f"\nWrote: {rel_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
