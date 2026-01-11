from __future__ import annotations

import argparse
from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from tools.notebook_runner import run_notebook_via_script  # noqa: E402


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the extension notebooks locally (no Jupyter required).")
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        choices=[1, 2, 3, 4],
        help="which extension steps to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only generate sanitized .py scripts (do not execute)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory for generated scripts + logs (default: tools/_runs/extension_local)",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue even if a step fails",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    workdir = root
    out_dir = args.out_dir or (root / "tools" / "_runs" / "extension_local")

    step_to_nb = {
        1: root / "extension_step1_localization.ipynb",
        2: root / "extension_step2_verification_baseline.ipynb",
        3: root / "extension_step3_task_graph_matching.ipynb",
        4: root / "extension_step4_gnn_classification.ipynb",
    }

    step_embeddings_path = root / "extension_data" / "step_embeddings_gt.pkl"
    realized_graphs_path = root / "extension_data" / "realized_task_graphs.pkl"

    for step in args.steps:
        nb = step_to_nb[step]
        if not nb.exists():
            print(f"[error] Missing notebook for step {step}: {nb}", file=sys.stderr)
            return 2

        if step in (2, 3) and not step_embeddings_path.exists():
            print(
                f"[error] Missing required input: {step_embeddings_path}\n"
                "Run step 1 first: `python tools/run_extension_local.py --steps 1`",
                file=sys.stderr,
            )
            return 2

        if step == 4 and not realized_graphs_path.exists():
            print(
                f"[error] Missing required input: {realized_graphs_path}\n"
                "Run step 3 first: `python tools/run_extension_local.py --steps 3`",
                file=sys.stderr,
            )
            return 2

        print(f"[run] step {step}: {nb.name}")
        res = run_notebook_via_script(
            nb,
            workdir=workdir,
            out_dir=out_dir / f"step{step}",
            dry_run=bool(args.dry_run),
        )
        print(f"[out] script: {res.generated_script}")
        print(f"[out] log:    {res.log_path}")

        if res.returncode != 0:
            print(f"[error] step {step} failed (exit {res.returncode})", file=sys.stderr)
            if not args.keep_going:
                return res.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
