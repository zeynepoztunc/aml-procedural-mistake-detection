from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


NOTEBOOKS = [
    ("step1", "extension_step1_localization.ipynb"),
    ("step2", "extension_step2_verification_baseline.ipynb"),
    ("step3", "extension_step3_task_graph_matching.ipynb"),
    ("step4", "extension_step4_gnn_classification.ipynb"),
]


def _repo_root_from_cwd() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "requirements.txt").exists():
        return cwd
    raise SystemExit("Run this from the repo root (the folder containing requirements.txt).")


def _parse_steps(raw: str) -> set[str]:
    normalized = raw.strip().lower()
    if normalized in {"all", "*"}:
        return {name for name, _ in NOTEBOOKS}

    steps = {s.strip().lower() for s in normalized.split(",") if s.strip()}
    valid = {name for name, _ in NOTEBOOKS}
    unknown = sorted(steps - valid)
    if unknown:
        raise SystemExit(f"Unknown steps: {unknown}. Valid: {sorted(valid)}")
    return steps


def _run_nbconvert(
    *,
    repo_root: Path,
    notebook: Path,
    output_notebook: Path,
    log_path: Path,
    timeout_sec: int,
    env: dict[str, str],
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        f"--ExecutePreprocessor.timeout={timeout_sec}",
        "--output",
        str(output_notebook),
        str(notebook),
    ]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n\n")
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute the extension notebooks (Step 1–4) using nbconvert."
    )
    parser.add_argument(
        "--steps",
        default="all",
        help="Comma-separated steps to run: step1,step2,step3,step4 (default: all).",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=3600,
        help="Per-notebook execution timeout in seconds (default: 3600).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running subsequent notebooks even if one fails.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Set HF/Transformers offline env vars (Step 3 will require cached models).",
    )
    parser.add_argument(
        "--run-id",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Run id used under tools/_runs/extension_local_notebooks/<run-id>/",
    )
    args = parser.parse_args()

    repo_root = _repo_root_from_cwd()
    os.chdir(repo_root)

    if importlib.util.find_spec("jupyter") is None:
        raise SystemExit(
            "Missing dependency 'jupyter'. Install it in your active venv:\n"
            "  pip install jupyter nbconvert\n"
            "or:\n"
            "  pip install jupyterlab\n"
        )

    selected = _parse_steps(args.steps)
    run_root = repo_root / "tools" / "_runs" / "extension_local_notebooks" / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    if args.offline:
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"

    # Preflight check (data paths + imports)
    preflight_log = run_root / "preflight.log"
    with preflight_log.open("w", encoding="utf-8") as log:
        log.write("$ python tools/run_extension_local.py --check-only\n\n")
        subprocess.run(
            [sys.executable, "tools/run_extension_local.py", "--check-only"],
            cwd=str(repo_root),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    failures: list[str] = []
    for step_name, nb_name in NOTEBOOKS:
        if step_name not in selected:
            continue

        notebook = repo_root / nb_name
        if not notebook.exists():
            failures.append(f"{step_name} (missing notebook: {nb_name})")
            if not args.continue_on_error:
                break
            continue

        step_dir = run_root / step_name
        step_dir.mkdir(parents=True, exist_ok=True)
        out_nb = step_dir / nb_name
        log_path = step_dir / f"{notebook.stem}.log"

        print(f"[{step_name}] executing {nb_name} ...")
        rc = _run_nbconvert(
            repo_root=repo_root,
            notebook=notebook,
            output_notebook=out_nb,
            log_path=log_path,
            timeout_sec=args.timeout_sec,
            env=env,
        )
        if rc != 0:
            failures.append(f"{step_name} (exit {rc})")
            print(f"[{step_name}] FAILED (see {log_path})")
            if not args.continue_on_error:
                break
        else:
            print(f"[{step_name}] OK (output {out_nb})")

    summary_path = run_root / "SUMMARY.txt"
    summary_lines = [
        f"run_id: {args.run_id}",
        f"steps: {', '.join(sorted(selected))}",
        f"offline: {args.offline}",
        f"timeout_sec: {args.timeout_sec}",
        f"failures: {', '.join(failures) if failures else 'none'}",
        f"logs/output: {run_root}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
