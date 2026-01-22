from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_STEPS = ["step1", "step2", "step3", "step4"]


def _repo_root_from_cwd() -> Path:
    cwd = Path.cwd().resolve()
    marker = cwd / "requirements.txt"
    if marker.exists():
        return cwd

    child_repo = cwd / "aml-procedural-mistake-detection"
    if (child_repo / "requirements.txt").exists():
        return child_repo.resolve()

    for parent in [cwd, *cwd.parents]:
        if (parent / "requirements.txt").exists():
            return parent

    raise SystemExit("Could not find repo root (requirements.txt).")


def _step_to_script(*, step: str, actionformer_step1: bool) -> str:
    if step == "step1":
        return "extension_step1_actionformer.py" if actionformer_step1 else "extension_step1_pipeline.py"
    if step == "step2":
        return "extension_step2_verification_baseline.py"
    if step == "step3":
        return "extension_step3_task_graph_matching.py"
    if step == "step4":
        return "extension_step4_gnn_classification.py"
    raise SystemExit(f"Unknown step: {step}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run exported extension notebooks as plain Python scripts.\n"
            "Export first via: python tools/export_notebooks_to_py.py --include-actionformer"
        ),
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=DEFAULT_STEPS,
        help="Steps to run (any of: step1 step2 step3 step4).",
    )
    parser.add_argument(
        "--actionformer-step1",
        action="store_true",
        help="Use extension_step1_actionformer.py instead of extension_step1_pipeline.py for step1.",
    )
    parser.add_argument(
        "--scripts-dir",
        default=str(Path("tools") / "_exports" / "extension_py"),
        help="Directory containing exported scripts (relative to repo root).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running subsequent steps even if one fails.",
    )
    parser.add_argument(
        "--run-id",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Run id used under tools/_runs/extension_exported_py/<run-id>/",
    )
    args = parser.parse_args()

    repo_root = _repo_root_from_cwd()
    os.chdir(repo_root)

    scripts_dir = (repo_root / args.scripts_dir).resolve()
    if not scripts_dir.exists():
        raise SystemExit(
            f"Missing scripts dir: {scripts_dir}\n"
            "Create it by running:\n"
            "  python tools/export_notebooks_to_py.py --include-actionformer"
        )

    run_root = repo_root / "tools" / "_runs" / "extension_exported_py" / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for step in args.steps:
        script_name = _step_to_script(step=step, actionformer_step1=args.actionformer_step1)
        script_path = scripts_dir / script_name
        if not script_path.exists():
            failures.append(f"{step} (missing {script_name})")
            print(f"[{step}] SKIP (missing {script_path})")
            if not args.continue_on_error:
                break
            continue

        log_path = run_root / f"{step}.{script_path.stem}.log"
        print(f"[{step}] running {script_path.relative_to(repo_root)} ...")
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"$ {sys.executable} {script_path}\n\n")
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(repo_root),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if proc.returncode != 0:
            failures.append(f"{step} (exit {proc.returncode})")
            print(f"[{step}] FAILED (see {log_path})")
            if not args.continue_on_error:
                break
        else:
            print(f"[{step}] OK (log {log_path})")

    summary = run_root / "SUMMARY.txt"
    summary.write_text(
        "\n".join(
            [
                f"run_id: {args.run_id}",
                f"steps: {', '.join(args.steps)}",
                f"actionformer_step1: {args.actionformer_step1}",
                f"scripts_dir: {scripts_dir}",
                f"failures: {', '.join(failures) if failures else 'none'}",
                f"logs: {run_root}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

