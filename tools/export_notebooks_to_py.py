from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable


DEFAULT_NOTEBOOKS = [
    "extension_step1_pipeline.ipynb",
    "extension_step2_verification_baseline.ipynb",
    "extension_step3_task_graph_matching.ipynb",
    "extension_step4_gnn_classification.ipynb",
]

STEP_OUTPUT_FILENAMES: dict[str, str] = {
    "extension_step1_actionformer.ipynb": "step1_actionformer.py",
    "extension_step1_pipeline.ipynb": "step1_pipeline.py",
    "extension_step2_verification_baseline.ipynb": "step2_verification_baseline.py",
    "extension_step3_task_graph_matching.ipynb": "step3_task_graph_matching.py",
    "extension_step4_gnn_classification.ipynb": "step4_gnn_classification.py",
}


def _repo_root_from_cwd() -> Path:
    cwd = Path.cwd().resolve()
    marker = cwd / "requirements.txt"
    if marker.exists():
        return cwd

    # Common layout for this workspace: run from parent folder that contains the repo.
    child_repo = cwd / "aml-procedural-mistake-detection"
    if (child_repo / "requirements.txt").exists():
        return child_repo.resolve()

    # Otherwise, try walking upwards.
    for parent in [cwd, *cwd.parents]:
        if (parent / "requirements.txt").exists():
            return parent

    raise SystemExit("Could not find repo root (requirements.txt).")


def _read_ipynb(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed reading notebook as JSON: {path} ({type(e).__name__}: {e})")


def _cell_source(cell: dict[str, Any]) -> str:
    src = cell.get("source") or ""
    if isinstance(src, list):
        return "".join(src)
    if isinstance(src, str):
        return src
    return str(src)


def _looks_colab_setup(code: str) -> bool:
    needles = (
        "google.colab",
        "drive.mount(",
        "Running in Google Colab",
        "REPO_URL",
        "CODE_BASE",
        "DRIVE_BASE",
        "!git ",
        "!pip ",
        "!apt-get ",
    )
    return any(n in code for n in needles)


def _sanitize_code_lines(lines: Iterable[str]) -> tuple[list[str], bool]:
    """
    Comments out IPython-specific lines (e.g., !git, %matplotlib).
    Returns (sanitized_lines, had_ipython_lines).
    """
    out: list[str] = []
    had_ipython = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(("!", "%", "%%")):
            had_ipython = True
            out.append(f"# [ipython-only] {line}")
        else:
            out.append(line)
    return out, had_ipython


def _comment_block(text: str) -> list[str]:
    lines = text.splitlines()
    if not lines:
        return []
    return ["# " + ln if ln else "#" for ln in lines]


def export_notebook_to_py(
    *,
    notebook_path: Path,
    out_path: Path,
    include_markdown: bool,
    strip_colab_setup: bool,
) -> None:
    nb = _read_ipynb(notebook_path)
    cells = nb.get("cells") or []
    if not isinstance(cells, list):
        raise SystemExit(f"Invalid notebook format (cells not a list): {notebook_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrote_ipython_note = False
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(
            "# Auto-generated from a Jupyter notebook.\n"
            f"# Source: {notebook_path.name}\n"
            "#\n"
            "# Notes:\n"
            "# - Lines starting with !/%/%% are commented out (IPython-only).\n"
            "# - Run from the repo root (folder containing requirements.txt).\n"
            "\n"
        )

        for idx, cell in enumerate(cells):
            cell_type = cell.get("cell_type")
            src = _cell_source(cell)

            f.write(f"# %% [cell {idx}]\n")

            if cell_type == "markdown":
                if include_markdown:
                    f.write("\n".join(_comment_block(src)))
                    f.write("\n\n")
                else:
                    f.write("# (markdown cell omitted)\n\n")
                continue

            if cell_type != "code":
                f.write(f"# (unknown cell type: {cell_type!r})\n\n")
                continue

            if strip_colab_setup and _looks_colab_setup(src):
                f.write("# (colab-only setup cell omitted)\n\n")
                continue

            lines = src.splitlines()
            sanitized, had_ipython = _sanitize_code_lines(lines)
            if had_ipython and not wrote_ipython_note:
                wrote_ipython_note = True
                f.write(
                    "# NOTE: This script was exported from a notebook.\n"
                    "# Some IPython-specific lines were commented out.\n\n"
                )

            f.write("\n".join(sanitized))
            if sanitized and not sanitized[-1].endswith("\n"):
                f.write("\n")
            f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export extension notebooks (*.ipynb) to runnable Python scripts (*.py)."
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path("tools") / "_exports" / "extension_py"),
        help="Output directory for exported .py files (relative to repo root).",
    )
    parser.add_argument(
        "--include-markdown",
        action="store_true",
        help="Include markdown cells as commented text.",
    )
    parser.add_argument(
        "--keep-colab-setup",
        action="store_true",
        help="Do not strip Google Colab setup cells (may leave IPython-only syntax).",
    )
    parser.add_argument(
        "--include-actionformer",
        action="store_true",
        help="Also export extension_step1_actionformer.ipynb.",
    )
    parser.add_argument(
        "--use-step-filenames",
        action="store_true",
        help="Name outputs as step*.py (e.g. step2_verification_baseline.py) instead of mirroring notebook stem.",
    )
    parser.add_argument(
        "--notebooks",
        nargs="*",
        default=None,
        help="Explicit notebook filenames to export (default: extension steps 1-4).",
    )
    args = parser.parse_args()

    repo_root = _repo_root_from_cwd()
    os.chdir(repo_root)

    notebook_names = list(args.notebooks) if args.notebooks else list(DEFAULT_NOTEBOOKS)
    if args.include_actionformer and "extension_step1_actionformer.ipynb" not in notebook_names:
        notebook_names.insert(0, "extension_step1_actionformer.ipynb")

    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exported: list[str] = []
    missing: list[str] = []
    for nb_name in notebook_names:
        nb_path = repo_root / nb_name
        if not nb_path.exists():
            missing.append(nb_name)
            continue
        if args.use_step_filenames and nb_name in STEP_OUTPUT_FILENAMES:
            out_name = STEP_OUTPUT_FILENAMES[nb_name]
        else:
            out_name = nb_path.stem + ".py"
        out_path = out_dir / out_name
        export_notebook_to_py(
            notebook_path=nb_path,
            out_path=out_path,
            include_markdown=args.include_markdown,
            strip_colab_setup=not args.keep_colab_setup,
        )
        exported.append(str(out_path.relative_to(repo_root)))

    print("=== Export notebooks to .py ===")
    print(f"Output dir: {out_dir.relative_to(repo_root)}")
    if exported:
        print("\nExported:")
        for p in exported:
            print(f"  - {p}")
    if missing:
        print("\nMissing (skipped):")
        for n in missing:
            print(f"  - {n}")

    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
