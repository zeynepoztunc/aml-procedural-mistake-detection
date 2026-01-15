from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from typing import Iterable


def _repo_root_from_cwd() -> Path:
    cwd = Path.cwd().resolve()
    marker = cwd / "requirements.txt"
    if marker.exists():
        return cwd
    raise SystemExit(
        "Run this from the repo root (the folder containing requirements.txt)."
    )


def _check_paths(repo_root: Path) -> list[str]:
    required_paths = [
        repo_root / "data" / "features" / "egovlp",
        repo_root / "annotations" / "annotation_json" / "step_annotations.json",
        repo_root / "er_annotations" / "recordings_combined_splits.json",
    ]

    missing: list[str] = []
    for p in required_paths:
        if not p.exists():
            missing.append(str(p))
    return missing


def _import_ok(modules: Iterable[str]) -> tuple[list[str], list[str]]:
    ok: list[str] = []
    missing: list[str] = []
    for m in modules:
        try:
            importlib.import_module(m)
            ok.append(m)
        except Exception:
            missing.append(m)
    return ok, missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local sanity checks for the extension notebooks (Step 1–4)."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only run checks; do not create output folders.",
    )
    args = parser.parse_args()

    repo_root = _repo_root_from_cwd()
    os.chdir(repo_root)

    missing_paths = _check_paths(repo_root)
    base_modules = [
        "numpy",
        "torch",
        "sklearn",
        "torcheval",
        "matplotlib",
    ]
    extension_modules = [
        "transformers",
        "scipy",
        "networkx",
        "torch_geometric",
    ]

    _, missing_base = _import_ok(base_modules)
    _, missing_ext = _import_ok(extension_modules)

    if not args.check_only:
        (repo_root / "extension_data").mkdir(exist_ok=True)

    print("=== Extension local check ===")
    print(f"Repo root: {repo_root}")

    if missing_paths:
        print("\nMissing required paths:")
        for p in missing_paths:
            print(f"  - {p}")
    else:
        print("\nRequired paths: OK")

    if missing_base:
        print("\nMissing base Python deps (install via requirements.txt):")
        for m in missing_base:
            print(f"  - {m}")
    else:
        print("\nBase Python deps: OK")

    if missing_ext:
        print(
            "\nMissing extension notebook deps (Step 2–4 may fail without these):"
        )
        for m in missing_ext:
            print(f"  - {m}")
        print("\nSuggested installs (choose what you need):")
        print("  pip install transformers scipy networkx")
        print("  # torch-geometric installation is platform/PyTorch specific")
    else:
        print("\nExtension notebook deps: OK")

    print("\nNext:")
    print("  - Run: extension_step1_localization.ipynb")
    print("  - Then: extension_step2_verification_baseline.ipynb")
    print("  - Then: extension_step3_task_graph_matching.ipynb")
    print("  - Then: extension_step4_gnn_classification.ipynb")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

