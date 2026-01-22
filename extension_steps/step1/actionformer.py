from __future__ import annotations

import sys
from pathlib import Path


def _ensure_import_root() -> None:
    # Pipeline wiring: allow running this file directly by ensuring the package root is importable.
    # Allow running this file directly:
    #   python aml-procedural-mistake-detection/extension_steps/step1/actionformer.py
    # by adding `aml-procedural-mistake-detection/` to sys.path.
    pkg_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(pkg_root))


_ensure_import_root()

from extension_steps.step1.actionformer.main import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
