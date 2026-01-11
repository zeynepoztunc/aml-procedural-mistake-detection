from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable


_MAGIC_LINE_RE = re.compile(r"^\s*[!%]")
_ASSIGN_INT_RE = re.compile(r"^(\s*[A-Z_]+)\s*=\s*(\d+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class NotebookRunResult:
    notebook: Path
    generated_script: Path
    log_path: Path
    returncode: int


def _iter_code_cells(notebook_json: dict) -> Iterable[tuple[int, str]]:
    for idx, cell in enumerate(notebook_json.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        if isinstance(src, list):
            yield idx, "".join(src)
        else:
            yield idx, str(src)


def _should_skip_cell(code: str) -> bool:
    lowered = code.lower()
    if "google.colab" in lowered:
        return True
    if "drive.mount" in lowered:
        return True
    return False


def sanitize_notebook_to_script(notebook_path: Path) -> str:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))

    lines: list[str] = []
    lines.append(f"# Auto-generated from: {notebook_path.name}")
    lines.append("from __future__ import annotations")
    lines.append("")
    # Notebooks assume a Colab bootstrap cell that defines these; provide safe local defaults.
    lines.append("IN_COLAB = False")
    lines.append('DRIVE_BASE = ""')
    lines.append("")
    lines.append("# Avoid Windows console UnicodeEncodeError (cp1252) for symbols/emoji printed in notebooks.")
    lines.append("import os")
    lines.append("import sys")
    lines.append("try:")
    lines.append("    sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[attr-defined]")
    lines.append("except Exception:")
    lines.append("    pass")
    lines.append("")

    for cell_idx, code in _iter_code_cells(nb):
        if not code.strip():
            continue
        if _should_skip_cell(code):
            lines.append(f"# --- skipped cell {cell_idx} (colab-only) ---")
            lines.append("")
            continue

        cell_lines = code.splitlines()
        kept: list[str] = []
        for ln in cell_lines:
            if _MAGIC_LINE_RE.match(ln):
                continue
            kept.append(ln)

        if not any(l.strip() for l in kept):
            continue

        lines.append(f"# --- cell {cell_idx} ---")
        lines.extend(kept)
        lines.append("")

    lines.append('if __name__ == "__main__":')
    lines.append("    pass")
    lines.append("")
    script_text = "\n".join(lines)
    return _apply_local_overrides(notebook_path, script_text)


def _apply_local_overrides(notebook_path: Path, script_text: str) -> str:
    """
    Make notebooks runnable on a laptop by allowing runtime overrides via env vars.

    This keeps the notebooks "as-is" but provides escape hatches for extremely slow
    default settings (e.g. leave-one-recipe-out with many epochs).
    """
    stem = notebook_path.stem

    if stem == "extension_step2_verification_baseline":
        # Allow: EXT_NUM_EPOCHS=3, EXT_MAX_RECIPES=5 for a quick smoke run.
        script_text = re.sub(
            r"^NUM_EPOCHS\s*=\s*\d+\s*$",
            'NUM_EPOCHS = int(os.getenv("EXT_NUM_EPOCHS", "30"))',
            script_text,
            flags=re.MULTILINE,
        )

        marker_line = 'print(f"\\nNumber of unique recipes: {len(recipe_groups)}")'
        if marker_line in script_text:
            inject = (
                'MAX_RECIPES = int(os.getenv("EXT_MAX_RECIPES", "0"))\n'
                "if MAX_RECIPES > 0:\n"
                "    recipe_groups = dict(list(recipe_groups.items())[:MAX_RECIPES])\n"
            )
            script_text = script_text.replace(marker_line, inject + marker_line, 1)

    return script_text


def run_notebook_via_script(
    notebook_path: Path,
    *,
    workdir: Path,
    out_dir: Path,
    timeout_s: int | None = None,
    dry_run: bool = False,
) -> NotebookRunResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    script_text = sanitize_notebook_to_script(notebook_path)

    script_path = out_dir / f"{notebook_path.stem}.py"
    script_path.write_text(script_text, encoding="utf-8")

    log_path = out_dir / f"{notebook_path.stem}.log"
    if dry_run:
        log_path.write_text("dry-run: notebook was not executed\n", encoding="utf-8")
        return NotebookRunResult(
            notebook=notebook_path,
            generated_script=script_path,
            log_path=log_path,
            returncode=0,
        )

    proc = subprocess.run(
        [sys.executable, "-u", str(script_path)],
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    return NotebookRunResult(
        notebook=notebook_path,
        generated_script=script_path,
        log_path=log_path,
        returncode=int(proc.returncode),
    )
