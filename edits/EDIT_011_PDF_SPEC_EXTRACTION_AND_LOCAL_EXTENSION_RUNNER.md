# EDIT 011 — PDF Spec Extraction + Local Extension Runner (Jan 11, 2026)

This edit documents changes made during the Codex session to:

1) extract the official project PDF into searchable text, and  
2) make the extension notebooks runnable locally (without Jupyter), including GPU support.

## Summary of code/doc changes

### PDF → text extraction (spec readability)

- Updated `tools/extract_pdf_requirements.py`:
  - Adds UTF-8 stdout reconfigure to avoid Windows `UnicodeEncodeError`.
  - Makes `pypdf` optional and provides a clearer error if neither `pypdf` nor `pdfminer.six` is available.
- Added `extract_pdf_text.py`:
  - Thin wrapper to run `tools/extract_pdf_requirements.py`.
  - Output goes to `tools/_extracted/AML-2025_Mistake_Detection_Project.txt`.
- Updated `README.md`:
  - Documents the PDF extraction command.

### Extension: local runner (no Jupyter) + reproducibility

- Added `tools/notebook_runner.py`:
  - Converts `extension_step*.ipynb` code cells into a runnable `.py`.
  - Skips Colab-only cells (`google.colab`, Drive mounting) and notebook magics (`%...`, `!...`).
  - Injects local defaults (e.g., `IN_COLAB=False`) and forces UTF-8 stdout to avoid Windows encoding crashes.
  - Adds optional runtime overrides for Step 2 via env vars:
    - `EXT_NUM_EPOCHS` (default 30)
    - `EXT_MAX_RECIPES` (default 0 = all)
- Added `tools/run_extension_local.py`:
  - CLI to run extension steps 1–4 and write logs/scripts under `tools/_runs/extension_local/`.
  - Adds guardrails: refuses to run Step 2/3 without `extension_data/step_embeddings_gt.pkl`, and Step 4 without `extension_data/realized_task_graphs.pkl`.
- Added `tools/__init__.py`:
  - Ensures `tools.*` imports work when running the runner as a script.
- Added/updated docs:
  - `docs/RUN_EXTENSION_LOCALLY.md` now documents running the full extension locally and expected outputs in `extension_data/`.
  - `README.md` links to the extension runner docs.

### Small correctness fixes found during audit

- Fixed `core/models/lstm.py`:
  - Removes an unreachable duplicate `return`.
- Updated `core/evaluate.py`:
  - Expands allowed `--backbone` and `--variant` choices so evaluation matches the rest of the codebase (e.g., `egovlp`, `LSTM`).

### “Tasks vs implementation” and evaluation status docs

- Added `docs/SPEC_TASKS_AND_IMPLEMENTATION_STATUS.md`:
  - Maps PDF tasks to repo files and notes remaining gaps.
- Updated `PROJECT_LEARNING_GUIDE.md`:
  - Adds a more complete “Extension” section (what each step does, why, outputs, how metrics are computed, targets, improvement levers).
- Replaced `EVALUATION_REPORT.md`:
  - Updates compliance status to reflect what is actually implemented now (extension runnable locally; error-type analysis exists).

## Local environments created (not tracked in git)

These folders were created on disk to make the extension runnable:

- `aml-procedural-mistake-detection/.venv-ext` (Python 3.12 + CPU torch + PyG)
- `aml-procedural-mistake-detection/.venv-ext-cuda` (Python 3.12 + CUDA torch + PyG)

Reason: the repo root venv uses Python 3.13; PyTorch Geometric deps (e.g. `torch_scatter`) do not ship wheels for 3.13, so Step 4 could not be installed there.

## How to run (GPU recommended)

From `aml-procedural-mistake-detection/`:

- Full extension: `.\.venv-ext-cuda\Scripts\python.exe tools\run_extension_local.py --steps 1 2 3 4`
- PDF extraction: `..\.\.venv\Scripts\python.exe extract_pdf_text.py`

Expected outputs:

- `extension_data/step_embeddings_gt.pkl`
- `extension_data/task_verification_results.json`
- `extension_data/realized_task_graphs.pkl`
- `extension_data/gnn_results.json`

## Known spec-alignment gaps (still open)

- Step 3 uses `distilbert-base-uncased` + projection for task-graph text encoding, not the aligned EgoVLP/PerceptionEncoder text encoder described in the PDF.
- Per-error-type breakdown is implemented as separate analysis artifacts (not emitted by the main ER evaluator `base.py:test_er_model`).
- SlowFast features are supported in code but may not be present locally (data-dependent).

