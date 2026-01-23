# Root file categorization (organization plan)

This document explains how root-level files under `aml-procedural-mistake-detection/` are categorized and where they should live.

## Keep at repo root (import paths / entrypoints)

- `base.py` (core entry glue)
- `constants.py`
- `train_er.py`
- `requirements.txt`
- `README.md`, `LICENSE`, `.gitignore`, `.gitmodules`

## Move to `notebooks/` (experiments / exploration)

- `colab_feature_extraction.ipynb`
- `colab_quickstart.ipynb`
- `error_type_analysis.ipynb`
- `extension_step1_actionformer.ipynb`
- `extension_step1_pipeline.ipynb`
- `extension_step2_verification_baseline.ipynb`
- `extension_step3_task_graph_matching.ipynb`
- `extension_step4_gnn_classification.ipynb`
- `old-version-train_egovlp_baseline.ipynb`
- `train_lstm_baseline.ipynb`
- `train_lstm_baseline_v2.ipynb`

## Move to `docs/analysis/` (writeups)

- `error_type_analysis.md`
- `LSTM_results_analysis.md`
- `LOCAL_BRANCH_COMPARISON.md`

## Move to `docs/spec/` (project spec / PDFs)

- `AML-2025_Mistake_Detection_Project.pdf`

## Move to `assets/figures/` (static images)

- `gnn_oversmoothing.png`

## LaTeX report

- LaTeX sources live under `latex_report/`.
- Root `report.*` files are legacy build artifacts and can be cleaned up with `latex_report/cleanup_root_report_files.ps1` once no viewer is locking them.

## Status

Copies of the artifacts have been placed in the folders above.
If Windows/VSCode has files open, moving/deleting the originals may fail; close open tabs/previews and then delete the root duplicates.

