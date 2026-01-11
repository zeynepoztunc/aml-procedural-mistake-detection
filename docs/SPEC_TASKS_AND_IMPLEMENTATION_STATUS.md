# AML-2025 Mistake Detection Project: Tasks vs. This Repo

This document is a quick “what’s required” checklist (from `AML-2025_Mistake_Detection_Project.pdf`) mapped to the current code/notebooks in `aml-procedural-mistake-detection/`.

To regenerate the plain-text spec locally, run:

`..\.\.venv\Scripts\python.exe aml-procedural-mistake-detection\extract_pdf_text.py`

Output is written to `aml-procedural-mistake-detection/tools/_extracted/AML-2025_Mistake_Detection_Project.txt`.

## Step 1 — Literature Review (non-code)

- Read the suggested papers (CaptainCook4D + procedure learning). No repo implementation required.

## Step 2 — Mistake Detection baselines (SupervisedER)

- Download pre-extracted features (Omnivore/SlowFast) and place under `aml-procedural-mistake-detection/data/` (see `aml-procedural-mistake-detection/README.md`).
- Reproduce baseline V1 (MLP) + V2 (Transformer), reporting Accuracy/Precision/Recall/F1/AUC.
  - Implemented: training pipeline in `aml-procedural-mistake-detection/train_er.py` and evaluation in `aml-procedural-mistake-detection/core/evaluate.py`.
  - Models: `aml-procedural-mistake-detection/core/models/blocks.py` (MLP) and `aml-procedural-mistake-detection/core/models/er_former.py` (Transformer).
- Analyze performance by error type.
  - Implemented: `aml-procedural-mistake-detection/analyze_error_types.py`, `aml-procedural-mistake-detection/error_type_analysis.ipynb`, and `aml-procedural-mistake-detection/error_type_analysis.md`.
- Propose a new baseline (example given: RNN/LSTM).
  - Implemented: `aml-procedural-mistake-detection/core/models/lstm.py` plus notebooks `aml-procedural-mistake-detection/train_lstm_baseline.ipynb` and `aml-procedural-mistake-detection/train_lstm_baseline_v2.ipynb`.
- Extend to a new feature backbone (suggested: EgoVLP or PerceptionEncoder).
  - Partially implemented:
    - EgoVLP is wired into the dataloaders and model input dims (`aml-procedural-mistake-detection/dataloader/CaptainCookStepDataset.py`, `aml-procedural-mistake-detection/core/models/blocks.py`, `aml-procedural-mistake-detection/constants.py`).
    - Colab feature extraction notebook exists: `aml-procedural-mistake-detection/colab_feature_extraction.ipynb`.
  - Not implemented in-code: PerceptionEncoder-specific support (only mentioned in the spec).

## Extension — From Mistake Detection to Task Verification

The spec’s suggested extension is a 3-stage pipeline: step localization → step-to-graph matching → graph classification.

- Substep 1: Recipe step localization (ActionFormer or HiERO-style).
  - Implemented (notebooks): `aml-procedural-mistake-detection/extension_step1_localization.ipynb`, `aml-procedural-mistake-detection/extension_step1_v2_actionformer.ipynb`.
- Substep 2: Simple task-verification baselines (binary recipe-level correctness; leave-one-recipe-out evaluation).
  - Implemented (notebook): `aml-procedural-mistake-detection/extension_step2_verification_baseline.ipynb`.
- Substep 3: Task-graph encoding + step matching (EgoVLP/PE text encoder + Hungarian matching).
  - Implemented (notebook): `aml-procedural-mistake-detection/extension_step3_task_graph_matching.ipynb`.
- Substep 4: Classify the realized task-graph with a GNN (optionally DAG-aware layers).
  - Implemented (notebook): `aml-procedural-mistake-detection/extension_step4_gnn_classification.ipynb`.

## What’s still “missing” (if you want full spec coverage)

- A PerceptionEncoder backbone integration (feature extraction + loader support + model input dimension).
- Optional: converting the extension notebooks into a runnable, versioned Python pipeline (CLI/scripts + configs) if you want reproducibility outside Colab.

