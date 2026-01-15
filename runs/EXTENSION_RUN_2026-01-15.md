# Extension Run (Local) — 2026-01-15

This document records a full local execution of the **extension pipeline (Steps 1–4)** on the `recordings` split artifacts already present in the repo.

## What was run

Pipeline:

1. **Step 1 (GT localization → step embeddings)**: `extension_step1_localization.ipynb`
2. **Step 2 (task verification baselines)**: `extension_step2_verification_baseline.ipynb`
3. **Step 3 (task-graph encoding + matching)**: `extension_step3_task_graph_matching.ipynb`
4. **Step 4 (GNN classification)**: `extension_step4_gnn_classification.ipynb`

Execution was done via the nbconvert runner:

- `python tools/run_extension_notebooks.py --steps step1`
- `python tools/run_extension_notebooks.py --steps step2`
- `python tools/run_extension_notebooks.py --steps step3`
- `python tools/run_extension_notebooks.py --steps step4`

Run logs are stored under `tools/_runs/extension_local_notebooks/`:

- Step 1: `tools/_runs/extension_local_notebooks/20260114_232246/SUMMARY.txt`
- Step 2: `tools/_runs/extension_local_notebooks/20260114_232507/SUMMARY.txt`
- Step 4: `tools/_runs/extension_local_notebooks/20260115_022403/SUMMARY.txt`

## Inputs / expected data layout

- EgoVLP features: `data/features/egovlp/`
- Step annotations: `annotations/annotation_json/step_annotations.json`
- Split file: `er_annotations/recordings_combined_splits.json`

## Outputs (artifacts)

All outputs are written to `extension_data/`:

- Step 1:
  - `extension_data/step_embeddings_gt.pkl`
  - `extension_data/step_localization_stats.png`
- Step 2:
  - `extension_data/task_verification_results.json`
  - `extension_data/task_verification_baselines.png`
- Step 3:
  - `extension_data/realized_task_graphs.pkl`
  - `extension_data/matching_analysis.png`
  - `extension_data/sample_realized_graph.png`
- Step 4:
  - `extension_data/gnn_results.json`
  - `extension_data/gnn_comparison.png`

## Metrics and what they mean

All reported metrics are **mean ± std over Leave-One-Recipe-Out cross-validation folds** (24 folds).

- **Accuracy**: fraction of correct predictions.
- **Precision / Recall / F1**: mistake-class detection quality; **F1** is typically the headline metric with imbalance.
- **AUC**: ranking quality across thresholds.

## Results

### Step 2 — Task Verification baselines (no graphs)

Source: `extension_data/task_verification_results.json`

- `MLP`: Accuracy **0.6189 ± 0.1368**, F1 **0.6600 ± 0.1615**, AUC **0.6648 ± 0.1338**
- `Transformer`: Accuracy **0.6275 ± 0.1227**, F1 **0.6554 ± 0.1452**, AUC **0.6768 ± 0.1501**
- `LSTM`: Accuracy **0.6197 ± 0.1243**, F1 **0.6723 ± 0.1421**, AUC **0.6881 ± 0.1399**

### Step 4 — Graph models (uses realized task graphs)

Source: `extension_data/gnn_results.json`

- `SimplePooling` (non-GNN baseline): Accuracy **0.6407 ± 0.1545**, F1 **0.7028 ± 0.1283**, AUC **0.7541 ± 0.1202**
- `GCN`: Accuracy **0.6674 ± 0.1047**, F1 **0.7062 ± 0.0969**, AUC **0.7819 ± 0.0859**
- `GAT`: Accuracy **0.7501 ± 0.0952**, F1 **0.7533 ± 0.1094**, AUC **0.8299 ± 0.0806**
- `GraphSAGE`: Accuracy **0.7253 ± 0.0979**, F1 **0.7568 ± 0.0916**, AUC **0.8345 ± 0.1010**

**Best overall (this run):** `GraphSAGE` (highest F1 and AUC).

## Notes / caveats

- Step 1 used **ground-truth step boundaries** (the ActionFormer-based Step 1 v2 was not used for this run’s embeddings).
- Step 4 requires **PyTorch Geometric**; on Windows this was run in a separate Python 3.12 environment (`.venv-pyg`) so PyG wheels could be installed.

