# Project Evaluation & Gap Analysis

**Date:** January 10, 2026 (updated Jan 11, 2026)  
**Subject:** AML-2025 Mistake Detection Project Implementation Review

This report tracks what is implemented in this repo versus the requirements in `AML-2025_Mistake_Detection_Project.pdf`.

## 1. Requirement Compliance Summary

| Requirement | Status | Evidence / Notes |
| :--- | :--- | :--- |
| Env setup | Implemented | `requirements.txt`, local venv instructions in `README.md` |
| Step 2 baseline V1 (MLP) | Implemented | `core/models/blocks.py` + `base.py` |
| Step 2 baseline V2 (Transformer) | Implemented | `core/models/er_former.py` + `base.py` |
| New baseline (example: LSTM) | Implemented | `core/models/lstm.py` (+ notebooks) |
| Standard metrics (Acc/Prec/Recall/F1/AUC) | Implemented | `base.py:test_er_model` |
| Error-type analysis | Implemented (separate workflow) | `error_type_analysis.ipynb`, `error_type_analysis.md`, `docs/ERROR_TYPE_DISTRIBUTION_2026-01-10.md` |
| New backbone (EgoVLP or PerceptionEncoder) | Partially implemented | EgoVLP supported in code + features present; PerceptionEncoder not integrated |
| Extension Step 1 (localization/step embeddings) | Implemented | `extension_step1_localization.ipynb` writes `extension_data/step_embeddings_gt.pkl` |
| Extension Step 2 (task-verification baselines) | Implemented | `extension_step2_verification_baseline.ipynb` writes `extension_data/task_verification_results.json` |
| Extension Step 3 (graph encoding + matching) | Implemented | `extension_step3_task_graph_matching.ipynb` writes `extension_data/realized_task_graphs.pkl` |
| Extension Step 4 (GNN classification) | Implemented | `extension_step4_gnn_classification.ipynb` writes `extension_data/gnn_results.json` |
| Run extension locally | Implemented | `docs/RUN_EXTENSION_LOCALLY.md` + `tools/run_extension_local.py` |

## 2. Remaining Gaps / Notes

### A. SlowFast features (PDF Step 2.1)

The PDF mentions Omnivore and SlowFast. This repo supports SlowFast in code, but local data currently includes Omnivore; SlowFast features are not present under `data/video/slowfast/`.

### B. PerceptionEncoder backbone (PDF Step 2.3)

PerceptionEncoder is mentioned in the PDF, but there is no PerceptionEncoder-specific integration (feature extraction + dataloader path + input dimension) in the codebase.

### C. Error-type breakdown inside the main ER evaluation (interpretation of PDF Step 2.2a)

You do have error-type analysis implemented, but it is not wired into the main `test_er_model` evaluation output for a single checkpoint (the ER evaluator still reports global metrics only).

### D. Extension Step 3 text encoder alignment (PDF Substep 3)

The PDF suggests encoding task-graph text using an aligned EgoVLP/PE textual encoder. The current Step 3 notebook uses a HuggingFace text model (`distilbert-base-uncased`) and projects to the visual feature dimension. This works, but it is not the same as using the aligned EgoVLP/PE text encoder.

## 3. Next Steps

- Optional: add per-error-type metric breakdown into `base.py:test_er_model` (in addition to the existing analysis notebooks).
- Optional: integrate PerceptionEncoder (if you want full spec coverage beyond EgoVLP).
- Write the final report (8 pages, CVPR template) focusing on the extension, as required by the PDF.

