# Project Evaluation & Gap Analysis

**Date:** January 10, 2026
**Subject:** AML-2025 Mistake Detection Project Implementation Review

## 1. Requirement Compliance Summary

Based on the `AML-2025_Mistake_Detection_Project.pdf` and codebase analysis.

| Requirement | Status | Detailed Observation |
| :--- | :--- | :--- |
| **Env Setup** | ✅ | Virtual env active, requirements listed. |
| **Baseline V1 (MLP)** | ✅ | Implemented in `core.models.blocks.MLP` + `base.py`. |
| **Baseline V2 (Transformer)** | ✅ | Implemented in `core.models.er_former.ErFormer` + `base.py`. |
| **New Baseline (LSTM)** | ✅ | Implemented in `core.models.lstm` (referenced in `base.py`). |
| **Feature Support (EgoVLP)** | ✅ | `EGOVLP` constant and dataloader support present. |
| **Standard Metrics** | ✅ | Acc, Prec, Recall, F1, AUC computed in `test_er_model`. |
| **Error Type Analysis** | ❌ **MISSING** | Code loads error types but aggregates metrics globally. No per-category F1/AUC output. |
| **Extension: Localization** | ✅ | Notebooks for Step 1 (ActionFormer) present. |
| **Extension: Verification** | ✅ | Notebooks for Step 2 (Verification baseline) present. |
| **Extension: Graph Match** | ✅ | Notebooks for Step 3 (Graph Matching) present. |
| **Extension: GNN Classif.** | ✅ | Notebooks for Step 4 (GNN) present. |

## 2. Identified Gaps

### Critical Gap: Missing Per-Error-Type Analysis
**Requirement:** "You should analyze the performance of the model on different error types." (PDF Page 4, Item 2a).

**Current State:**
- `test_er_model` in `base.py` calculates metrics on flattened arrays `all_targets` vs `all_outputs`.
- `CaptainCookStepDataset` correctly loads error categories (Temperature, Timing, etc.) into `_error_step_dict` and `recording_step_dictionary`.
- However, these specific error labels are NOT passed through the `DataLoader` or used in `test_er_model` to stratify the results.

### Minor Observations
- **PDF Extraction:** Failed initially with `pypdf` due to font bbox issues; fixed with `pdfminer.six` fallback.
- **Empty Steps:** `test_er_model` has a fallback for empty steps (`if len(step_output) == 0`) but this might mask data loading issues if frequent.

## 3. Suggested Fixes (Plan)

Do NOT modify code yet (per user instruction), but the following changes would be required:

1.  **Modify `collate_fn` / `CaptainCookStepDataset`**: Ensure `error_category_labels` are returned as part of the batch meta-data (or a separate tensor) during testing.
2.  **Update `test_er_model` in `base.py`**:
    -   Accept error category labels from the dataloader.
    -   Inside the metric calculation block, iterate through unique error categories (Temperature, Timing, Preparation, etc.).
    -   Filter `all_targets` and `all_outputs` by category and compute Precision/Recall/F1/AUC for each subset.
    -   Print/Log these per-category metrics alongside the global ones.
3.  **Update `save_results_to_csv`**: Add columns to the results CSV to persist these per-category metrics.

## 4. Next Steps
- User should authorize the implementation of the per-error-type analysis logic.
- Verify the extension notebooks run end-to-end (currently only static analysis was performed).
