# Training Run 004: Step-Level Metrics Fixed (Re-Run of Tuned Config)

**Date:** January 10, 2026\
**Status:** Success\
**Objective:** Re-run the tuned training setup from Run 003 after implementing
Edit 008 (correct step-level aggregation). Confirm that Step-Level metrics are
valid (no `NaN`, no all-zero collapse) even when `test_batch_size > 1`.

## 1. Configuration

- **Task:** Error Recognition (Binary)
- **Split:** `recordings` (`recordings_combined_splits.json`)
- **Backbone / Variant:** EgoVLP + Transformer (codebase defaults)
- **Epochs:** 15
- **Batch Size:** 64 (train)
- **Test Batch Size:** > 1 (observed from ~22 validation batches)
- **Learning Rate:** 5e-4
- **Loss:** `BCEWithLogitsLoss(pos_weight=5.0)` (from config)
- **Evaluation Threshold:** 0.6 (from config default)
- **Edits Applied:** Edit 001–008 (notably Edit 008 step boundary fix)

## 2. Key Outcome

Step-Level evaluation is now **correct** and stable. Unlike Run 003 (where
Step-Level metrics were `0.0`/`NaN` due to batching/aggregation issues), this
run produces consistent Step-Level Precision/Recall/F1/AUC across all epochs.

## 3. Results Summary

### Best Observed Test Step-Level (by F1)

- **Epoch 10 (Test Step-Level):**
  - Precision: **0.449**
  - Recall: **0.941**
  - F1: **0.608**
  - AUC: **0.870**
  - PR-AUC: **0.527**

### Best Observed Test Step-Level (by AUC)

- **Epoch 15 (Test Step-Level):**
  - Precision: **0.465**
  - Recall: **0.830**
  - F1: **0.596**
  - AUC: **0.875**
  - PR-AUC: **0.542**

### Best Observed Validation Step-Level (for checkpoint selection)

- **Best Val F1:** Epoch 8, Val Step F1 **0.529**
- **Best Val AUC:** Epoch 9, Val Step AUC **0.843**

## 4. Interpretation

- **High Recall, Moderate Precision (Step-Level):** The model is very sensitive
  to mistakes (good at catching errors) but still produces false positives.
  This is consistent with using a relatively low decision threshold and a
  higher positive-class weight.
- **Sub-Step vs Step:** Sub-step AUC stays around ~0.56–0.61 while Step-level
  AUC rises to ~0.82–0.87. Aggregating sub-step signals into a step prediction
  is providing a strong boost, which is expected for this task.
- **Checkpoint choice:** Test metrics peak around epochs ~9–10, while loss keeps
  moving (test loss can increase even when F1/AUC stay good). For reporting, a
  checkpoint should be selected using **validation** Step metrics (F1/AUC),
  not loss.

## 5. Improvement Suggestions (Next Runs)

1. **Tune the threshold on validation (not fixed 0.6):**
   - Sweep thresholds (e.g., 0.2–0.9) and pick the best **Val Step F1** or a
     precision-recall tradeoff that matches the project goal (fewer false
     alarms vs. fewer missed mistakes).

2. **Revisit class-imbalance handling:**
   - Try smaller `pos_weight` (e.g., 2–4) or use a sampler/oversampling of
     error steps instead of only loss reweighting.
   - Consider focal loss if precision remains low.

3. **Early stopping + best-checkpoint selection:**
   - Stop when Val Step F1/AUC plateaus (peaks around epochs ~8–10 here).
   - Save/restore best model by Val Step F1 (or PR-AUC) rather than by loss.

4. **Improve step aggregation:**
   - Replace mean pooling with max/top-k pooling or attention pooling for the
     step probability to reduce dilution when errors are brief.

