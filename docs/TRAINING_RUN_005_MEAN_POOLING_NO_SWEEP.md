# Training Run 005: Mean Pooling (No Threshold Sweep)

**Date:** January 10, 2026\
**Status:** Completed\
**Objective:** Train the EgoVLP+Transformer baseline on the `recordings` split
with stable evaluation (Edits 001–009). This run did **not** enable the planned
threshold sweep / top-k pooling flags (see notes).

## 1. Configuration Used (from console print)

**Data/Model**

- Task: `error_recognition` (binary)
- Backbone: `egovlp`
- Variant: `Transformer`
- Split: `recordings`
- Modality: `video`

**Training**

- `batch_size`: 64
- `num_epochs`: 15
- `lr`: 5e-4
- `weight_decay`: 1e-3
- `pos_weight`: 5.0
- `best_metric`: `f1` (used for best checkpoint selection)
- `early_stop_patience`: 0 (disabled)

**Evaluation**

- `test_batch_size`: 32 (OK with Edit 008 step-boundary fix)
- `threshold`: 0.6
- `step_pooling`: `mean`
- `sweep_thresholds`: False (disabled)

## 2. What Happened

- Step-level metrics are valid and stable with `test_batch_size=32` because the
  evaluation code now preserves per-step boundaries (Edit 008).
- The “big gains” knobs from Edit 009 (threshold sweep + top-k pooling) were
  **not enabled** in this run, so results reflect the fixed-threshold, mean
  pooling baseline (`threshold=0.6`, `step_pooling=mean`).

## 3. Quantitative Results (Step-Level)

### Best Validation Step-Level (by F1)

- **Epoch 8 (Val Step-Level):**
  - Precision: **0.3870**
  - Recall: **0.8632**
  - F1: **0.5344**
  - AUC: **0.8311**
  - PR-AUC: **0.4119**

### Best Test Step-Level (by F1)

- **Epoch 10 (Test Step-Level):**
  - Precision: **0.4500**
  - Recall: **0.9333**
  - F1: **0.6072**
  - AUC: **0.8696**
  - PR-AUC: **0.5253**

### Best Test Step-Level (by AUC)

- **Epoch 15 (Test Step-Level):**
  - Precision: **0.4689**
  - Recall: **0.8370**
  - F1: **0.6011**
  - AUC: **0.8767**
  - PR-AUC: **0.5476**

## 4. Sub-Step Notes (High-Level)

- Sub-step metrics are consistently lower than step metrics (expected), e.g.
  around epoch 8: Test Sub-Step F1 ≈ **0.540**, AUC ≈ **0.605**.
- Step aggregation boosts performance substantially (step AUC reaching ~0.87).

## 5. Interpretation

- The model trends toward **high recall** with **moderate precision** at the
  default threshold (0.6), meaning it catches most mistakes but still triggers
  false alarms.
- Val Step F1 peaks around **epoch 8**, while later epochs fluctuate. This
  supports selecting checkpoints by **validation Step F1** (and optionally
  enabling early stopping).

## 6. Improvement Suggestions (Next Run)

1. **Enable threshold sweep on validation (`--sweep_thresholds`)**
   - This usually yields the biggest immediate jump in Step F1 because the
     precision/recall tradeoff is threshold-sensitive.

2. **Switch to top-k pooling (`--step_pooling topk --step_topk_frac 0.2`)**
   - Helps when errors are brief within a step (mean pooling can dilute the
     signal).

3. **Enable early stopping on Val Step F1 (`--early_stop_patience 4`)**
   - This run’s best Val F1 occurs at epoch ~8; early stopping saves time and
     reduces over-training past the validation peak.
