# Training Run 006: Top-K Pooling + Val Threshold Sweep + Early Stopping

**Date:** January 10, 2026\
**Status:** Completed (Early Stopped)\
**Objective:** Enable the “big gains” knobs from Edit 009/010:
top-k pooling for step aggregation, validation-only threshold sweep, and early
stopping based on validation Step-F1.

## 1. Command Used

```
python train_er.py --ckpt_directory checkpoints --sweep_thresholds --step_pooling topk --step_topk_frac 0.2 --early_stop_patience 4
```

## 2. Configuration Used (from console print)

- **Task:** `error_recognition`
- **Backbone:** `egovlp`
- **Variant:** `Transformer`
- **Split:** `recordings`
- **Batch Size:** 64 (train)
- **Test Batch Size:** 32
- **Epochs (max):** 15
- **LR:** 5e-4
- **Weight Decay:** 1e-3
- **pos_weight:** 5.0
- **Base threshold:** 0.6

**Enabled knobs**

- **Step pooling:** `topk`
- **Top-k fraction:** 0.2
- **Threshold sweep:** enabled on **validation** (`sweep_min=0.1`, `sweep_max=0.9`, `sweep_step=0.05`)
- **Best checkpoint metric:** `f1`
- **Early stop patience:** 4

## 3. What Happened

1. **Validation threshold sweep ran** and reported a `best_threshold` as part of
   the printed Step-Level metrics for the validation phase.
2. **Test was evaluated using the validation-chosen threshold** (no test-set
   threshold tuning), per Edit 010.
3. Training **early stopped at epoch 10**:
   - `Early stopping at epoch 10 (best epoch: 6, f1=0.533333).`
   - Interpretation: Val Step-F1 peaked at epoch 6 and did not improve for 4
     consecutive epochs.

## 4. Quantitative Results (Captured From Output)

### Validation (Epoch 10)

- **Val Sub-Step:** F1 **0.4653**, AUC **0.5678**, PR-AUC **0.3470**
- **Val Step-Level (after sweep):**
  - Precision: **0.3506**
  - Recall: **0.9829**
  - F1: **0.5169**
  - AUC: **0.8198**
  - PR-AUC: **0.3544**
  - **best_threshold:** **0.75**

### Test (Epoch 10, using Val-chosen threshold)

- **Test Sub-Step:** F1 **0.5312**, AUC **0.5947**, PR-AUC **0.4525**
- **Test Step-Level:**
  - Precision: **0.4259**
  - Recall: **1.0000**
  - F1: **0.5973**
  - AUC: **0.8635**
  - PR-AUC: **0.4746**

## 5. Interpretation

- **Recall remains extremely high** at the selected threshold, while precision
  is moderate. This indicates the model is still biased toward “flagging”
  mistakes, but the threshold sweep provides a systematic way to control that
  tradeoff on validation.
- **Early stopping is working as intended:** the run automatically stopped once
  validation Step-F1 stopped improving, preventing extra epochs beyond the
  validation peak.

## 6. Next Improvements

1. **Log the best epoch metrics explicitly:** store `best_threshold` and the
   full Val/Test metrics for the best epoch (epoch 6) to avoid relying on the
   final epoch printout.
2. **Try alternative `step_topk_frac`:** 0.1, 0.3, 0.5 can shift sensitivity to
   brief errors vs. longer errors.
3. **Tune `pos_weight`:** reducing it (e.g., 3–4) often increases precision if
   false positives are still too frequent.

