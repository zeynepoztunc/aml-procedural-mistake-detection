# Training Run 003: Tuned Hyperparameters & "Lazy Trap" Escape

**Date:** Jan 10, 2026\
**Status:** Completed (Pending Re-Evaluation)\
**Configuration:**

- **Batch Size:** 64 (Training), 32 (Testing - _Bug detected, see below_)
- **Epochs:** 15
- **Learning Rate:** 5e-4
- **Clip Range:** [-100, 100]

## Outcomes

### 1. Training Dynamics (The "Lazy Trap")

- **Phase 1 (Epochs 1-3):** The model exhibited high variance in accuracy (0.68
  -> 0.47 -> 0.77). This confirmed it was "exploring" the loss landscape rather
  than collapsing to a trivial solution (guessing "Normal" for everything).
- **Phase 2 (Convergence):** Training loss dropped significantly from ~1.51 to
  ~0.73, indicating successful learning on the training set.

### 2. Metric Analysis

- **Sub-Step AUC (0.61):**
  - Significant improvement over Run 002 (0.52).
  - **Meaning:** The model has successfully learned to distinguish individual
    video segments (frames/sub-steps) associated with errors better than random
    chance.

- **Step-Level Metrics (NaN / Zero):**
  - **Issue:** All Step metrics (AUC, Precision, Recall) reported `0.0` or
    `NaN`.
  - **Root Cause:** A configuration bug. `test_batch_size` was set to 32. The
    evaluation code in `base.py` aggregates _all items in a batch_ into a single
    "Step" prediction.
  - **Effect:** The code merged 32 different steps into one data point,
    destroying the signal and resulting in a single-class target array
    (`y_true`), which causes `sklearn` to fail with `UndefinedMetricWarning`.

## Next Steps

1. **Bug Fix:** Hardcoded `test_batch_size = 1` in `core/config.py` to ensure
   the evaluation loop processes one step at a time.
2. **Validation:** Running a standalone evaluation script
   (`evaluate_checkpoint.py`) on the trained checkpoint to recover the _true_
   Step-Level metrics.
