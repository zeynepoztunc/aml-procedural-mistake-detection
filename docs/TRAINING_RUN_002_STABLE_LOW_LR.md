# Training Run 002: Stable Baseline (Low LR)

**Date:** January 10, 2026 **Status:** Success (Stable, but Underfitting)
**Objective:** Verify that the "NaN Loss" and "Empty Batch" fixes (Edits
001-004) stabilized the training loop.

## 1. Configuration Changes (vs Run 001)

- **Previous Run Issues:** `Test Loss: NaN`, `Precision: 0.0` (Sub-steps),
  frequent "Empty Batch" crashes.
- **Fixes Applied:**
  - **Data Guard:** Skipping empty inputs/corrupt files.
  - **Normalization:** Standard Scaling + Clipping (-10, 10).
  - **Learning Rate:** Lowered from `1e-3` to `1e-4` (10x slower) to prevent
    gradient explosions.

## 2. Quantitative Results (Epoch 10)

The model is **stable** (no crashes, valid numbers), but **performs poorly**
compared to Run 001.

| Metric        | Run 1 (Unstable) | Run 002 (Stable) | Interpretation                                                          |
| :------------ | :--------------- | :--------------- | :---------------------------------------------------------------------- |
| **Test Loss** | `NaN`            | **2.84**         | **Success.** The fixes worked. We have a valid loss.                    |
| **Step AUC**  | 0.81             | **0.52**         | **Regression.** The model is barely better than random guessing (0.50). |
| **Recall**    | 1.00             | **0.02**         | **Underfitting.** The model is too cautious/slow to learn.              |
| **Precision** | 0.35             | **0.37**         | Comparable.                                                             |

## 3. Analysis: Why did performance drop?

The drop in Step AUC (from 0.81 to 0.52) indicates **Underfitting**.

1. **Learning Rate Too Low:** We reduced the LR to `1e-4` to fix the stability
   issues. While this stopped the NaNs, it also slowed the learning process to a
   crawl. The model likely needs 50-100 epochs at this speed to reach the same
   level.
2. **Clipping Impact:** We clipped features to `[-10, 10]`. If specific EgoVLP
   features rely on high-magnitude values (e.g., 50.0) to signal an "error", we
   might have suppressed that signal.

## 4. Conclusion & Next Steps

**Status:** The Codebase is fixed. The Stability is fixed. **Problem:** The
Hyperparameters are too conservative.

**Action Plan for Run 003:**

1. **Increase Learning Rate:** Bump to `5e-4` (halfway to original).
2. **Increase Epochs:** Run for 20 Epochs to allow convergence.
3. **Relax Clipping:** Widen range to `[-100, 100]` to preserve strong signals.
