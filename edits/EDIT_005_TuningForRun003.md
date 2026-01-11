# Edit 005: Tuning for Run 003 (Batch64, LR+)

**Date:** January 10, 2026 **File(s) Modified:** `core/config.py`,
`dataloader/CaptainCookStepDataset.py`

## Description of Changes

1. **Batch Size Increased:** `32` $\to$ `64`.
   - **Goal:** Faster training iterations and potentially smoother gradient
     estimates.
   - **Validation:** User's 8GB RTX 4060 has sufficient VRAM for these small
     feature tensors.

2. **Epochs Adjustment:** `20` $\to$ `15`.
   - **Goal:** Run 002 (10 Epochs) only got to AUC 0.52. 15 Epochs should be
     enough to see if the curve is rising without waiting too long.

3. **Learning Rate Increased:** `1e-4` $\to$ `5e-4`.
   - **Goal:** Speed up convergence. Run 002 (1e-4) was too slow (Underfitting).
     This is a safe middle ground.

4. **Clipping Relaxed:** `[-10, 10]` $\to$ `[-100, 100]`.
   - **Goal:** The previous clip might have destroyed valuable high-magnitude
     signal features. This wider range preserves outliers while still preventing
     Infinity/NaNs.

## Implementation Reasoning

**Why tune now?** Run 002 was stable (no NaNs) but dumb (AUC 0.50). We are
shifting from "Survival Mode" (fixing crashes) to "Performance Mode" (getting
good results).
