# Edit 002: Robust Training Loop (NaN Guard)

**Date:** January 10, 2026 **File(s) Modified:** `base.py`

## Description of Changes

1. **Empty Batch Handling:**
   - In `train_epoch`, added a check `if len(data) == 0: continue`.
   - This handles cases where `collate_fn` (from Edit 001) returns an empty
     batch because all samples in that batch were corrupt/missing.

2. **Input Data Guard:**
   - Added `if torch.isnan(data).any(): ... continue`.
   - Before feeding data to the model, we check if it contains any corrupt
     values. If so, we skip the batch and log a warning.

3. **Loss Guard:**
   - Added `if torch.isnan(loss).any(): ... continue`.
   - After calculating loss, we check if it is valid. if the loss is `NaN`, we
     skipping the backward pass (weight update) to prevent poisoning the model's
     weights.

## Implementation Reasoning

**Why was this implemented?** Even with clean data loading, unstable training
dynamics (exploding gradients) can sometimes produce `NaN` losses.

- **Consequence of inaction:** If a single batch produces a `NaN` loss, the
  gradients become `NaN`. When the optimizer updates the weights, _all_ model
  weights become `NaN`. The model is effectively "dead" and will never recover
  (producing random or constant outputs forever).
- **Benefit:** By skipping these specific bad batches, the training run can
  survive occasional instabilities and continue learning from the good data.
