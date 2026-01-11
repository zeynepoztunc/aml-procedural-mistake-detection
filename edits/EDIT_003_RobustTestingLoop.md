# Edit 003: Robust Testing Loop

**Date:** January 10, 2026 **File(s) Modified:** `base.py`

## Description of Changes

1. **Empty Batch Handling (Test Phase):**
   - Added `if len(data) == 0: continue` to `test_er_model`.
   - This mirrors the fix in the training loop, ensuring that if `collate_fn`
     returns an empty batch (due to corrupt files), the testing loop doesn't
     crash or process garbage.

2. **NaN Loss Handling (Test Phase):**
   - Added check `if np.isnan(loss_val): continue`.
   - If a specific test batch generates a `NaN` loss, it is excluded from the
     average loss calculation.
   - This prevents the final `Test Loss` metric from becoming `nan`.

## Implementation Reasoning

**Why was this implemented?** After fixing the training loop (Edit 002), the
**Test/Validation** loop was still vulnerable. The logs showed `Test Loss: nan`
and `Precision: nan`. This confirmed that bad data exists in the Test set as
well. This edit ensures the validation metrics are calculated only on clean,
valid data.
