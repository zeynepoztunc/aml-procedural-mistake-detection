# Edit 001: Robust Data Loading & Feature Normalization

**Date:** January 10, 2026 **File(s) Modified:**
`dataloader/CaptainCookStepDataset.py`

## Description of Changes

1. **Feature Loading Safety:**
   - Modified `_get_video_features` to catch `OSError` and `Exception` during
     `.npz` file loading.
   - If a file is missing or corrupt, the method now returns `None, None`
     instead of crashing the program.

2. **NaN Handling & Normalization:**
   - Added `np.nan_to_num(recording_features)` right after loading to convert
     potential `NaN`s in the source files to 0.0.
   - Implemented **Standard Scaling** (Z-score normalization):
     `(features - mean) / std`.
   - Added outlier clipping: `np.clip(..., -10.0, 10.0)` to prevents extreme
     values from destabilizing gradients.

3. **Batch Filtering:**
   - Updated `__getitem__` to return `None` if features failed to load.
   - Updated `collate_fn` to filter out any `None` entries from the batch list
     before stacking tensors.

## Implementation Reasoning

**Why was this implemented?** During the initial training run, the loss function
frequently returned `NaN` (Not a Number), and the model achieved 0.0 Precision
on sub-steps. This is often caused by:

1. **Corrupt Data:** Some `.npz` files might contain infinity or missing values.
2. **Unscaled Features:** Neural networks struggle if input values are too large
   or vary wildly in scale.
3. **Missing Files:** The code previously crashed or asserted if a specific
   recording ID file was missing.

This edit ensures the data entering the model is clean, normalized, and safe,
improving training stability.
