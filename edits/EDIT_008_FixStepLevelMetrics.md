# Edit 008: Fix Step-Level Metrics Aggregation & PR-AUC Computation

**Date:** January 10, 2026 **File(s) Modified:**
`dataloader/CaptainCookStepDataset.py`, `base.py`

## Description of Changes

1. **Preserve Step Boundaries in Batching:**
   - Updated `collate_fn` in `dataloader/CaptainCookStepDataset.py` to return a
     third value: `step_lengths`.
   - `step_lengths` stores the number of sub-steps (seconds) contributed by
     each dataset item in the batch, enabling reconstruction of per-step
     segments after concatenation.

2. **Make Train/Test Loops Compatible with New Collate Output:**
   - Updated training loops in `base.py` to unpack batches defensively
     (`data, target = batch[0], batch[1]`) so the extra `step_lengths` field
     does not break training.
   - Updated `test_er_model` to read `step_lengths` when present.

3. **Correct Step-Level Aggregation (Per Step, Not Per Batch):**
   - Replaced step-level slicing based on `test_step_start_end_list` (which was
     effectively batching-level) with reconstruction using `step_lengths`.
   - This ensures each step’s prediction is computed from the correct range of
     sub-step outputs/targets.

4. **Fix Metric Edge Cases and PR-AUC Inputs:**
   - Added `zero_division=0` for precision/recall/F1 to avoid warnings when
     there are no positives or no predicted positives.
   - Guarded ROC-AUC computation when only one class exists in `y_true`
     (returns `nan` instead of throwing).
   - Updated PR-AUC to be computed from probabilities (`all_*_outputs`) instead
     of hard thresholded labels (`pred_*_labels`).

5. **Fix Minor Step-Level Logic Bugs:**
   - Corrected the length check from `if start - end > 1` to `if end - start > 1`.
   - Relaxed the step target rule from `np.mean(step_target) > 0.95` to
     `np.mean(step_target) > 0.5` to better reflect “step is erroneous if any
     meaningful portion of its sub-steps are labeled erroneous”.

## Implementation Reasoning

**Why was this implemented?** The training logs showed:

- Step-level `precision/recall/f1 = 0.0` while sub-step metrics looked
  reasonable.
- Warnings like “Only one class is present in y_true. ROC AUC score is not
  defined”, and “Recall is ill-defined … due to no true samples”.

This pointed to an evaluation issue rather than purely a modeling failure.
`test_er_model` was aggregating step predictions using boundaries derived from
the DataLoader iteration, which depends on `test_batch_size`. When
`test_batch_size > 1`, this mixes multiple steps into a single “step” segment
and can easily collapse the step targets to a single class (especially with a
very strict `> 0.95` labeling rule). The result is misleading step-level
metrics (all zeros, `nan` AUC).

By preserving per-step boundaries in `collate_fn` and reconstructing step
segments using `step_lengths`, step-level evaluation becomes consistent and
independent of the batch size, and PR-AUC becomes meaningful by using
probability scores instead of thresholded predictions.

