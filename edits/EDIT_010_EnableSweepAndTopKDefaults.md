# Edit 010: Enable Threshold Sweep + Top-K Pooling by Default (Val-Only Sweep)

**Date:** January 10, 2026 **File(s) Modified:**
`core/config.py`, `base.py`

## Description of Changes

1. **Enable “Big Gains” Defaults:**
   - Updated defaults in `core/config.py` so a plain `python train_er.py` uses:
     - `sweep_thresholds=True` (with opt-out)
     - `step_pooling=topk` (with `step_topk_frac=0.2`)
     - `early_stop_patience=4` (can be set to 0 to disable)

2. **Opt-Out Flag for Threshold Sweep:**
   - Switched `--sweep_thresholds` to `BooleanOptionalAction`, enabling:
     - `--sweep_thresholds` (enable)
     - `--no-sweep-thresholds` (disable)
   - This prevents getting “stuck” with enabled defaults.

3. **Prevent Test-Set Threshold Tuning (Val-Only Sweep):**
   - Updated `train_model_base` in `base.py` so threshold sweep is applied on
     **validation only**.
   - The selected `best_threshold` from validation is then used to evaluate
     the test set with `sweep_thresholds=False`.

## Implementation Reasoning

**Why was this implemented?** The threshold sweep and pooling strategy are
high-impact improvements, but they were not enabled in the previous run because
the run used default CLI arguments. Enabling them by default makes the “better”
evaluation/training behavior the standard.

Additionally, threshold sweeping on the test set is a form of leakage (you are
optimizing a hyperparameter on the test distribution). Sweeping only on
validation and then evaluating test at the chosen validation threshold keeps
the evaluation procedure clean for reporting.

