# Edit 006: Fix Config Argument Overwrites

**Date:** January 10, 2026 **File(s) Modified:** `core/config.py`, `base.py`

## Description of Changes

1. **Config Parser Defaults Updated:**
   - The `Config` class was using `argparse` to parse command line arguments.
   - Even though I updated the class variables in Edit 005, the `argparse`
     defaults (`batch_size=1`, `epochs=10`) were overwriting them at runtime
     because `self.__dict__.update(self.args)` was called.
   - **Fix:** Updated the default values inside `setup_parser()` to match our
     desired tuning (Batch 64, Epochs 15, LR 5e-4).

2. **Test DataLoader Batch Size:**
   - In `base.py`, the test loader was hardcoded to `batch_size=1`.
   - **Fix:** Changed it to use `config.test_batch_size` (set to 32) so
     validation runs faster.

## Implementation Reasoning

**Why did Run 002 look like it ignored my changes?** Because the code
prioritized the `argparse` defaults over the class attributes. This edit ensures
the actual runtime configuration matches our tuning plan.
