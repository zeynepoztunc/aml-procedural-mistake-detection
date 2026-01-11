## Edit 007: Increase Positive Class Weight for Loss Function

**Date:** 2026-01-10 **Files Modified:** core/config.py

### What was changed?

- Increased the default value of `pos_weight` in the argument parser from `2.5`
  to `5.0`.

### Why?

- The model was consistently predicting the majority class ("Normal") and
  failing to detect errors, despite a moderate class imbalance (error steps
  ~33-39%).
- Increasing the positive class weight in `BCEWithLogitsLoss` will penalize
  missed error predictions more, encouraging the model to learn to detect
  errors.

### How to use:

- This change will take effect automatically for new training runs unless
  overridden by a command-line argument.

---

_This edit was made as part of a systematic effort to address class imbalance
and improve recall for error detection._
