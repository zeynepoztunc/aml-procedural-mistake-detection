# Error Type Distribution Analysis (Jan 10, 2026)

## Overview

This analysis summarizes the distribution of error types in the train,
validation, and test splits for the procedural mistake detection dataset.
Understanding this distribution is critical for interpreting model performance
and guiding further experiments.

## Error Type Counts

| Error Type        | Train | Val | Test |
| ----------------- | ----- | --- | ---- |
| Preparation Error | 302   | 42  | 50   |
| Measurement Error | 234   | 46  | 51   |
| Order Error       | 497   | 153 | 144  |
| Timing Error      | 130   | 20  | 27   |
| Technique Error   | 374   | 65  | 59   |
| Temperature Error | 44    | 13  | 8    |
| Missing Step      | 224   | 26  | 34   |
| Other             | 8     | 0   | 0    |

## Key Observations

- **Order Error** and **Technique Error** are the most common error types across
  all splits.
- **Other** errors are extremely rare or absent in validation and test splits.
- All error types are represented in each split, but some (e.g., Temperature,
  Timing, Missing Step) are much less frequent.
- The class imbalance is moderate, with error steps making up about 33-39% of
  each split.

## Implications

- The model may struggle to learn rare error types due to limited examples
  (especially "Other" and "Temperature Error").
- Performance metrics should be interpreted with caution, especially for
  underrepresented error types.
- Future experiments could consider:
  - Focusing on the most common error types for targeted improvement.
  - Data augmentation or oversampling for rare error types.
  - Per-error-type evaluation to identify specific weaknesses.

---

_Generated automatically after running `analyze_error_types.py` on Jan
10, 2026._
