# Training Run 001: Baseline EgoVLP Test

**Date:** January 10, 2026\
**Status:** Success (Partial Run)\
**Objective:** Verify local environment setup, data loading, and training
pipeline functionality using the EgoVLP backbone.

## 1. System Configuration

- **Hardware:** Local Machine (Windows 11, NVIDIA RTX 4060 8GB)
- **Environment:** Python 3.13 (.venv), PyTorch 2.6.0+cu118
- **Codebase Branch:** `NadimDev2`

## 2. Model & Data Configuration

- **Task:** Error Recognition (Binary Classification: Correct vs. Mistake)
- **Backbone:** `EgoVLP` (Pre-extracted features)
- **Input Features:** `.npz` files located in `data/features/egovlp/`
- **Model Architecture:** Transformer (Baseline V2)
- **Hyperparameters:**
  - Batch Size: 32
  - Segment Length: 1s
  - Optimization: Adam
  - Loss Function: Binary Cross Entropy
  - Workers: 0 (Single-process for Windows compatibility)

## 3. Execution Performance

- **Epoch Time:** ~1 minute 5 seconds per training epoch.
- **Inference Time:** ~5 seconds for validation/testing.
- **Total Duration:** ~15 minutes (for initial 10 epoch observation).

## 4. Quantitative Results (At Epoch 10)

The model showed promising results on the **Step-Level** classification
(identifying if a step contains an error), but failed at **Sub-Step**
localization (identifying _when_ exactly the error occurred).

### Step-Level Metrics (Test Set)

| Metric        | Score     | Notes                                                                 |
| :------------ | :-------- | :-------------------------------------------------------------------- |
| **AUC**       | **0.837** | Strong performance for a baseline.                                    |
| **Accuracy**  | **71.2%** |                                                                       |
| **F1 Score**  | **0.583** |                                                                       |
| **Precision** | 0.412     |                                                                       |
| **Recall**    | 1.000     | Model is very sensitive (catches all errors) but has false positives. |

### Sub-Step Level Metrics

- **Precision:** 0.0
- **Recall:** 0.0
- _Observation:_ The model is likely predicting "Correct" for every single
  frame, or failing to converge on the fine-grained temporal features.

## 5. Issues & Warnings Observed

During training, the following warnings were frequent:

1. **NaN Loss Detected:** multiple occurrences
   (`Warning: NaN loss detected at epoch 10...`).
2. **Empty Batch Encountered:** This suggests some data loading issues or
   filtered out corrupt feature files.

## 6. Conclusion

The system is fully operational. The `EgoVLP` features are loading correctly,
and the Transformer model is learning high-level patterns (good step-level AUC).

**Next Steps:**

1. Investigate `NaN` loss causes (likely unnormalized features or infinite
   values in `.npz` files).
2. Implement "Per-Error-Type" analysis to understand which errors are being
   caught.

## 7. Learning: Concepts & Interpretation

### Interpretation of Results

- **High Recall (1.0) / Low Precision (0.412):**
  - **Recall:** "Did we catch all the mistakes?" A Recall of 1.0 means the model
    **didn't miss a single error**. It successfully flagged every instance of a
    mistake.
  - **Precision:** "When we flagged a mistake, was it actually a mistake?" A
    Precision of 0.412 means that for every 10 times the model shouted "Error!",
    it was only right about 4 times. 6 times it was a false alarm.
  - **Trade-off:** This is typical for an early baseline. The model is being
    "safe" by calling everything an error to ensure it doesn't miss any, but it
    hasn't learned to be specific yet.

### What are "Epochs"?

- An **Epoch** represents one complete cycle through the entire training
  dataset.
- The model doesn't learn everything in one glance. It needs to see the examples
  multiple times.
- **Process:** During one epoch, the model looks at every video in the dataset,
  attempts to predict errors, checks the answer key (Ground Truth), and adjusts
  its internal math (Weights) to be slightly more accurate next time.
- **In this run:** We ran 10 Epochs. This means the model studied the entire
  coursework 10 times over before we tested it.

### Who decides "Right" and "Wrong"? (The Judge)

The "Judge" is a mathematical formula called the **Loss Function**.

- **In this project:** We use `BCEWithLogitsLoss` (Binary Cross Entropy).
- **How it works:**
  - It looks at the **Prediction** (e.g., "I think this is an error with 70%
    confidence") and the **Ground Truth** (e.g., "This is actually Normal").
  - It calculates a **Loss Score** (Penalty).
  - If the model said "70% Error" and it was an Error, the penalty is small.
  - If the model said "70% Error" and it was Normal, the penalty is HUGE.
  - The model's only goal in life is to get this Loss Score as close to zero as
    possible.

### What is "Ground Truth"?

**Ground Truth** is the "Answer Key" provided by human experts. It is the
absolute truth that the model tries to mimic.

- **Source:** The `annotations/` folder in your workspace (specifically
  `error_annotations.json`).
- **Content:** Humans watched these cooking videos and wrote down exact
  timestamps: "At 00:15, the user cut the tomato incorrectly."
- **Role:** During training, the model is allowed to peek at this answer key to
  learn. During testing, the answer key is hidden to see how well the model
  learned.
