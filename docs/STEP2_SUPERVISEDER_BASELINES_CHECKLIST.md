# Step 2 (Professor): SupervisedER baseline reproduction checklist

Goal: reproduce CaptainCook4D **SupervisedER** baselines (**V1 = MLP**, **V2 = Transformer**) using the same metrics (Accuracy/Precision/Recall/F1/AUC), analyze performance by **error type**, and propose a **new baseline** (e.g., LSTM/RNN) for comparison.

## 0) Scope + definitions
- [ ] Confirm the exact task definition: **SupervisedER** (step-level mistake detection), input features (Omnivore/SlowFast per paper), target labels, and which split(s) to report (e.g., `step`, `recordings`).
- [ ] Confirm evaluation metrics and any thresholding policy used in the paper (fixed threshold vs tuned threshold; threshold value per split if applicable).
- [ ] Write down a “repro contract”: no test leakage, fixed seed(s), exact dataset version, and consistent preprocessing.

## 1) Environment + data sanity
- [ ] Create/activate the correct environment (same Python/Torch versions as the repo expects).
- [ ] Verify dataset/annotations exist locally and are readable:
  - [ ] Step annotations JSON/CSV exist and load without errors.
  - [ ] Error-type labels are present (Technique/Timing/Temperature/Measurement/Preparation).
- [ ] Verify feature files exist:
  - [ ] Omnivore features (paper baseline input).
  - [ ] SlowFast features (paper baseline input).
- [ ] Run a quick integrity check:
  - [ ] Count steps, labels, and error-type distribution (overall + per split).
  - [ ] Check class imbalance (positive/negative ratio).
  - [ ] Check for missing/NaN features.

## 2) Reproduce V1 (MLP) baseline
- [ ] Identify the baseline entrypoint in this repo (script/notebook) used for SupervisedER training/eval.
- [ ] Ensure configuration matches paper as closely as possible:
  - [ ] Backbone/features (Omnivore or SlowFast) match the reported table you’re reproducing.
  - [ ] Training schedule (epochs/steps), batch size, LR, weight decay, dropout, etc.
  - [ ] Split protocol and threshold are exactly as required.
- [ ] Train or load the official checkpoint (if provided) and evaluate:
  - [ ] Accuracy
  - [ ] Precision
  - [ ] Recall
  - [ ] F1
  - [ ] AUC
- [ ] Save outputs (for reproducibility):
  - [ ] CLI command used
  - [ ] Config dump (JSON/YAML)
  - [ ] Metrics summary (JSON/CSV)
  - [ ] Per-step predictions (optional but useful for error-type breakdown)

## 3) Reproduce V2 (Transformer) baseline
- [ ] Repeat V1 steps but with V2 architecture:
  - [ ] Same features, same split, same thresholding policy.
  - [ ] Same evaluation protocol and metrics.
- [ ] Save outputs exactly like V1 (commands/config/metrics).
- [ ] Compare vs V1 and vs the paper:
  - [ ] Are results within a reasonable tolerance?
  - [ ] If not: document differences (threshold, split, feature version, seed variance, preprocessing).

## 4) Error-type performance analysis (required extra)
- [ ] Define how you slice by error type:
  - [ ] Evaluate each error type separately (binary per-type) OR
  - [ ] Stratify the evaluation set by error type and report metrics per subset.
- [ ] Compute per-error-type metrics for **both V1 and V2**:
  - [ ] Precision/Recall/F1/AUC per error type
  - [ ] Support (number of samples) per error type
- [ ] Add at least one helpful visualization:
  - [ ] Bar chart of F1 per error type (V1 vs V2)
  - [ ] Or PR curves per type (optional)
- [ ] Write 3–5 bullet observations:
  - [ ] Which error types are easiest/hardest and why (hypothesis grounded in data/visual semantics).
  - [ ] Any counter-intuitive behavior (e.g., identical metrics across types).

## 5) New baseline proposal (required)
Example: **LSTM** over the sequence of sub-segment features within each step.
- [ ] Define the baseline clearly:
  - [ ] Input: sequence length, feature dimension, padding/masking.
  - [ ] Model: BiLSTM (or LSTM) + pooling + classifier.
  - [ ] Loss: BCEWithLogits, class weighting (if used).
- [ ] Keep the comparison fair:
  - [ ] Same features as V1/V2.
  - [ ] Same split protocol and threshold policy.
  - [ ] Similar training budget (epochs/steps).
- [ ] Train + evaluate on the same metrics:
  - [ ] Accuracy/Precision/Recall/F1/AUC
- [ ] (Optional but recommended) Run a small hyperparameter sweep:
  - [ ] LR, hidden dim, dropout, pos_weight, threshold
- [ ] Document model size/complexity vs gains:
  - [ ] Params count (optional)
  - [ ] Training time (optional)

## 6) Final comparison table + verification
- [ ] Create a single comparison table for V1/V2/New baseline:
  - [ ] Metrics on the chosen split(s)
  - [ ] Note feature type (Omnivore/SlowFast)
  - [ ] Note whether results are from official ckpt vs retrained
- [ ] Add a short “repro verification” section:
  - [ ] Exact commands to reproduce each number
  - [ ] Random seed(s)
  - [ ] Hardware/device

## 7) Common pitfalls checklist (quick)
- [ ] Threshold mismatch between paper vs repo defaults (record the exact threshold used).
- [ ] Split mismatch (`step` vs `recordings`) or data leakage (same recipe/person overlap).
- [ ] Different feature extraction version than paper (dim, normalization).
- [ ] Comparing tuned threshold to untuned baseline (keep operating points consistent).

