# Step 2 — Task Verification Notes

## Goal (what Step 2 is)

Step 2 is a **recording-level mistake detection** task:

- **Input:** a variable-length sequence of **step embeddings** for one recording (from Step 1).
- **Output:** a single probability/logit and a binary decision: **does this recording contain an error?** (`recipe_label ∈ {0,1}`)

In other words: Step 1 turns video → steps; Step 2 turns steps → *is the whole recipe execution correct or not?*

---

## Inputs (what files Step 2 consumes)

Step 2 loads one or more Step‑1 `.pkl` files, typically:

- `extension_data/step_embeddings_gt.pkl` (oracle segmentation upper bound)
- `extension_data/step_embeddings_actionformer.pkl` (predicted segmentation)

Each `.pkl` is expected to contain:

- `data`: `{recording_id -> record}`
- `splits`: train/val/test ids (kept for compatibility; Step 2 uses recipe-based folds)
- `feature_dim`: embedding dimension `D`
- `method`: metadata string (`"ground_truth"` or `"actionformer"`)

Each `record` must include:

- `step_embeddings`: numpy array `(num_steps, D)`
- `recipe_id`: int (which recipe/task this recording belongs to)
- `recipe_label`: `0/1` (recording-level error label)

Loader: `extension_steps/step2/io.py` (`load_step_embeddings_pkl`).

---

## Outputs (what Step 2 produces)

1) **Console summary** per embeddings file:
- mean ± std across folds for: `accuracy`, `precision`, `recall`, `f1`, `auc`

2) Optional **results JSON** (`--out-json`) written under `extension_data/reports/`:
- fold-by-fold metrics + a summarized mean/std

CLI entrypoint: `extension_steps/step2/main.py` (run as `python -m extension_steps.step2.main ...`).

---

## Methodology (how evaluation works)

### Leave-One-Recipe-Out (LORO) cross-validation

Instead of randomly splitting recordings, Step 2 evaluates generalization across **recipes**:

- Group recordings by `recipe_id`.
- For each fold: hold out all recordings from one recipe as test, train on the rest.
- Report mean ± std across folds.

Implementation: `extension_steps/step2/eval.py` (`leave_one_recipe_out_eval`).

### Why LORO matters

If you used random train/test splits, the model could overfit recipe-specific patterns (background, tools, common actions).
LORO tests whether the verifier generalizes to **unseen tasks/recipes**.

---

## Models (what is trained)

Step 2 provides baseline sequence models that map `(num_steps, D)` → scalar logit:

- **MLP**: simple baseline (fastest)
- **Transformer**: learns attention over steps (often stronger)
- **BiLSTM**: sequential modeling over steps

Models live in: `extension_steps/step2/models.py` and are selected by `--model`.

---

## Training details (important knobs)

Training uses `BCEWithLogitsLoss` with class weighting:

- `--pos-weight`: increases the loss weight of positive (error) recordings
  - higher → more recall-heavy (fewer false negatives, more false positives)

At evaluation time, predictions are converted to labels using:

- `--threshold`: decision threshold on sigmoid(logit)
  - higher → fewer predicted positives → precision up / recall down

---

## Key results (what we concluded so far)

See `extension_steps/step2/CONCLUSIONS.md` for the full timeline. Highlights:

### Step 1 embeddings are “good enough” for Step 2 (MLP baseline)

MLP performance on **ActionFormer embeddings** was very close to the **GT/oracle embeddings** baseline, suggesting Step 1 is not a major bottleneck for Step 2 anymore.

### Two recommended operating points (ActionFormer embeddings, MLP)

These are both valid depending on the project goal:

1) **Recall-first (mistake-sensitive)**  
   - `pos_weight=6`, `threshold=0.50`  
   - recall ≈ **0.99**, precision ≈ **0.59**

2) **Precision-first (fewer false positives)**  
   - `pos_weight=3`, `threshold=0.70`  
   - precision ≈ **0.63**, recall ≈ **0.77**

For a mistake detection project, it is usually better to **favor recall**, then choose the highest precision you can tolerate under a minimum recall constraint.

---

## How to run (paste-ready)

### Evaluate Step 2 on one embeddings file

```bash
python -m extension_steps.step2.main \
  --embeddings-pkl extension_data/step_embeddings_actionformer.pkl \
  --model mlp
```

### Compare GT vs ActionFormer

```bash
python -m extension_steps.step2.main \
  --embeddings-pkl extension_data/step_embeddings_gt.pkl extension_data/step_embeddings_actionformer.pkl \
  --model mlp
```

### Hyperparameter sweep (writes CSV)

```bash
python -m extension_steps.step2.sweep \
  --embeddings-pkl extension_data/step_embeddings_actionformer.pkl \
  --models mlp \
  --pos-weight 1 2 3 4 5 6 \
  --thresholds 0.4 0.5 0.6 \
  --top-n 10 \
  --sort-metric f1_mean \
  --out-csv extension_data/sweeps/step2_sweep.csv
```

---

## What to improve next (high leverage)

1) Try `--model transformer` and sweep a small grid over `--threshold` and `--pos-weight`.
2) Decide your target operating point explicitly:
   - recall-first (safety): enforce `recall ≥ 0.90`
   - precision-first (quiet): enforce `precision ≥ 0.65`
3) If Step 3/4 depend heavily on step-level correctness, consider adding a **step-level** verifier (not just recipe-level).
