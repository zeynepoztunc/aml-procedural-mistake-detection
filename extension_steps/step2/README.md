# Step 2 — Task Verification (Baselines)

Step 2 predicts whether an entire **recipe recording** contains an error (binary classification), given the **step embeddings** produced by Step 1.

This folder refactors the original notebook export `extension_steps/step2_verification_baseline.py` into small, reusable modules and a CLI.

---

## Input

A Step-1 output pickle, typically one of:

- `extension_data/step_embeddings_gt.pkl` (oracle / GT segmentation)
- `extension_data/step_embeddings_actionformer.pkl` (ActionFormer segmentation)

The pickle is expected to contain:

- `data`: `{recording_id -> record}`
- `splits`: train/val/test ids (kept for compatibility; LORO does not require them)

Each `record` must include:

- `step_embeddings`: `(num_steps, D)` numpy array
- `recipe_id`: int
- `recipe_label`: `0/1` (recording-level error label)

---

## What this step does

- **Dataset**: pads/truncates each recording to `max_steps` embeddings and creates a mask.
- **Models**: simple baselines (MLP / Transformer / BiLSTM).
- **Evaluation**: **Leave-One-Recipe-Out (LORO)** cross-validation:
  - train on all recipes except one, test on the held-out recipe

---

## Run

From the repo root (`aml-procedural-mistake-detection/`):

## Conclusions / experiment log

- `extension_steps/step2/CONCLUSIONS.md`
- `extension_steps/step2/TASK_VERIFICATION_NOTES.md`

### Evaluate one embeddings file

```bash
python -m extension_steps.step2.main \
  --embeddings-pkl extension_data/step_embeddings_gt.pkl \
  --model mlp
```

### Compare multiple Step-1 methods

```bash
python -m extension_steps.step2.main \
  --embeddings-pkl extension_data/step_embeddings_gt.pkl extension_data/step_embeddings_actionformer.pkl \
  --model mlp
```

### Save results to JSON

```bash
python -m extension_steps.step2.main \
  --embeddings-pkl extension_data/step_embeddings_gt.pkl \
  --model mlp \
  --out-json extension_data/step2_verification_results.json
```

---

## Hyperparameter sweep (writes CSV)

Sweep simple Step 2 knobs (e.g., `pos_weight`, decision `threshold`) and write a CSV:

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

Add `--no-progress` to disable progress bars.

---

## Code layout

- `extension_steps/step2/main.py`: CLI entrypoint
- `extension_steps/step2/io.py`: load Step-1 `.pkl` files (+ small legacy-key normalization)
- `extension_steps/step2/data.py`: dataset + collate
- `extension_steps/step2/models.py`: MLP / Transformer / LSTM baselines
- `extension_steps/step2/train.py`: train/eval loops + metrics
- `extension_steps/step2/eval.py`: LORO evaluation + fold summarization
- `extension_steps/step2/verification_baseline.py`: convenience wrapper for running as a file

For reference, the original notebook-exported script is still available at:

- `extension_steps/step2_verification_baseline.py` (and copied to `extension_steps/step2/legacy_notebook_export.py`)
