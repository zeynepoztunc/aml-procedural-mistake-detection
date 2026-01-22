# Step 1 — Localization → Step Embeddings

Step 1 turns **per-video EgoVLP feature sequences** into a **list of step segments** and a **fixed-size embedding per step**.  
Those step embeddings are the input to Step 2 (verification) and later steps.

There are two variants:

1) **GT / Oracle localization** (uses human timestamps)  
2) **ActionFormer localization** (predicts timestamps automatically)

---

## What “localization” means here

**Input (per recording)**: EgoVLP features `X ∈ R^(T×D)` extracted from the video (no raw video decoding here).  
**Goal**: produce a set of step segments `[(start, end), ...]` over time.  
**Then**: “pool” features inside each segment to get one vector per step.

---

## Inputs (common)

- **EgoVLP features**: `data/features/egovlp/<recording_id>_360p_224.npz`
  - Each file contains a sequence of feature vectors over time.
- **Ground-truth step annotations**: `annotations/annotation_json/step_annotations.json`
  - Per recording: list of steps with `start_time`, `end_time`, `description`, `has_errors`.
- **Train/val/test split**: `er_annotations/recordings_combined_splits.json`

---

## Outputs (what Step 2 expects)

Both variants produce a pickle with the same structure:

- `extension_data/step_embeddings_gt.pkl` (GT pipeline)
- `extension_data/step_embeddings_actionformer.pkl` (ActionFormer pipeline)

Each file is a dict like:

- `method`: `"gt"` or `"actionformer"`
- `feature_dim`: feature dimension `D`
- `splits`: train/val/test ids
- `data`: mapping `{recording_id -> record}`

Each `record` contains (keys used by later steps):

- `step_embeddings`: `(num_steps, D)` array
- `step_labels`: list of 0/1 (step error label)
- `recipe_id`: int
- `recipe_label`: 0/1 recording-level label
- `descriptions`: list of strings (optional depending on method)
- `segments`: list of `(start_time, end_time)` in seconds
- `step_ids`: list of ints (optional depending on method)

---

## Comparing ActionFormer vs GT (Step 1 evaluation)

You can evaluate Step 1 localization quality by comparing predicted segments to GT segments using:

- `extension_steps/step1/compare_segments.py`

It reports:
- **Boundary precision/recall/F1** (boundaries matched within a time tolerance)
- **Greedy matched IoU** (simple 1–1 segment overlap score; not mAP, but easy to interpret)

Example (test split, default tolerance = 2s):

```bash
python extension_steps/step1/compare_segments.py \
  --gt-pkl extension_data/step_embeddings_gt.pkl \
  --pred-pkl extension_data/step_embeddings_actionformer.pkl \
  --split test \
  --boundary-tol-sec 2
```

Multiple tolerances in one run:

```bash
python extension_steps/step1/compare_segments.py \
  --gt-pkl extension_data/step_embeddings_gt.pkl \
  --pred-pkl extension_data/step_embeddings_actionformer.pkl \
  --split test \
  --boundary-tols-sec 1 2 5
```

### Inference sweep (writes CSV)

Tune inference post-processing (`threshold`, `min_seg_len`) on the **val** split and write results to a CSV:

```bash
python extension_steps/step1/sweep_inference.py \
  --checkpoint-path extension_data/actionformer_best.pt \
  --gt-pkl extension_data/step_embeddings_gt.pkl \
  --split val \
  --thresholds 0.3 0.4 0.5 0.6 0.7 \
  --min-seg-lens 8 15 25 \
  --tolerances 1 2 5 \
  --smooth-windows 1 5 \
  --smooth-types box gaussian \
  --peak-distances 0 5 \
  --segment-modes threshold topk \
  --target-num-segments 15 \
  --max-seg-lens 0 60 \
  --sort-metric score_iou_count \
  --score-count-penalty 0.02 \
  --top-n 10 \
  --out-csv extension_data/sweeps/inference_sweep.csv
```

### GT step duration stats

This helps choose reasonable values for `min_seg_len` and boundary tolerances based on the GT distribution.

```bash
python extension_steps/step1/gt_step_stats.py --feature-fps 0.5
```

---

## Variant A: GT / Oracle pipeline (recommended first run)

**Task**: “Given the true step timestamps, pool features into step embeddings.”

This is the simplest and fastest way to validate the downstream pipeline.

### Entry point

- `extension_steps/step1/gt_pipeline.py`
  - Wrapper that runs the exported script.

### Main script (implementation)

- `extension_steps/step1_pipeline.py`
  - Loads annotations + splits
  - Loads EgoVLP features per recording
  - Pools features within each GT step segment (mean pooling)
  - Saves `extension_data/step_embeddings_gt.pkl`

### Input → Output

- Input segments: from `step_annotations.json` (GT start/end times)
- Output segments/embeddings: saved as `step_embeddings_gt.pkl`

---

## Variant B: ActionFormer pipeline (automatic localization)

**Task**: “Learn to detect step boundaries from feature sequences, then pool features using predicted segments.”

This is the realistic setting for new videos where GT timestamps are not available.

### Entry point

- `extension_steps/step1/actionformer.py`
  - Runs the refactored ActionFormer pipeline.

### Refactored package layout

Folder: `extension_steps/step1/actionformer/`

- `main.py`
  - Orchestrates: load data → train → infer segments → export `.pkl`
  - CLI args for paths + thresholds.
- `data.py`
  - `load_features(...)`: reads `.npz` feature sequences
  - `StepLocalizationDataset`: creates boundary labels from GT for training
  - `collate_fn(...)`: pads variable-length sequences
- `model.py`
  - `ActionFormer`: backbone + FPN + heads (simplified from notebook export)
  - `get_sinusoid_encoding(...)`: positional encoding for temporal order
- `train.py`
  - `train_one_epoch(...)`, `evaluate(...)`
  - `cosine_with_warmup(...)`: learning-rate schedule helper
- `infer.py`
  - `predict_segments(...)`: converts boundary probabilities → segments
  - `segments_from_probs(...)`: local maxima + minimum segment length logic
- `export.py`
  - Pools features per predicted segment and saves the Step-2-compatible `.pkl`
  - Optional: align predicted segments to the closest GT step to copy descriptions/labels (`--align-to-gt`)

### Input → Output

- Input segments for training labels: from GT (to supervise the boundary detector)
- Output segments for embedding extraction: predicted by the model
- Output file: `extension_data/step_embeddings_actionformer.pkl`

### Run

From repo root:

```bash
python extension_steps/step1/actionformer.py --align-to-gt
```

Common knobs:

- `--threshold` controls how confident a boundary must be.
- `--min-seg-len` prevents tiny segments in feature frames.
- `--smooth-type gaussian --smooth-window W` can reduce jitter while keeping peaks localized.
- `--min-peak-prominence P --prominence-window W` filters peaks that don't stand out from nearby baseline (reduces spurious boundaries).
- `--segment-mode topk --target-num-segments 15` forces ~15 segments by picking the strongest peaks.
- `--max-seg-len 60` (optional) splits very long segments using strong peaks (a simple duration prior).
- `--feature-fps` must match how often features were extracted (this repo typically uses `0.5`).

---

## Legacy notebook export (for reference only)

- `extension_steps/step1_actionformer.py`

This is the large, notebook-derived linear script. It contains extra grid-search and analysis code; use the refactored package above for studying/running.
