# Step 1 — ActionFormer Notes (Implementation + Training Data)

This document explains **what the Step 1 ActionFormer pipeline trains on**, what the model predicts, how the loss is computed, and how the model updates its weights.

The refactored implementation lives in:

- `extension_steps/step1/actionformer/`
- Entry point: `extension_steps/step1/actionformer.py`

---

## 1) What data does ActionFormer train on?

### Inputs (X): EgoVLP feature sequences

For each recording id, we load a sequence of EgoVLP features from:

- `data/features/egovlp/<recording_id>_360p_224.npz`

Code:

- `extension_steps/step1/actionformer/data.py` → `load_features(...)`

Shape conceptually:

- `X ∈ R^(T×D)`
  - `T` = number of feature timesteps for that recording
  - `D` = feature dimension (typically `256` in this project)

Important: Step 1 operates on **features**, not raw video frames.

### Targets (y): boundary label vector from GT timestamps

Ground-truth step timestamps come from:

- `annotations/annotation_json/step_annotations.json`

Each recording contains a list of steps, each with:

- `start_time` (seconds)
- `end_time` (seconds)
- `description`
- `has_errors`

Training does **boundary detection**, so it converts these step times into a binary vector over time:

- `y[t] = 1` if timestep `t` is close to a step boundary (start or end)
- `y[t] = 0` otherwise

This is built in:

- `extension_steps/step1/actionformer/data.py` → `StepLocalizationDataset.__getitem__`

#### Feature FPS conversion (seconds → feature index)

The dataset converts seconds to feature indices using `feature_fps`:

- `start_idx = int(start_time * feature_fps)`
- `end_idx   = int(end_time   * feature_fps)`

In this repo, `feature_fps` is typically:

- `0.5` (≈ 1 feature every 2 seconds)

#### Boundary tolerance window

Instead of marking only a single index, we mark a small window around each boundary:

- default `boundary_tolerance=3` feature steps

So you get a few neighboring `1`s near each start/end boundary.

---

## 2) What does the model output?

The model predicts a **boundary logit per timestep** (after a `sigmoid`, that’s a probability).

In training, we use the first output level’s logits:

- `cls_logits = out_cls[0].squeeze(1)` → shape `(B, T')`

Where:

- `B` = batch size
- `T'` may differ from the padded `T_max` due to internal downsampling

Code:

- `extension_steps/step1/actionformer/train.py` → `train_one_epoch(...)`, `evaluate(...)`

---

## 3) How batching works (padding + masks)

Recordings have different lengths `T`, so the DataLoader pads them.

Batch construction:

- `extension_steps/step1/actionformer/data.py` → `collate_fn(...)`

It builds:

- `features`: `(B, T_max, D)`
- `boundary_labels`: `(B, T_max)`
- `masks`: `(B, T_max)` where `1` = real timestep, `0` = padding

The mask is used so padding does **not** contribute to the loss/metrics.

---

## 4) How training works (loss + weight updates)

### Loss function

Training uses **binary cross entropy with logits** per timestep:

- `BCEWithLogits(logit[t], label[t])`

Because boundaries are rare (many more zeros than ones), we use `pos_weight` to upweight positive labels.

Implementation:

- `extension_steps/step1/actionformer/train.py` → `compute_loss(...)`

Key details:

- Uses `binary_cross_entropy_with_logits(...)` (stable form: sigmoid + BCE combined)
- Multiplies by `masks` to ignore padded timesteps
- Averages over the number of valid (non-padding) timesteps

### Backpropagation + optimizer step

Within each minibatch:

1) Forward pass: compute logits from features
2) Compute loss against `boundary_labels`
3) `loss.backward()` computes gradients for all parameters
4) `optimizer.step()` updates weights (AdamW)

Code:

- `extension_steps/step1/actionformer/train.py` → `train_one_epoch(...)`
- Optimizer creation: `extension_steps/step1/actionformer/main.py` (AdamW)

---

## 5) How we decide if a predicted boundary is “good”

There are two different notions of “good”, and they measure different things:

### A) Training/validation F1 (timestep classification)

During validation, we threshold boundary probabilities at `0.5`:

- predict boundary if `sigmoid(logit) > 0.5`

Then compute precision/recall/F1 against the binary label vector `y[t]`.

Code:

- `extension_steps/step1/actionformer/train.py` → `evaluate(...)`

This measures: “did the model put high probability at the same timesteps we labeled as boundary?”

### B) Localization timing metrics (post-hoc segmentation)

After training, inference turns probabilities into segments by:

1) finding local maxima above a threshold
2) forming segments between successive boundary indices

Code:

- `extension_steps/step1/actionformer/infer.py` → `predict_segments(...)`, `segments_from_probs(...)`

Then `extension_steps/step1/compare_segments.py` compares predicted segments to GT segments using:

- boundary tolerance sweep (±1s, ±2s, ±5s)
- greedy matched segment IoU

This measures: “are the predicted boundary *times* close to the GT boundary times?”

---

## 6) End-to-end pipeline flow (Step 1 ActionFormer)

Entry point:

- `python extension_steps/step1/actionformer.py ...`

Main orchestration:

- `extension_steps/step1/actionformer/main.py` → `main()`

Stages:

1) Load GT annotations + splits (`data.py: load_json`)
2) Create datasets and loaders (`data.py: StepLocalizationDataset + collate_fn`)
3) Train ActionFormer (`train.py: train_one_epoch + evaluate`)
4) Pick best checkpoint by validation F1 (best-by-val-F1 snapshot)
5) Infer segments for each recording (`infer.py: predict_segments`)
6) Pool features inside predicted segments (`export.py: pool_features_from_frames`)
7) Save a Step-2-compatible `.pkl` (`export.py: save_step_embeddings_pkl`)

Output artifact:

- `extension_data/step_embeddings_actionformer.pkl`

Optional analysis convenience:

- `--align-to-gt` copies GT descriptions/labels onto predicted segments by best IoU match (only for easier downstream comparison; it’s not an evaluation metric).

---

## Human like explanation

### What ActionFormer does (example)

Think of a cooking video as a long timeline:

- 0–30s: take ingredients out  
- 30–60s: crack the egg  
- 60–90s: mix  
- 90–120s: cook  

Step 1’s job is to place **cut points** on that timeline: “a step ended here, a new step started here”.

ActionFormer looks at the video *over time* and outputs a “boundary likelihood” curve:

- low most of the time
- spikes near places where a step transition happens

Then we convert those spikes into **segments** (intervals between spikes). Those segments are the predicted steps.

### What is a “feature” here? (example)

A feature is a **numeric summary** of what’s happening in the video during a short time slice.

Instead of using raw pixels, EgoVLP already turned each video into a sequence like:

- timestep 0: a 256-number vector
- timestep 1: another 256-number vector
- timestep 2: another…

So a single video becomes a table: `T rows × 256 columns`.

With `feature_fps = 0.5`, you get roughly **one feature vector every ~2 seconds**. That limits how precise your boundary timestamps can be.

### Training/validation F1 vs GT comparison F1 (why they differ)

You have two different “F1” scores and they measure different things:

1) **Training/validation F1 (~0.68 in your logs)**  
   This is timestep-level classification: “at each timestep, boundary vs not-boundary”, compared to the binary label vector produced from GT timestamps (with a tolerance window).

2) **GT comparison boundary F1 (~0.10–0.37 depending on tolerance)**  
   This is timing-based: after converting probabilities → segments, it checks whether predicted boundary times land within ±X seconds of GT boundaries.

It’s normal for the training F1 to be much higher than strict timing F1, because the latter is affected by:

- coarse feature sampling (≈ every 2 seconds),
- post-processing choices (threshold, local-maxima picking, minimum segment length),
- the model producing “wide bumps” rather than sharp peaks at the exact boundary.
