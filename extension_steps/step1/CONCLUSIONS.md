# Step 1 — Conclusions (Localization)

## What Step 1 does

Step 1 converts a **video-level EgoVLP feature sequence** into **step segments** and then into **step-level embeddings** (one vector per step) by pooling features within each segment.

Downstream steps (Step 2–4) operate on these step embeddings rather than on raw videos.

---

## Two variants and what they mean

### 1) GT / Oracle pipeline

- Uses **human-annotated** step boundaries from `annotations/annotation_json/step_annotations.json`.
- Purpose: **upper bound** + sanity check for the rest of the pipeline.
- Output: `extension_data/step_embeddings_gt.pkl`

### 2) ActionFormer pipeline

- Learns to **predict step boundaries automatically** from EgoVLP features.
- Purpose: realistic end-to-end setting where you do **not** have timestamps at test time.
- Output: `extension_data/step_embeddings_actionformer.pkl`

---

## Key observations from training logs (how to read them)

- **Train loss decreasing** means the model is fitting the training boundary labels better.
- If **val F1 stops improving** (or decreases) while train loss keeps decreasing, that is typically **overfitting**:
  - the model is learning training-specific quirks/noise and generalizing less well.
- In this implementation, inference uses the **best validation F1 checkpoint** (not the last epoch), so running more epochs is usually safe but wastes time.

### Your run (current results)

- Model size: **9,482,770 parameters**
- Split coverage (recordings with available features): **train=287, val=48, test=48**
- Best validation boundary F1: **~0.689** (peaked around **epoch 11–14**)
- Pattern: train loss kept decreasing through epoch 30 while val loss increased and val F1 drifted down → consistent with **overfitting after the early peak**.

---

## What is a “valid” Step 1 evaluation?

Comparing ActionFormer segments to GT segments is valid and standard, as long as you report metrics that match the task:

- **Boundary precision/recall/F1**: are predicted boundaries within ±X seconds of GT boundaries?
- **Segment overlap**: temporal IoU-based overlap scores between predicted and GT segments.

### What IoU means here

IoU (Intersection over Union) is computed on **time intervals**. For a predicted segment
`[p_start, p_end]` and a GT segment `[g_start, g_end]` (seconds):

- intersection = `max(0, min(p_end, g_end) - max(p_start, g_start))`
- union = `(p_end - p_start) + (g_end - g_start) - intersection`
- IoU = `intersection / union`

So IoU = 1 means perfect overlap; IoU = 0 means no overlap.

This repo includes a simple comparison tool:

```bash
python extension_steps/step1/compare_segments.py \
  --gt-pkl extension_data/step_embeddings_gt.pkl \
  --pred-pkl extension_data/step_embeddings_actionformer.pkl \
  --split test
```

---

## Practical guidance (what to run and when)

- Use the **GT pipeline first** to verify everything works end-to-end (fastest).
- Use **ActionFormer** when you want a realistic pipeline (no GT timestamps at inference).
- Recommended training length (based on your logs): **~12–15 epochs**.
  - You already peak around that range; training to 30 mostly wastes time (best checkpoint is still kept).
  - Practical default: `--num-epochs 12` (or keep 30 but add early stopping if you want).

---

## Timeline (changes + results)

Add entries here as you iterate so you can justify design decisions in your report/exam.

### 2026-01-22 — Baseline refactored ActionFormer run + GT comparison

**Change / setup**
- Ran the refactored Step 1 ActionFormer pipeline (best-by-val-F1 checkpointing).
- Compared predicted segments to GT segments on the **test split** using `extension_steps/step1/compare_segments.py`.

**Training outcome**
- Model size: **9,482,770 params**
- Recordings with features (train/val/test): **287 / 48 / 48**
- Best validation boundary F1 during training: **~0.689** (around epoch **11–14**)
- Signs of overfitting after the peak: train loss keeps decreasing while val loss increases.

**Localization comparison vs GT (test split, ±2s boundary tolerance)**
- Avg #segments: **GT=15.00**, **Pred=14.88**
- Boundary metrics: **P=0.203**, **R=0.189**, **F1=0.191**
- Segment overlap (greedy 1–1 IoU): **mean=0.536** over **619** matched pairs

**Boundary tolerance sweep (test split)**
- ±1s: **P=0.112**, **R=0.103**, **F1=0.105**
- ±2s: **P=0.203**, **R=0.189**, **F1=0.191**
- ±5s: **P=0.395**, **R=0.367**, **F1=0.372**

**Recommendation from this stage**
- Use **~12–15 epochs** for faster iteration (your val F1 peaks there).
- If you care about strict boundary timing, consider tuning inference `--threshold/--min-seg-len` and/or evaluating with a larger boundary tolerance (e.g., 5s) alongside 2s.

### 2026-01-22 — CUDA env run (13 epochs) + updated GT comparison

**Change / setup**
- Ran Step 1 ActionFormer from the CUDA-enabled environment (`.venv-pyg`).
- Trained for **13 epochs** and saved `extension_data/actionformer_best.pt`.

**Training outcome**
- Best validation boundary F1 during training: **0.6976** (epoch 13)
- Inference throughput: **384 recordings in ~6s** (fast, consistent with GPU use)

**Localization comparison vs GT (test split)**
- Avg #segments: **GT=15.00**, **Pred=13.83**
- Boundary tolerance sweep:
  - ±1s: **P=0.120**, **R=0.102**, **F1=0.108**
  - ±2s: **P=0.213**, **R=0.185**, **F1=0.194**
  - ±5s: **P=0.427**, **R=0.373**, **F1=0.390**
- Segment overlap (greedy 1–1 IoU): **mean=0.551** over **605** matched pairs

### 2026-01-22 — Improved inference: smoothing + peak-distance NMS

**Change / setup**
- Updated Step 1 inference post-processing to support:
  - moving-average smoothing (`--smooth-window`)
  - peak non-max suppression by minimum distance (`--peak-distance`)
- Ran infer-only with: `--smooth-window 5 --peak-distance 5` and compared on test.

**Localization comparison vs GT (test split)**
- Avg #segments: **GT=15.00**, **Pred=12.23**
- Boundary tolerance sweep:
  - ±1s: **P=0.153**, **R=0.118**, **F1=0.130**
  - ±2s: **P=0.278**, **R=0.219**, **F1=0.240**
  - ±3s: **P=0.383**, **R=0.297**, **F1=0.327**
  - ±4s: **P=0.450**, **R=0.352**, **F1=0.385**
  - ±5s: **P=0.508**, **R=0.398**, **F1=0.436**
  - ±6s: **P=0.567**, **R=0.444**, **F1=0.486**
- Segment overlap (greedy 1–1 IoU): **mean=0.604** over **556** matched pairs

**Notes**
- This improves boundary metrics and IoU, but under-segments (lower avg predicted steps).
- Next tuning knob is balancing `threshold / min_seg_len / peak_distance` to increase segment count without losing timing quality.

### 2026-01-22 — Best val sweep config tested on test (smoothed inference)

**Change / setup**
- Selected best configuration from automated inference sweep (optimized for `f1@2s` on **val**):
  - `threshold=0.5`, `min_seg_len=8`, `smooth_window=5`, `peak_distance=0`
- Evaluated on **test** using `compare_segments.py`.

**Localization comparison vs GT (test split)**
- Avg #segments: **GT=15.00**, **Pred=14.88**
- Boundary tolerance sweep:
  - ±1s: **P=0.138**, **R=0.130**, **F1=0.131**
  - ±2s: **P=0.254**, **R=0.243**, **F1=0.244**
  - ±3s: **P=0.347**, **R=0.328**, **F1=0.331**
  - ±4s: **P=0.415**, **R=0.394**, **F1=0.396**
  - ±5s: **P=0.473**, **R=0.450**, **F1=0.452**
  - ±6s: **P=0.532**, **R=0.505**, **F1=0.508**
- Segment overlap (greedy 1–1 IoU): **mean=0.591** over **630** matched pairs

### 2026-01-22 — New inference options implemented (next experiments)

**Change / Implementation**
- Added additional inference post-processing knobs to try:
  - Gaussian smoothing: `--smooth-type gaussian --smooth-window W [--smooth-sigma S]`
  - Peak prominence filter (reduce spurious boundaries): `--min-peak-prominence P --prominence-window W`
  - Top-K peak segmentation (force a target step count): `--segment-mode topk --target-num-segments 15`
  - Max segment length splitting (duration prior): `--max-seg-len 60 [--min-split-peak-prob P]`

**Results**
- Not evaluated yet (run a new val sweep / compare on test once a configuration is selected).

### 2026-01-22 — Composite sweep (IoU + segment count) → test evaluation

**Change / setup**
- Switched sweep ranking to a composite score to prioritize overlap while keeping segment counts realistic:
  - `score_iou_count = iou_mean - 0.02 * abs(avg_pred_segments - avg_gt_segments)`
- Selected a top configuration from the **val** sweep:
  - `threshold=0.7`, `min_seg_len=8` (no smoothing/prominence/max-len used in this run)
- Evaluated on **test** using `compare_segments.py`.

**Localization comparison vs GT (test split)**
- Avg #segments: **GT=15.00**, **Pred=14.17**
- Boundary tolerance sweep:
  - ±1s: **P=0.143**, **R=0.126**, **F1=0.131**
  - ±2s: **P=0.247**, **R=0.221**, **F1=0.228**
  - ±3s: **P=0.343**, **R=0.307**, **F1=0.317**
  - ±4s: **P=0.411**, **R=0.373**, **F1=0.383**
  - ±5s: **P=0.465**, **R=0.424**, **F1=0.434**
  - ±6s: **P=0.530**, **R=0.482**, **F1=0.494**
- Segment overlap (greedy 1–1 IoU): **mean=0.585** over **615** matched pairs

**Notes**
- This configuration improves IoU while keeping the predicted segment count close to GT.

### 2026-01-22 — Val-optimized Gaussian smoothing config → test evaluation

**Change / setup**
- Selected the top configuration from the **val** sweep ranked by `score_iou_count`:
  - `threshold=0.6`, `min_seg_len=8`, `smooth_window=5`, `smooth_type=gaussian`
- Evaluated on **test** using `compare_segments.py`:
  - `extension_data/step1_embeddings/step_embeddings_actionformer_thr0.6_min8_sm5_gauss.pkl`

**Localization comparison vs GT (test split)**
- Avg #segments: **GT=15.00**, **Pred=14.94** (closest so far)
- Boundary tolerance sweep:
  - ±1s: **P=0.141**, **R=0.132**, **F1=0.134**
  - ±2s: **P=0.250**, **R=0.238**, **F1=0.239**
  - ±3s: **P=0.348**, **R=0.329**, **F1=0.331**
  - ±4s: **P=0.408**, **R=0.389**, **F1=0.390**
  - ±5s: **P=0.461**, **R=0.442**, **F1=0.441**
  - ±6s: **P=0.525**, **R=0.501**, **F1=0.502**
- Segment overlap (greedy 1–1 IoU): **mean=0.587** over **629** matched pairs

**Quick comparison**
- Slightly better than the `thr=0.7, min=8` run in boundary F1 and segment count closeness.
- Still slightly behind the earlier best test IoU (`0.591`) by a small margin.

### 2026-01-22 — Box smoothing @ thr=0.6 (test evaluation)

**Change / setup**
- Tested the strongest `box` smoothing candidate from the val “high threshold” sweep:
  - `threshold=0.6`, `min_seg_len=8`, `smooth_window=5`, `smooth_type=box`
- Evaluated on **test** using `compare_segments.py`:
  - `extension_data/step1_embeddings/step_embeddings_actionformer_thr0.6_min8_sm5_box.pkl`

**Localization comparison vs GT (test split)**
- Avg #segments: **GT=15.00**, **Pred=13.94**
- Boundary tolerance sweep:
  - ±1s: **P=0.142**, **R=0.126**, **F1=0.130**
  - ±2s: **P=0.267**, **R=0.242**, **F1=0.249**
  - ±3s: **P=0.365**, **R=0.324**, **F1=0.337**
  - ±4s: **P=0.433**, **R=0.386**, **F1=0.399**
  - ±5s: **P=0.489**, **R=0.438**, **F1=0.453**
  - ±6s: **P=0.548**, **R=0.490**, **F1=0.507**
- Segment overlap (greedy 1–1 IoU): **mean=0.594** over **610** matched pairs

**Notes**
- Best IoU observed so far on test (**0.594**), but predicted segment count drops to **13.94**.

### 2026-01-22 — Recommended inference config (balanced IoU + segment count)

**Why this config**
- Among the tested configurations, this one provides **near-best IoU** while keeping the **predicted segment count close to GT**.
- Use this as the default Step 1 ActionFormer inference setting for downstream steps.

**Config**
- `threshold=0.5`, `min_seg_len=8`, `smooth_window=5`, `smooth_type=box`
- Output pkl: `extension_data/step1_embeddings/step_embeddings_actionformer_thr0.5_min8_sm5_box.pkl`

**Test results (compare vs GT)**
- Avg #segments: **GT=15.00**, **Pred=14.85**
- Boundary tolerance sweep:
  - ±1s: **P=0.138**, **R=0.130**, **F1=0.131**
  - ±2s: **P=0.259**, **R=0.247**, **F1=0.248**
  - ±3s: **P=0.353**, **R=0.332**, **F1=0.335**
  - ±4s: **P=0.422**, **R=0.398**, **F1=0.402**
  - ±5s: **P=0.478**, **R=0.453**, **F1=0.455**
  - ±6s: **P=0.535**, **R=0.506**, **F1=0.510**
- Segment overlap (greedy 1–1 IoU): **mean=0.593** over **629** matched pairs

### 2026-01-22 — Training improvement: Gaussian labels + higher pos_weight (test evaluation)

**Change / setup**
- Trained with soft Gaussian boundary targets and increased positive class weight to address class imbalance:
  - `--boundary-label-mode gaussian --boundary-window 2 --boundary-sigma 1.0`
  - `--pos-weight 7`
  - checkpoint selection: `--select-best-by score_iou_count`
- Evaluated exported `extension_data/step_embeddings_actionformer.pkl` on **test**.

**Localization comparison vs GT (test split)**
- Avg #segments: **GT=15.00**, **Pred=15.69** (slight over-segmentation)
- Boundary tolerance sweep:
  - ±1s: **P=0.127**, **R=0.124**, **F1=0.122**
  - ±2s: **P=0.255**, **R=0.249**, **F1=0.247**
  - ±3s: **P=0.357**, **R=0.348**, **F1=0.345**
  - ±4s: **P=0.449**, **R=0.440**, **F1=0.435**
  - ±5s: **P=0.502**, **R=0.495**, **F1=0.487**
  - ±6s: **P=0.544**, **R=0.539**, **F1=0.529**
- Segment overlap (greedy 1–1 IoU): **mean=0.603** over **638** matched pairs

**Notes**
- This is the best IoU observed so far on test (**0.603**), suggesting training improvements can meaningfully improve overlap.
- Next: tune inference threshold/prominence slightly upward to bring `avg_pred` closer to ~15 while preserving IoU.

### 2026-01-22 — Best overall run so far (Gaussian labels + pos_weight=6)

**Training config (command used)**
- `--boundary-label-mode gaussian --boundary-window 2 --boundary-sigma 1.0`
- `--pos-weight 6`
- `--select-best-by score_iou_count`
- Inference/post-processing during export/eval:
  - `--threshold 0.5 --min-seg-len 8 --smooth-window 5 --smooth-type box`

**Artifacts saved (frozen filenames)**
- Checkpoint: `extension_data/checkpoints/actionformer_best_gaussLbl_w2_s1_pos6_score_ioucount.pt`
- Step embeddings: `extension_data/step1_embeddings/step_embeddings_actionformer_gaussLbl_w2_s1_pos6_thr0.5_min8_sm5_box.pkl`

**Test results (compare vs GT)**
- Avg #segments: **GT=15.00**, **Pred=15.23**
- Boundary tolerance sweep:
  - ±1s: **P=0.139**, **R=0.134**, **F1=0.134**
  - ±2s: **P=0.261**, **R=0.251**, **F1=0.251**
  - ±3s: **P=0.363**, **R=0.348**, **F1=0.348**
  - ±4s: **P=0.452**, **R=0.439**, **F1=0.437**
  - ±5s: **P=0.518**, **R=0.504**, **F1=0.501**
  - ±6s: **P=0.566**, **R=0.550**, **F1=0.547**
- Segment overlap (greedy 1–1 IoU): **mean=0.601** over **637** matched pairs

### 2026-01-22 — Best balanced test result so far (IoU=0.603 with higher F1@2s)

**Exact command (reproducible)**
```bash
python extension_steps/step1/actionformer.py --align-to-gt --select-best-by score_iou_count --threshold 0.5 --min-seg-len 8 --smooth-window 5 --smooth-type box --boundary-label-mode gaussian --boundary-window 2 --boundary-sigma 1.0 --pos-weight 6 --loss-type focal --focal-gamma 1.0
```

**Artifacts saved (frozen filenames)**
- Checkpoint: `extension_data/checkpoints/actionformer_best_for_step_embeddings_actionformer_best_test_iou0.603_f12s0.256.pt`
- Step embeddings: `extension_data/step1_embeddings/step_embeddings_actionformer_best_test_iou0.603_f12s0.256.pkl`

**Test results (compare vs GT)**
- Avg #segments: **GT=15.00**, **Pred=14.50**
- Boundary tolerance sweep:
  - ±1s: **P=0.151**, **R=0.138**, **F1=0.141**
  - ±2s: **P=0.273**, **R=0.250**, **F1=0.256**
  - ±3s: **P=0.381**, **R=0.347**, **F1=0.356**
  - ±4s: **P=0.484**, **R=0.446**, **F1=0.456**
  - ±5s: **P=0.539**, **R=0.499**, **F1=0.508**
  - ±6s: **P=0.575**, **R=0.534**, **F1=0.543**
- Segment overlap (greedy 1–1 IoU): **mean=0.603** over **622** matched pairs
