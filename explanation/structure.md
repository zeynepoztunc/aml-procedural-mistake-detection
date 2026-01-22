# Project understanding structure (technical map)

## 0) What problem is being solved

- **Primary task (Error Recognition / ER):** classify whether a **recipe step clip** contains a mistake (binary: correct vs mistake).
- **Extension task (Task Verification):** classify whether an **entire recording** follows a valid recipe/task graph; uses step embeddings + graph matching + GNNs.

---

## 1) Data & storage (what exists, where, what it represents)

### 1.1 Annotations / labels

- **Step boundaries + recipe text:** `annotations/annotation_json/step_annotations.json`
  - For each recording: list of steps with start/end times + step description text.
- **Splits:** `er_annotations/recordings_combined_splits.json`
  - Train/val/test recording IDs for the ER task.
- **Recording-level error label:** stored/derived in extension outputs (e.g., `recording_error_labels` inside `extension_data/step_embeddings_gt.pkl`).

### 1.2 Features (inputs to models)

- **EgoVLP features:** `data/features/egovlp/*.npz`
  - Pre-extracted embeddings over time; your models train on these instead of raw video.
- (Other backbones may exist in other branches/paths, but your extension run uses EgoVLP.)

### 1.3 Generated artifacts (outputs)

- **Step embeddings (Step 1 output):** `extension_data/step_embeddings_gt.pkl`
- **Task verification results (Step 2 output):** `extension_data/task_verification_results.json`
- **Realized graphs (Step 3 output):** `extension_data/realized_task_graphs.pkl`
- **GNN results (Step 4 output):** `extension_data/gnn_results.json`

---

## 2) Metrics (what is computed and why it matters)

### 2.1 Classification metrics (used in ER + Step 2 + Step 4)

- **Accuracy:** overall correctness; can be misleading with imbalance.
- **Precision / Recall / F1:** key for “mistake” class; **F1** is usually the headline.
- **AUC / PR-AUC:** threshold-independent ranking quality; PR-AUC is more informative under class imbalance.

### 2.2 Localization metrics (ActionFormer-style step boundary detection)

- Not “accuracy”; typical:
  - **Boundary precision/recall/F1** with tolerance window
  - **mAP@tIoU** for temporal localization (if implemented)

---

## 3) ER pipeline (baseline training runs you documented)

### 3.1 Goal

Learn a model that predicts mistake vs correct at **step-level**, optionally with sub-step signals aggregated.

### 3.2 What the code is doing conceptually

- Load per-recording feature sequences from `.npz`
- Form training examples by slicing/aggregating features by step boundaries
- Train a classifier (MLP/Transformer/LSTM variants exist)
- Evaluate on val/test, including threshold selection (your Run 006 sweeps threshold on validation)

### 3.3 Important knobs you used (Run 006)

- **Threshold sweep on validation** → avoids tuning on test.
- **Top-k pooling** → focuses on strongest mistake evidence in a step.
- **Early stopping on Val F1** → selects a generalizing checkpoint.

---

## 4) Extension pipeline (Steps 1–4 you just ran)

### Step 1: Step localization → step embeddings

**Goal:** produce a compact representation per step for each recording.

- **Input:** EgoVLP time-series features + step boundaries.
- **Process:** pool features within each step segment.
- **Output:** `extension_data/step_embeddings_gt.pkl` (GT boundaries baseline)

Variant:

- **Step 1 v2 (ActionFormer-style):** predict boundaries first, then pool within predicted segments.

### Step 2: Task verification baselines (no graphs)

**Goal:** predict whether a recording is “valid/invalid” (mistake) using only step embeddings.

- **Models:**
  - **MLP:** simple pooled representation → classifier.
  - **Transformer:** attends across step sequence.
  - **LSTM:** sequential model across steps.
- **Eval:** **Leave-One-Recipe-Out** cross-validation (mean ± std).

### Step 3: Task graph encoding + matching

**Goal:** align video steps to recipe graph nodes to build a “realized” graph for each recording.

- **Inputs:** step embeddings + recipe step text.
- **Text encoder:** e.g., DistilBERT projected to the visual embedding dim.
- **Matching:** assign step embeddings to graph nodes (often Hungarian / similarity-based).
- **Output:** `extension_data/realized_task_graphs.pkl` (graph structure + node features + match quality signals)

### Step 4: Graph-based classification

**Goal:** classify mistake vs correct using the realized graphs.

- **Baseline:** `SimplePooling` (graph features pooled → classifier)
- **GNN models (alternatives):**
  - **GCN:** normalized neighbor averaging.
  - **GAT:** learned attention weights over neighbors.
  - **GraphSAGE:** learned aggregation (often strong for classification).
- **Eval:** Leave-One-Recipe-Out CV; results in `extension_data/gnn_results.json`.

---

## 5) “What models mean” (how to interpret them)

- **MLP:** learns from global pooled stats; weakest temporal/structural inductive bias.
- **Transformer:** can learn which steps matter and step-to-step relations.
- **LSTM:** captures order but less flexible than attention for long-range.
- **GCN/GAT/GraphSAGE:** explicitly learn from **graph structure** (recipe flow + matched nodes), capturing dependencies that plain sequence models can miss.

---

## 6) How data is processed / filtered (mental model)

- **Feature loading:** `.npz` per recording → time × dim tensor.
- **Segmentation:** step boundaries determine which timesteps belong to which step.
- **Pooling:** mean/top-k/attention pooling compresses variable-length segments to fixed-size step vectors.
- **Dataset construction:**
  - ER: examples are step clips (and possibly sub-step signals aggregated).
  - Extension: examples are recordings; input is a sequence of step vectors or a realized task graph.
- **Cross-validation grouping:** extension uses recipe IDs to avoid leakage across the same recipe type.

---

## 7) How results are computed (end-to-end)

- **ER:** train → pick threshold/checkpoint (often via validation) → report metrics on test.
- **Extension Step 2:** for each held-out recipe fold:
  - train on other recipes → test on held-out recipe → compute metrics → aggregate mean±std.
- **Extension Step 4:** same as Step 2 but with graph inputs and GNN models.

---

## 8) Goals per step (why each exists)

- **Step 1:** create the representation (bottleneck) that everything else relies on.
- **Step 2:** establish a “no-graph” baseline ceiling from pure visual step embeddings.
- **Step 3:** inject recipe/task structure via text + matching → create a meaningful graph representation.
- **Step 4:** exploit graph structure to improve classification beyond Step 2.

---

## 9) High-value improvement directions (what to try next)

### Step 1 improvements (usually huge leverage)

- Better pooling (top-k fraction, attention pooling).
- Boundary quality: ActionFormer post-processing (thresholds, min segment length, NMS), better evaluation (mAP@tIoU).
- Normalize or smooth features before pooling; handle corrupt/empty segments.

### Step 2 improvements

- Tune decision threshold on validation folds.
- Regularization and class imbalance (pos_weight, focal loss, sampling).
- Better sequence modeling (transformer depth, dropout, positional encoding).
- Calibrate outputs (temperature scaling) if threshold sensitivity is high.

### Step 3 improvements

- Use a stronger / aligned text encoder (or cache HF models locally).
- Better similarity metric (cosine + normalization), add “no-match” option to avoid forced bad matches.
- Incorporate temporal order constraints during matching (monotonic alignment).

### Step 4 improvements

- Graph features: add match confidence, node coverage, temporal stats.
- Model/hyperparams: layers, hidden dim, dropout, global pooling choice, edge directionality.
- Better graph construction: ensure edges reflect recipe flow; reduce noise from mismatched nodes.

### Experiment hygiene

- Fix seeds + determinism if you need reproducibility.
- Keep a `runs/` record with config + metrics + artifact hashes.

---

## 10) Suggested “deep dive order”

1. Data files & annotations (what labels mean, how splits avoid leakage)
2. Step 1 embedding construction (pooling + boundary assumptions)
3. ER metrics & thresholding (how F1/AUC change with threshold)
4. Step 2 CV protocol (how folds are formed and why)
5. Step 3 matching mechanics (how nodes get assigned)
6. Step 4 GNN architecture and what graph signals it uses

