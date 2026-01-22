# 0) Two tasks in this project: "Error Recognition" vs "Extension (Task Verification)"

This project has two related goals. Think of them like two different ways of inspecting a job site:

- **Primary task (Error Recognition / ER):** Inspect **one step** of the job (e.g., "cutting wood") and decide: did a mistake happen inside this step?
- **Extension task (Task Verification):** Inspect the **whole job** (the full sequence of steps) and decide: does the overall process match a valid plan, or is something wrong in the sequence?

I'll explain both as if you're a carpenter and the system is your "quality inspector".

## A) Primary task: Error Recognition (ER)

### A1) What ER is trying to answer

You take a single work step (a short segment of a video) like:

- measure and mark
- cut the board
- drill pilot holes

The ER model answers a yes/no question:

> Is this step correct, or does it contain a procedural mistake?

This is a **binary classification** problem: `Correct` vs `Mistake`.

### A2) Why ER is useful

If you can reliably tell whether each step is correct:

- you can flag mistakes early
- you can compute how often mistakes happen
- you can build a feedback system ("this step was done wrong")

### A3) The big practical constraint: raw video is too heavy

Reviewing raw video frame-by-frame is expensive. So the project uses pre-extracted **features**:

- A big pretrained video model converts each short time window into a **vector** of numbers (an "embedding").
- The ER model trains on these vectors instead of raw pixels.

In this repo, those embeddings are stored as `.npz` files under `data/features/egovlp/`.

Carpenter analogy:

- raw video = watching the entire job in real time
- embeddings = a compact "inspection note" per time slice (still numbers, but it summarizes what's happening)

### A4) What the ER model actually learns

ER does not learn "vision" from scratch; it learns a smaller model that maps embeddings to a mistake probability.

Common model families used here:

- **MLP:** simplest. Like a learned checklist applied to pooled features.
  - Good when average evidence is enough.
  - Weak at modeling timing/order.

- **LSTM:** reads step embeddings in order and keeps a memory.
  - Better for "what happened first vs later".

- **Transformer:** uses attention, can focus on the most informative moments and compare parts of the step.
  - Often strongest for sequence modeling.

### A5) "Sub-step" vs "Step-level"

You'll see two evaluation levels:

- **Sub-step level:** "when inside the step did the mistake happen?" (fine-grained timeline)
- **Step-level:** "did this step contain any mistake at all?" (coarse yes/no per step)

Most reporting focuses on **Step-level** because it is the main goal and more stable.

### A6) Thresholds and why they matter

Models output a probability (e.g., 0.73 means "73% mistake"). A **threshold** (e.g., 0.75) turns that into yes/no:

- low threshold: catch more mistakes (high recall), but more false alarms (lower precision)
- high threshold: fewer false alarms, but miss more mistakes

That's why runs often include a **validation threshold sweep**: pick the threshold that maximizes F1 on validation (and then evaluate test with that threshold).

### A7) ER metrics (plain meaning)

- **Accuracy:** how often the yes/no label is correct (can be misleading with imbalance).
- **Precision:** when we call "mistake", how often we're right (false-alarm rate).
- **Recall:** out of real mistakes, how many we catch (miss rate).
- **F1:** balance of precision and recall (often the headline metric).
  - Why it's named like that: it comes from the **F-measure** family, and the **"1"** means precision and recall are weighted **equally** (a "harmonic mean" of the two).
- What the **F-measure family** is:
  - The **F-measures** are a family of single-number scores that combine **precision** and **recall** so you don't have to judge two metrics separately.
  - The general form is **F-beta (Fβ)**, where **beta** controls what you care about more:
    - **beta > 1** emphasizes **recall** (missing a true mistake is costly).
    - **beta < 1** emphasizes **precision** (false alarms are costly).
    - **beta = 1** gives **F1**, which balances them equally.
  - (Formula) `Fβ = (1 + β²) * (Precision * Recall) / (β² * Precision + Recall)`.
- **AUC / PR-AUC:** ranking quality across thresholds (threshold-independent view).
  - Why it's named like that:
    - **AUC** means **Area Under the Curve**: the curve is usually the ROC curve (tradeoff between catching mistakes and false alarms as you vary the threshold).
      - What the **ROC curve** is:
        - **ROC** stands for **Receiver Operating Characteristic**.
        - It plots how your classifier behaves as you sweep the decision threshold:
          - **x-axis:** **False Positive Rate (FPR)** = fraction of correct examples incorrectly flagged as mistakes.
          - **y-axis:** **True Positive Rate (TPR)** = fraction of real mistakes correctly caught (this is the same as **recall**).
        - Each point on the curve corresponds to one threshold setting; moving the threshold changes the tradeoff between false alarms and caught mistakes.
        - The closer the curve is to the top-left corner, the better the model’s ranking behavior (high recall with low false alarms).
    - **PR-AUC** means **Precision-Recall Area Under the Curve**: the curve is precision vs recall as you vary the threshold; it is often more informative when the classes are imbalanced.

### A8) Practical ways to improve ER

1. **Better temporal aggregation**
   - If mistakes are brief, averaging across the step can "wash out" the signal.
   - Use **top-k pooling** or attention pooling to focus on the most mistake-like moments.
2. **Better thresholding**
   - Always tune threshold on validation (not test).
3. **Better class imbalance handling**
   - `pos_weight`, sampling, focal loss.
4. **Data hygiene**
   - handle corrupt/missing `.npz` entries; avoid NaNs.

## B) Extension task: Task Verification (sequence + graph reasoning)

### B1) What the extension is trying to answer

Instead of checking one step, we check the whole recording:

> Did the person follow a valid recipe/task plan overall?

Many procedural mistakes are about the **process**, not just the motion:

- skipping a required step
- doing steps in the wrong order
- repeating steps incorrectly

Carpenter analogy:

- ER checks whether a single cut is wrong.
- Task verification checks whether the whole build followed the blueprint.

### B2) Why graphs enter the picture

A recipe/blueprint has structure:

- some steps must precede others
- some steps depend on others

A natural representation is a **graph**:

- **nodes** = steps ("measure", "cut", "assemble")
- **edges** = allowed transitions/dependencies ("measure" -> "cut")

### B3) Extension Step 1: create step embeddings (the "notes" per step)

Goal:

- Convert each step segment into one vector that represents that step's visual content.

How:

- Pool EgoVLP features inside each step boundary.

What this means in concrete terms:

1) What the EgoVLP data looks like (input)

- For each recording, you load a time sequence of feature vectors from a file like:
  - `data/features/egovlp/<recording_id>_360p_224.npz`
- Conceptually it looks like a 2D array:
  - shape = `(T, D)`
  - `T` = number of time windows (like 1-second or 2-second chunks, depending on extraction)
  - `D` = feature dimension (in your extension run it is typically `256`)

Short example (fake numbers, real shapes):

- EgoVLP features for one recording: `X`
  - `X.shape = (301, 256)`
  - `X[0]` is a 256-number summary of the first time window

What a "256-number summary" looks like (shortened example):

```text
X[0] (D=256) ~= [ 0.12, -0.03, 0.88, 0.01, -0.44, 0.27, 0.09, -0.15, 0.62, ... , -0.07 ]
```

Interpretation:

- each position is a learned feature (not human-labeled like "saw" or "hammer")
- the whole vector acts like a compact numeric signature of what was happening in that time window

2) What the step boundaries look like (input)

- From `annotations/annotation_json/step_annotations.json`, each recording has steps like:
  - Step 1: `start_time=7.1`, `end_time=46.3`, `text="Coat a ramekin..."`
  - Step 2: `start_time=50.3`, `end_time=82.3`, ...

Carpenter analogy:

- The features are your "notes over time".
- The step boundaries tell you which time range corresponds to each step of the plan.

3) How pooling works (the core operation)

Pooling means: take all time-window feature vectors that fall inside a step and compress them into one vector.

Let a step cover indices `t0..t1` in the feature timeline. Then:

- Step slice: `S = X[t0:t1]`
  - `S.shape = (N, D)` where `N` is how many time windows are inside the step

Common pooling choices:

- Mean pooling (simple average):
  - `step_vec = mean(S, axis=0)`
  - Output shape: `(D,)`
  - Good when the step content is consistent across time.
  - Weakness: brief mistake evidence can get diluted by "normal" frames.

- Top-k pooling (focus on strongest evidence):
  - You first compute a per-time "score" (often the model's mistake probability/logit or another signal).
  - Take only the top `k` time windows and pool them (mean/max).
  - Helps when mistakes are brief and localized.

Tiny toy example (mean pooling):

- Suppose a step has `N=3` time windows and `D=4` features:
  - `S = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]`
  - `mean(S) = [0.33, 0.33, 0.33, 0.00]` (one vector representing the step)

4) What the output looks like (Step 1 result)

For each recording you get a matrix of step embeddings:

- `E.shape = (num_steps_in_recording, D)`

Example from your run artifacts:

- Recording `1_7` has `12` steps, embedding dim `256`:
  - `E.shape = (12, 256)`

These per-recording step embeddings are saved to:

- `extension_data/step_embeddings_gt.pkl`

That pickle is a dictionary with metadata and data; commonly you will see top-level keys like:

- `data`: mapping `recording_id -> { step_embeddings: np.ndarray, ... }`
- `splits`: train/val/test split lists used by the extension notebooks
- `feature_dim`: embedding dimension (e.g., 256)
- `method`: which boundary source was used (e.g., `ground_truth`)
- `feature_type`: which feature backbone was used (e.g., `egovlp`)

Inside each `data[recording_id]`, the important fields used downstream are:

- `step_embeddings`: `(num_steps, feature_dim)` float array (one vector per step)
- `step_labels`: list of per-step error labels (0/1)
- `recipe_id`: integer recipe id (derived from recording id prefix, e.g. `1_7 -> 1`)
- `recipe_label`: recording-level label (0/1), typically `1` if **any** step has an error

Outputs:

- `extension_data/step_embeddings_gt.pkl` (ground-truth boundary baseline).

Key improvement lever:

- Better pooling or better boundaries -> better downstream results.

### B4) Extension Step 2: baselines (no graphs yet)

Goal:

- Predict whether a recording has a mistake using only the sequence of step embeddings.

Models:

- MLP / Transformer / LSTM (same families, now operating over step embeddings).

Why do this:

- It sets the baseline: "how far can we get without graph structure?"

Evaluation approach:

- **Leave-One-Recipe-Out cross-validation**:
  - train on 23 recipe types
  - test on the held-out recipe type
  - repeat for all recipes

Why it helps:

- tests generalization; reduces "memorize one recipe" leakage.

### B5) Extension Step 3: task graph encoding + matching (connecting video to the plan)

Goal:

- Align what the person did (video steps) to blueprint steps (text nodes in the task graph).

Main components:

1. **Text encoder (e.g., DistilBERT)**
   - Input: step description text.
   - Output: a vector per node so it can be compared to video step vectors.
2. **Projection + normalization**
   - Project text vectors to the same dimension as visual embeddings (e.g., 256).
   - Normalize for meaningful similarity comparisons.
3. **Matching**
   - Compute similarity between each video step and each graph node.
   - Choose assignments; produce a "realized" task graph for the recording.

Output:

- `extension_data/realized_task_graphs.pkl`

Key improvement levers:

- stronger text encoder / better alignment to the visual space
- better matching constraints (avoid forced bad matches; enforce order constraints)

### B6) Extension Step 4: GNN classification (reasoning over structure)

Goal:

- Decide if the whole recording is correct/mistake by analyzing the realized task graph.

Why a GNN helps:

- It learns from graph connectivity: each node update depends on its neighbors.
- It can learn patterns like "required node missing" or "wrong transitions".

Models used (trained as alternatives, compared after):

- **GCN:** neighbor averaging with normalization (simple baseline GNN).
- **GAT:** learns attention weights over neighbors (focuses on important connections).
- **GraphSAGE:** learns an aggregation function for neighbors (often strong and robust).
- **SimplePooling:** non-GNN baseline (pool graph features, classify).

Output:

- `extension_data/gnn_results.json`

Key improvement levers:

- richer node/edge features (match confidence, coverage, temporal stats)
- better task-graph construction and edge directionality
- hyperparameter tuning (layers, hidden dim, dropout, pooling)

## C) How to interpret "improvement" in this project

Meaningful comparisons:

1. **Within ER:** compare ER runs/checkpoints using the same task and splits.
2. **Within the extension pipeline:** compare Step 2 baselines vs Step 4 graph models (same cross-validation protocol).

Not a clean claim:

- "Step 4 F1 beats ER F1"

…because they are different tasks with different evaluation protocols.

## D) Practical next deep dives

- If you care about "mistake inside a step": dig into ER thresholding + pooling.
- If you care about "process correctness": dig into Step 3 matching quality and Step 4 graph features (most gains come from these).
