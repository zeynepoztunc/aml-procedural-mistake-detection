# Step 4 — Task Verification Notes (GNN Classification)

## Task statement (what is being predicted)

Given a recipe recording, predict if it contains an error:

- Target label: `recipe_label ∈ {0,1}` (0 = correct, 1 = error)
- Unit of prediction: **recording-level**, not step-level.

Step 4 solves this using a graph representation of the recording.

---

## Inputs (where they come from)

### Primary input: realized task graphs (from Step 3)

File:
- `extension_data/realized_task_graphs*.pkl`

Structure:
- `payload["realized_graphs"][recording_id]` is a dict containing:
  - `node_features`: `(num_nodes, D)` float array
  - `edge_index`: `(2, E)` int array (directed edges between canonical steps)
  - `node_matched`: `(num_nodes,)` bool (optional but useful signal)
  - `match_similarities`: `(num_nodes,)` float (optional but useful signal)
  - `recipe_id`: which recipe template the recording belongs to
  - `recipe_label`: recording-level error label

Important variants:
- `extension_data/realized_task_graphs.pkl`: baseline matching (Hungarian forced matches).
- `extension_data/realized_task_graphs_minsim0.2.pkl`: “unmatched nodes” enabled in Step 3 via `--min-match-sim 0.2`.

---

## Outputs (what you get after running Step 4)

Running `python -m extension_steps.step4.main ...` prints:
- device used,
- number of folds (recipes),
- mean ± std for metrics across LORO folds.

Optionally writes a JSON report via `--out-json` containing:
- full per-fold metrics,
- summarized metrics,
- training/eval configs used.

---

## Methodology (how evaluation works)

### Cross-validation: Leave‑One‑Recipe‑Out (LORO)

- Group recordings by `recipe_id`.
- For each recipe:
  - train on recordings from the other recipes,
  - test on recordings of the held‑out recipe.

This measures how well the system generalizes to unseen recipes (not just unseen recordings).

### Metrics

Step 4 reports:
- accuracy
- precision / recall / F1 (at `--threshold`)
- AUC (threshold‑free ranking quality)

### Class imbalance handling

Training uses `--pos-weight` to upweight error examples in the loss (helps if errors are rarer or you want recall‑first behavior).

---

## Models implemented (how they work)

### `pooled` baseline (recommended starting point)

1) Pool node features into one recording vector:
   - mean pool + max pool (concatenate)
2) Feed pooled vector into an MLP classifier.

Pros:
- Fast, stable, strong baseline.
- Does not rely on graph edges being “perfect”.

### GNN models (`gcn`, `gat`, `sage`)

1) Message passing: each node aggregates neighbor information using `edge_index`.
2) Readout: pool node embeddings → recording embedding.
3) MLP head → prediction.

Pros:
- Can exploit order/structure of steps (edges).
Risks:
- **Oversmoothing** with too many layers (`--num-layers` too high).
- Requires `torch-geometric`.

---

## Practical run commands

Baseline (no PyG):
```bash
python -m extension_steps.step4.main ^
  --graphs-pkl extension_data/realized_task_graphs.pkl ^
  --model pooled ^
  --out-json extension_data/reports/step4_pooled.json
```

Compare “unmatched nodes” effect:
```bash
python -m extension_steps.step4.main ^
  --graphs-pkl extension_data/realized_task_graphs_minsim0.2.pkl ^
  --model pooled ^
  --out-json extension_data/reports/step4_pooled_minsim0.2.json
```

GNN example:
```bash
python -m extension_steps.step4.main ^
  --graphs-pkl extension_data/realized_task_graphs.pkl ^
  --model sage ^
  --num-layers 2 ^
  --out-json extension_data/reports/step4_sage_l2.json
```

---

## What to aim to improve (highest leverage)

In rough priority order:

1) **Graph input quality (Step 3)**:
   - tune `--min-match-sim` to make “unmatched” signal informative without discarding too much.
2) **Operating point**:
   - tune `--threshold` and `--pos-weight` (recall-first vs precision-first).
3) **Model capacity / regularization**:
   - `--hidden-dim`, `--dropout`, `--weight-decay`, and `--num-epochs`.
4) **GNN depth**:
   - prefer shallow GNNs (2–3 layers) unless you have strong evidence deeper helps.

