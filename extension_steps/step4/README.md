# Step 4 — Graph Classification (GNN / Baselines)

Step 4 classifies an entire **recipe recording** (correct vs incorrect) using the **realized task graphs** produced by Step 3.

This folder refactors the original notebook export `extension_steps/step4_gnn_classification.py` into small, reusable modules and a CLI.

---

## Input

Step 3 output:

- `extension_data/realized_task_graphs.pkl`

Expected structure:

- `realized_graphs`: `{recording_id -> realized_graph}`

Each `realized_graph` should include (at minimum):

- `node_features`: `(num_nodes, D)` array-like
- `edge_index`: `(2, E)` array-like
- `recipe_id`: int
- `recipe_label`: `0/1` (recording-level error label)

---

## Models supported

- `pooled` (default): mean+max pool node features → MLP classifier (no torch-geometric needed)
- `gcn`, `gat`, `sage`: GNN graph classifiers (requires `torch-geometric`)

All models use **Leave-One-Recipe-Out (LORO)** cross-validation, like Step 2.

---

## Run

From the repo root (`aml-procedural-mistake-detection/`):

### Baseline without torch-geometric

```bash
python -m extension_steps.step4.main \
  --graphs-pkl extension_data/realized_task_graphs.pkl \
  --model pooled
```

### GNN (requires torch-geometric)

```bash
python -m extension_steps.step4.main \
  --graphs-pkl extension_data/realized_task_graphs.pkl \
  --model sage
```

### Save results to JSON

```bash
python -m extension_steps.step4.main \
  --graphs-pkl extension_data/realized_task_graphs.pkl \
  --model pooled \
  --out-json extension_data/step4_graph_classification_results.json
```

---

## Sweep (recall-first)

To sweep Step 4 hyperparameters and find an operating point that maximizes **recall** (2nd priority: **precision**):

```bash
python -m extension_steps.step4.sweep \
  --graphs-pkl extension_data/realized_task_graphs.pkl \
  --model sage \
  --num-layers 2 \
  --pos-weight 2 3 4 5 6 \
  --thresholds 0.30 0.35 0.40 0.45 0.50 \
  --top-n 10 \
  --out-csv extension_data/sweeps/step4_sweep_recall.csv
```

---

## Code layout

- `extension_steps/step4/main.py`: CLI entrypoint
- `extension_steps/step4/io.py`: load Step-3 `.pkl`
- `extension_steps/step4/pooling.py`: pooled feature baseline (no PyG)
- `extension_steps/step4/pyg.py`: optional torch-geometric integration
- `extension_steps/step4/models.py`: pooled baseline model + GNN model builders
- `extension_steps/step4/train.py`: train/eval loops
- `extension_steps/step4/eval.py`: LORO evaluation + fold summarization
- `extension_steps/step4/metrics.py`: metrics (no sklearn dependency)
- `extension_steps/step4/gnn_classification.py`: convenience wrapper for running as a file

For reference, the original notebook-exported script is preserved as:

- `extension_steps/step4/legacy_notebook_export.py`

Backwards-compatible wrapper (preferred entrypoint if you were used to running the exported script):

- `extension_steps/step4_gnn_classification.py`
