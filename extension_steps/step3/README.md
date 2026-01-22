# Step 3 — Task Graph Encoding → Hungarian Matching

Step 3 builds a **task-graph template per recipe type** (nodes = canonical steps), embeds those node descriptions into the **same feature space as Step 1** embeddings, and then matches each recording’s observed step embeddings to graph nodes via **Hungarian (1–1) assignment**.

The result is a **“realized task graph”** per recording, saved to `extension_data/realized_task_graphs.pkl`, which is the input for Step 4 (GNN classification).

This folder is the refactored, runnable equivalent of the notebook export:

- Legacy linear export: `extension_steps/step3_task_graph_matching.py`
- Refactored entry point: `extension_steps/step3/task_graph_matching.py`

---

## Inputs

- Step 1 output (GT pipeline recommended):
  - `extension_data/step_embeddings_gt.pkl`
- Step annotations:
  - `annotations/annotation_json/step_annotations.json`
- Optional prebuilt task graphs (if present):
  - `annotations/annotation_json/task_graphs.json`

If `task_graphs.json` is missing, this step builds linear graphs from `step_annotations.json`.

---

## Text embeddings (local-friendly)

The original notebook uses EgoVLP’s text encoder (aligned with EgoVLP visual embeddings). For local runs (and to avoid downloading large checkpoints), this refactor uses a practical alternative:

1) Extract text features from step descriptions (prefers `transformers` DistilBERT **if already cached locally**)  
2) Fit a ridge-regression projection `W` from text features → Step 1 visual embedding space, using Step 1’s GT pairs `(description, step_embedding)`  
3) Encode task-graph node descriptions with `Z = XW`

The fitted `W` is cached at `extension_data/step3_text_to_visual_W.npy`.

---

## Output (what Step 4 expects)

Writes:

- `extension_data/realized_task_graphs.pkl`

With keys:

- `realized_graphs`: `{recording_id -> realized_graph}`
- `splits`: train/val/test ids (copied from Step 1)
- `config`: includes `feature_dim`

Each `realized_graph` contains:

- `node_features`: `(num_nodes, D)` float array
- `edge_index`: `(2, E)` int array
- `node_matched`: `(num_nodes,)` bool mask
- `match_similarities`: `(num_nodes,)` float array
- `num_valid_matches`: number of matches that passed `--min-match-sim`
- `recipe_id`, `recipe_label`

---

## Run

From repo root:

```bash
python extension_steps/step3/task_graph_matching.py
```

Common options:

```bash
python extension_steps/step3/task_graph_matching.py ^
  --step1-pkl extension_data/step_embeddings_gt.pkl ^
  --out-pkl extension_data/realized_task_graphs.pkl ^
  --alpha-visual 0.5 ^
  --refit-projection
```

### Treat low-similarity assignments as “unmatched”

Hungarian matching always outputs a full 1–1 assignment, even if some similarities are extremely low. In practice, those forced matches are often a useful *signal* (e.g., “this step doesn’t align to any canonical node”), so Step 3 supports marking them as unmatched:

- `--min-match-sim`: if a matched pair’s cosine similarity is below this value, the node is marked `node_matched=False` and its `node_features` are left as “unmatched” features.
- `--unmatched-node-features`:
  - `text` (default): unmatched nodes keep their **text** embedding.
  - `zero`: unmatched nodes get **zero** features.

Example:

```bash
python extension_steps/step3/task_graph_matching.py ^
  --step1-pkl extension_data/step_embeddings_gt.pkl ^
  --out-pkl extension_data/realized_task_graphs.pkl ^
  --min-match-sim 0.2 ^
  --unmatched-node-features text
```

If you do not want to use `transformers` at all:

```bash
python extension_steps/step3/task_graph_matching.py --no-transformers
```

---

## Inspect output quickly

To inspect the realized graphs without fighting PowerShell one-liners:

```bash
python extension_steps/step3/summarize_realized_graphs.py --pkl extension_data/realized_task_graphs.pkl --top-n 10
```
