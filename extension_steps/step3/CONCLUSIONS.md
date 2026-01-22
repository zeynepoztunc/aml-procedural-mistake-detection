# Step 3 — Conclusions (Task Graph Matching)

## What Step 3 does

Step 3 converts each recording into a **realized task graph**:

- **Input**: Step 1 step embeddings (`extension_data/step_embeddings_*.pkl`) + recipe task-graph templates from `annotations/annotation_json/step_annotations.json` (and/or `task_graphs.json`).
- **Process**:
  - Encode each task-graph node’s text description into the **same 256‑D space** as Step 1 embeddings (via a cached text→visual projection).
  - Match recording steps to task-graph nodes using **Hungarian (1–1) assignment** on cosine similarity.
  - Create per-node features by mixing text and matched visual embedding: `alpha*visual + (1-alpha)*text`.
- **Output**: `extension_data/realized_task_graphs*.pkl` (consumed by Step 4).

---

## Why we added “unmatched nodes”

Hungarian matching always returns a full 1–1 assignment, even when some matches are clearly wrong (very low cosine similarity).

Those “forced” bad matches are informative for mistake detection, so we added a switch to treat low-similarity assignments as **unmatched**.

### New behavior (implementation)

- Added CLI flags (Step 3 entry point: `extension_steps/step3/task_graph_matching.py`):
  - `--min-match-sim <float>`: minimum cosine similarity required to count a Hungarian assignment as a **valid match**.
  - `--unmatched-node-features {text,zero}`:
    - `text` (default): unmatched nodes keep their **text** embedding (node description).
    - `zero`: unmatched nodes are set to **zero** features.
- Updated realized graph fields:
  - `node_matched` now reflects **valid** matches only (after applying `--min-match-sim`).
  - `num_valid_matches` / `num_unmatched` become meaningful signals for Step 4.
  - `match_similarities` still stores the raw Hungarian similarities (useful for analysis).

Code locations:
- Matching + thresholding: `extension_steps/step3/realize.py`
- CLI wiring / saved config: `extension_steps/step3/main.py`
- Quick inspection helper: `extension_steps/step3/summarize_realized_graphs.py`

---

## Timeline (changes + results)

### 2026-01-22 — Baseline Step 3 run (Hungarian, no match threshold)

**Command**
```bash
python extension_steps/step3/task_graph_matching.py ^
  --step1-pkl extension_data/step_embeddings_gt.pkl ^
  --out-pkl extension_data/realized_task_graphs.pkl
```

**Summary**
- `min_match_similarity = 0.0` → **no nodes are considered unmatched** (`total_unmatched_nodes = 0`).
- Still, the *raw* similarities show a strong gap between errors vs non-errors:
  - `mean(min_sim)` error=1: **0.2595** vs error=0: **0.6230**
  - `frac(min_sim==0)` error=1: **0.5864** vs error=0: **0.0122**

Interpretation: Hungarian is **forcing** 1–1 matches even when similarity is ~0, and those low similarities correlate strongly with error recordings.

### 2026-01-22 — Added “unmatched nodes” via `--min-match-sim`

**Command**
```bash
python extension_steps/step3/task_graph_matching.py ^
  --step1-pkl extension_data/step_embeddings_gt.pkl ^
  --out-pkl extension_data/realized_task_graphs_minsim0.2.pkl ^
  --min-match-sim 0.2 ^
  --unmatched-node-features text
```

**Results (inspection)**
- `total_unmatched_nodes = 257` across **384 recordings** (≈ **0.67 unmatched nodes / recording**).
- Worst-by-similarity examples are mostly **label=1** and now show **multiple unmatched nodes** (e.g. `22_137: valid=7 unmatched=8`).

**Conclusion**
- This change introduces an explicit “some canonical steps do not match well” signal that Step 4 can exploit (instead of hiding it behind forced 1–1 matches).

