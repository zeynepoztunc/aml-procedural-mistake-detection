# Step 4 — Conclusions (Graph Classification)

## What Step 4 does

Step 4 predicts whether an entire **recipe recording** contains an error (`recipe_label ∈ {0,1}`) using the **realized task graphs** produced by Step 3.

- **Input**: `extension_data/realized_task_graphs*.pkl`
  - `realized_graphs[recording_id]` must contain `node_features (N,D)`, `edge_index (2,E)`, and `recipe_label (0/1)`.
- **Output**: printed cross‑validation metrics; optionally a JSON report via `--out-json`.

This step uses **Leave‑One‑Recipe‑Out (LORO)** cross‑validation (same idea as Step 2): train on 23 recipes and test on the held‑out recipe, repeated for all recipes.

---

## Models supported (what they mean)

- `pooled` (default, no PyG): mean+max pool node features → MLP classifier. This is a strong, simple baseline.
- `gcn`, `gat`, `sage` (requires `torch-geometric`): graph neural networks that use the edges (`edge_index`) to propagate information between steps.

---

## What to focus on (metrics + goals)

Because this is mistake detection, the “best” operating point depends on whether you want to:

- **Catch mistakes** (recall‑first): minimize false negatives.
- **Avoid false alarms** (precision‑first): minimize false positives.

Practically:
- Use **F1** as the primary summary metric (balanced tradeoff).
- Track **precision and recall** to understand the operating point.
- Use **AUC** to compare models independent of a fixed threshold.

---

## Timeline (changes + results)

Add entries here as you run Step 4 so you can reference them in your report/exam.

### 2026-01-22 — Step 4 baseline run (pooled)

**Command (paste‑ready)**
```bash
python -m extension_steps.step4.main ^
  --graphs-pkl extension_data/realized_task_graphs.pkl ^
  --model pooled ^
  --out-json extension_data/reports/step4_pooled.json
```

**Results**
- Fill in after running (see the JSON + console output).

### 2026-01-22 — Step 4: test impact of Step 3 “unmatched nodes”

Step 3 can emit graphs where low‑similarity node assignments are marked unmatched via `--min-match-sim`.

**Command (paste‑ready)**
```bash
python -m extension_steps.step4.main ^
  --graphs-pkl extension_data/realized_task_graphs_minsim0.2.pkl ^
  --model pooled ^
  --out-json extension_data/reports/step4_pooled_minsim0.2.json
```

**Hypothesis**
- If “unmatched nodes” captures mismatch signal, Step 4 should improve (or allow a better precision/recall tradeoff).

### 2026-01-22 — Step 4 GNN runs (GCN/GAT/GraphSAGE)

**Commands (paste‑ready)**
```bash
python -m extension_steps.step4.main ^
  --graphs-pkl extension_data/realized_task_graphs.pkl ^
  --model gcn ^
  --out-json extension_data/reports/step4_gcn.json
```

```bash
python -m extension_steps.step4.main ^
  --graphs-pkl extension_data/realized_task_graphs.pkl ^
  --model sage ^
  --out-json extension_data/reports/step4_sage.json
```

```bash
python -m extension_steps.step4.main ^
  --graphs-pkl extension_data/realized_task_graphs.pkl ^
  --model gat ^
  --out-json extension_data/reports/step4_gat.json
```

**Notes**
- Watch for **oversmoothing** as `--num-layers` increases (deep GNNs can make node features too similar and hurt performance).

### 2026-01-22 — Step 4 results (GraphSAGE, L=2) + recall-first sweep

These runs show that a shallow GNN (GraphSAGE with 2 layers) is a strong model for Step 4 on our realized graphs.

#### Baseline graphs vs “unmatched nodes” graphs (L=2)

**Baseline graphs**
- Input: `extension_data/realized_task_graphs.pkl`
- JSON: `extension_data/reports/step4_sage_baseline_l2.json`
- Metrics:
  - Accuracy: **71.94% ± 11.12%**
  - Precision: **75.23% ± 16.74%**
  - Recall: **79.80% ± 13.56%**
  - F1: **75.75% ± 10.66%**
  - AUC: **82.25% ± 9.52%**

**Unmatched nodes (min_match_sim=0.2)**
- Input: `extension_data/realized_task_graphs_minsim0.2.pkl`
- JSON: `extension_data/reports/step4_sage_minsim0.2_l2.json`
- Metrics:
  - Accuracy: **72.08% ± 10.75%**
  - Precision: **75.78% ± 16.18%**
  - Recall: **78.67% ± 13.54%**
  - F1: **75.50% ± 10.53%**
  - AUC: **81.51% ± 8.94%**

**Conclusion**
- For SAGE(L=2), the baseline graphs are slightly better overall (notably higher AUC and slightly higher recall/F1).
- Keep `realized_task_graphs.pkl` as the default graph input for Step 4.

#### Recall-first sweep (priority: recall, then precision)

**Sweep config**
- Script: `python -m extension_steps.step4.sweep`
- Input graphs: `extension_data/realized_task_graphs.pkl`
- Model: `sage`, `num_layers=2`
- Swept:
  - `pos_weight ∈ {2,3,4,5,6}`
  - `threshold ∈ {0.30,0.35,0.40,0.45,0.50}`
- Output CSV: `extension_data/sweeps/step4_sweep_recall.csv`

**Recommended operating points (two key configs)**

1) **Recall-first (default for mistake detection)**
- `pos_weight=6`, `threshold=0.30`
- Recall: **0.8480**
- Precision: **0.7552**
- F1: **0.7809**
- AUC: **0.8288**

Paste-ready command:
```bash
python -m extension_steps.step4.main ^
  --graphs-pkl extension_data/realized_task_graphs.pkl ^
  --model sage ^
  --num-layers 2 ^
  --pos-weight 6 ^
  --threshold 0.30 ^
  --device cuda ^
  --out-json extension_data/reports/step4_sage_l2_pw6_thr0.30.json
```

2) **More balanced (slightly lower recall, higher precision)**
- `pos_weight=6`, `threshold=0.50`
- Recall: **0.8326**
- Precision: **0.7763**
- F1: **0.7872**
- AUC: **0.8288**

Paste-ready command:
```bash
python -m extension_steps.step4.main ^
  --graphs-pkl extension_data/realized_task_graphs.pkl ^
  --model sage ^
  --num-layers 2 ^
  --pos-weight 6 ^
  --threshold 0.50 ^
  --device cuda ^
  --out-json extension_data/reports/step4_sage_l2_pw6_thr0.50.json
```

---

## Conclusions (to finalize after runs)

Once Step 4 results are collected, write a short conclusion here:
- best model (`pooled` vs GNN),
- best operating point (`--threshold`, `--pos-weight`),
- whether Step 3 `--min-match-sim` helps,
- and what tradeoff you prefer (recall‑first vs precision‑first).
