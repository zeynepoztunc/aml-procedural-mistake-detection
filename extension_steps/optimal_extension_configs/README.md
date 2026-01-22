# Optimal Extension Configs (Paste‑Ready Commands)

This folder is a single place to copy/paste the “best known” commands for each extension step.

These configs reflect the experiments recorded in:
- `extension_steps/step1/CONCLUSIONS.md`
- `extension_steps/step2/CONCLUSIONS.md`
- `extension_steps/step3/CONCLUSIONS.md`
- `extension_steps/step4/CONCLUSIONS.md`

See also:
- `extension_steps/optimal_extension_configs/BASELINE_VS_OPTIMAL.md` (baseline vs tuned comparison)

> Tip: run from repo root: `aml-procedural-mistake-detection/`

---

## Step 1 — Step segmentation + step embeddings

### A) Oracle / GT pipeline (recommended for sanity checks)

Produces: `extension_data/step_embeddings_gt.pkl`

```bash
python extension_steps/step1/gt_pipeline.py
```

### B) ActionFormer pipeline (best known config)

Produces: `extension_data/step_embeddings_actionformer.pkl` and a best checkpoint in `extension_data/actionformer_best.pt`.

This is the best balanced run we recorded (strong IoU, improved boundary metrics):

```bash
python extension_steps/step1/actionformer.py ^
  --align-to-gt ^
  --select-best-by score_iou_count ^
  --threshold 0.5 ^
  --min-seg-len 8 ^
  --smooth-window 5 ^
  --smooth-type box ^
  --boundary-label-mode gaussian ^
  --boundary-window 2 ^
  --boundary-sigma 1.0 ^
  --pos-weight 6 ^
  --loss-type focal ^
  --focal-gamma 1.0
```

Evaluate Step 1 vs GT (test):

```bash
python extension_steps/step1/compare_segments.py ^
  --pred-pkl extension_data/step_embeddings_actionformer.pkl ^
  --split test ^
  --boundary-tols-sec 1 2 3 4 5 6
```

---

## Step 2 — Task verification (recording-level)

Step 2 classifies a whole recording as error/no-error from Step 1 step embeddings.

### Recommended default (recall-first)

```bash
python -m extension_steps.step2.main ^
  --embeddings-pkl extension_data/step_embeddings_actionformer.pkl ^
  --model mlp ^
  --pos-weight 6 ^
  --threshold 0.50 ^
  --out-json extension_data/reports/step2_actionformer_mlp_recall_first.json
```

### Precision-first alternative

```bash
python -m extension_steps.step2.main ^
  --embeddings-pkl extension_data/step_embeddings_actionformer.pkl ^
  --model mlp ^
  --pos-weight 3 ^
  --threshold 0.70 ^
  --out-json extension_data/reports/step2_actionformer_mlp_precision_first.json
```

---

## Step 3 — Task graph matching (realized graphs for Step 4)

Step 3 consumes a Step 1 embeddings pickle and produces realized task graphs:

Produces: `extension_data/realized_task_graphs.pkl`

### Recommended default (baseline Hungarian)

```bash
python extension_steps/step3/task_graph_matching.py ^
  --step1-pkl extension_data/step_embeddings_gt.pkl ^
  --out-pkl extension_data/realized_task_graphs.pkl
```

### Optional: enable “unmatched nodes” signal (experimental)

```bash
python extension_steps/step3/task_graph_matching.py ^
  --step1-pkl extension_data/step_embeddings_gt.pkl ^
  --out-pkl extension_data/realized_task_graphs_minsim0.2.pkl ^
  --min-match-sim 0.2 ^
  --unmatched-node-features text
```

Quick summary / sanity check:

```bash
python extension_steps/step3/summarize_realized_graphs.py ^
  --pkl extension_data/realized_task_graphs.pkl ^
  --top-n 10
```

---

## Step 4 — Graph classification (final)

Step 4 consumes realized task graphs and predicts error/no-error using LORO CV.

### Best overall (GraphSAGE, 2 layers) + recall-first operating point

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

### More balanced alternative (slightly lower recall, higher precision)

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

### Sweep (recall-first, precision second)

```bash
python -m extension_steps.step4.sweep ^
  --graphs-pkl extension_data/realized_task_graphs.pkl ^
  --model sage ^
  --num-layers 2 ^
  --pos-weight 2 3 4 5 6 ^
  --thresholds 0.30 0.35 0.40 0.45 0.50 ^
  --top-n 10 ^
  --out-csv extension_data/sweeps/step4_sweep_recall.csv
```
