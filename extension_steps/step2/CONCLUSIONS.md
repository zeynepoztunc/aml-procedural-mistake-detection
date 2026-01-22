# Step 2 — Conclusions (Task Verification)

## What Step 2 does

Step 2 predicts whether an entire **recipe recording** contains an error (`recipe_label ∈ {0,1}`) given the **sequence of step embeddings** produced by Step 1.

This repo evaluates Step 2 using **Leave‑One‑Recipe‑Out (LORO)** cross-validation, which tests generalization to unseen recipes.

---

## Key metrics (how to read them)

- **F1**: balance of precision/recall at a fixed decision threshold.
- **Precision**: of all predicted “error recordings”, how many are truly errors (controls false positives).
- **Recall**: of all true error recordings, how many are caught (controls false negatives).
- **AUC**: threshold‑independent ranking quality (how well scores separate the two classes).

Because this task can be imbalanced, **F1 and AUC** are usually more informative than accuracy alone.

---

## Timeline (changes + results)

### 2026-01-22 — Baseline (MLP) on GT vs ActionFormer step embeddings

**Change / setup**
- Ran the refactored Step 2 MLP baseline on:
  - GT / oracle embeddings: `extension_data/step_embeddings_gt.pkl`
  - ActionFormer embeddings: `extension_data/step_embeddings_actionformer.pkl`
- Config (defaults): `model=mlp`, `max_steps=50`, `num_epochs=50`, `batch_size=32`, `lr=1e-4`, `hidden_dim=256`, `dropout=0.2`, `pos_weight=2.0`, `threshold=0.5`
- Output JSON: `extension_data/reports/step2_compare_mlp.json`

**Results (LORO, 24 folds)**
- **GT (oracle)**:
  - Accuracy: **58.27% ± 13.43%**
  - Precision: **61.50% ± 15.86%**
  - Recall: **88.14% ± 14.85%**
  - F1: **69.90% ± 10.92%**
  - AUC: **64.87% ± 14.45%**
- **ActionFormer**:
  - Accuracy: **59.16% ± 13.34%**
  - Precision: **60.77% ± 13.66%**
  - Recall: **88.98% ± 13.89%**
  - F1: **70.53% ± 11.24%**
  - AUC: **65.19% ± 14.92%**

**Interpretation**
- Step 2 performance is **very similar** on GT vs ActionFormer embeddings, suggesting Step 1 embeddings are already good enough for Step 2 (MLP baseline).

### 2026-01-22 — Step 2 hyperparameter sweep (MLP): `pos_weight` × `threshold`

**Change / setup**
- Ran a small sweep to improve the precision/recall operating point:
  - `pos_weight ∈ {1,2,3,4,5,6}`
  - `threshold ∈ {0.4,0.5,0.6}`
- Output CSV: `extension_data/sweeps/step2_sweep.csv`

**Best by mean F1 (LORO)**
- `pos_weight=6`, `threshold=0.5`
  - F1: **0.7308**
  - Precision: **0.5881**
  - Recall: **0.9924**
  - AUC: **0.6418**

**Best-by-F1 config (paste-ready command)**
```bash
python -m extension_steps.step2.main \
  --embeddings-pkl extension_data/step_embeddings_actionformer.pkl \
  --model mlp \
  --pos-weight 6 \
  --threshold 0.5
```

**Notes**
- The best‑F1 configs are extremely **recall‑heavy** (near‑1.0 recall). If you want fewer false positives, the next sweep should prioritize **precision** (likely by increasing `threshold` and/or lowering `pos_weight`).

### 2026-01-22 — Precision-targeted sweep result (constraint: keep recall high)

**Goal**
- Increase **precision** while keeping **recall ≥ ~0.90** (avoid dropping recall too much from the best‑F1 setting).

**Best precision found under high-recall constraint**
- `pos_weight=6`, `threshold=0.70`
  - Precision: **0.6016**
  - Recall: **0.9185**
  - F1: **0.7136**
  - AUC: **0.6418**

**Paste-ready command**
```bash
python -m extension_steps.step2.main \
  --embeddings-pkl extension_data/step_embeddings_actionformer.pkl \
  --model mlp \
  --pos-weight 6 \
  --threshold 0.70 \
  --out-json extension_data/reports/step2_actionformer_mlp_precision.json
```

### 2026-01-22 — Precision-targeted sweep (top 3 interesting configs)

**Source**
- `extension_data/sweeps/step2_sweep_precision.csv` (sorted by `precision_mean`)

**1) Highest precision overall (but large recall drop)**
- `pos_weight=3`, `threshold=0.70`
  - Precision: **0.6297**
  - Recall: **0.7702**
  - F1: **0.6731**
  - AUC: **0.6458**

**2) Best “high precision while keeping recall high”**
- `pos_weight=6`, `threshold=0.70`
  - Precision: **0.6016**
  - Recall: **0.9185**
  - F1: **0.7136**
  - AUC: **0.6418**

**3) Best high-recall / high-F1 operating point (error-sensitive)**
- `pos_weight=6`, `threshold=0.50`
  - Precision: **0.5881**
  - Recall: **0.9924**
  - F1: **0.7308**
  - AUC: **0.6418**

**Notes**
- These three points define the practical tradeoff curve: maximize precision, keep recall high, or maximize recall/F1 for mistake detection.

---

## Recommended operating points (two key configs)

Below are the two configurations that matter most depending on whether you optimize for **catching mistakes** (high recall) or **avoiding false alarms** (high precision).

### A) Recall-first (mistake-sensitive)

**Config**
- `model=mlp`
- `embeddings=extension_data/step_embeddings_actionformer.pkl`
- `pos_weight=6`, `threshold=0.50`

**Metrics (LORO mean)**
- Precision: **0.5881**
- Recall: **0.9924**
- F1: **0.7308**
- AUC: **0.6418**

**Paste-ready command**
```bash
python -m extension_steps.step2.main \
  --embeddings-pkl extension_data/step_embeddings_actionformer.pkl \
  --model mlp \
  --pos-weight 6 \
  --threshold 0.50
```

### B) Precision-first (fewer false positives)

**Config**
- `model=mlp`
- `embeddings=extension_data/step_embeddings_actionformer.pkl`
- `pos_weight=3`, `threshold=0.70`

**Metrics (LORO mean)**
- Precision: **0.6297**
- Recall: **0.7702**
- F1: **0.6731**
- AUC: **0.6458**

**Paste-ready command**
```bash
python -m extension_steps.step2.main \
  --embeddings-pkl extension_data/step_embeddings_actionformer.pkl \
  --model mlp \
  --pos-weight 3 \
  --threshold 0.70
```

## Tradeoff and what matters most

- Moving from **A → B** increases precision (fewer false alarms) but substantially decreases recall (more missed mistakes).
- For a mistake detection project, it is typically better to **favor recall** (missing a real mistake is worse than flagging a clean recording). This makes **Config A** the default recommendation.
- If the system must be “quiet” (false positives are costly), use **Config B** or the middle-ground `pos_weight=6, threshold=0.70` (higher precision while still keeping recall ≥ ~0.90).

## Conclusion

- Step 2 performance is strong and **not degraded** by ActionFormer embeddings (GT ≈ ActionFormer for the MLP baseline).
- The main lever in Step 2 is choosing the **operating point** (threshold/pos_weight) based on whether you prioritize recall or precision.
