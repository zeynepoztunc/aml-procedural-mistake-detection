# Baseline vs Optimal (Is this project a “win”?)

This document compares the **baseline** (defaults / first runnable refactor) against the **best configs we found** during tuning, so you can argue whether the project improved results.

All numbers below are taken from saved reports/sweeps in `extension_data/`.

---

## What counts as “baseline” here?

- “Baseline” = running the refactored steps with **default hyperparameters**, unless otherwise stated.
- For Step 2/4, evaluation is **Leave‑One‑Recipe‑Out (LORO)** with 24 folds, so we report mean ± std across folds.

---

## Step 2 — Task verification (recording-level MLP)

Baseline report (ActionFormer embeddings, default settings in `step2_compare_mlp.json`):
- `extension_data/reports/step2_compare_mlp.json` → method `actionformer`
  - Accuracy: **59.16% ± 13.34%**
  - Precision: **60.77% ± 13.66%**
  - Recall: **88.98% ± 13.89%**
  - F1: **70.53% ± 11.24%**
  - AUC: **65.19% ± 14.92%**

Optimal (recall-first) from the Step 2 sweep:
- `extension_data/sweeps/step2_sweep.csv` top by `f1_mean`
  - Config: `model=mlp`, `pos_weight=6`, `threshold=0.50`
  - Precision: **0.5881**
  - Recall: **0.9924**
  - F1: **0.7308**
  - AUC: **0.6418**

Win summary (Step 2):
- **Recall** improves significantly (≈ **0.89 → 0.99**) by changing the operating point (pos_weight/threshold).
- F1 improves (≈ **0.705 → 0.731**).
- AUC stays roughly similar (small changes); the main lever here is **operating point**, not ranking.

---

## Step 3 — Task graph matching (graph construction for Step 4)

Step 3 is not a classifier, so there is no “F1” to optimize directly at this step. The correct question is:

> Does a Step 3 change improve Step 4 downstream?

We implemented “unmatched nodes” (`--min-match-sim`) to expose low-quality Hungarian matches as a signal. It produced meaningful unmatched counts and strongly correlated with error labels, but:

Downstream impact (Step 4, GraphSAGE L=2):
- Baseline graphs: `extension_data/reports/step4_sage_baseline_l2.json`
  - F1: **75.75% ± 10.66%**, AUC: **82.25% ± 9.52%**
- With unmatched nodes (`min_match_sim=0.2`): `extension_data/reports/step4_sage_minsim0.2_l2.json`
  - F1: **75.50% ± 10.53%**, AUC: **81.51% ± 8.94%**

Conclusion (Step 3):
- The “unmatched nodes” feature is a **plausible signal**, but with the current Step 4 model it did **not** improve results.
- Keep it as an **experimental option**, but use baseline graphs for best Step 4 performance.

---

## Step 4 — Final graph classification (main project win)

This is the strongest place to claim a win, because Step 4 is the “final” model on realized graphs.

### Baseline (pooled, defaults)

Baseline pooled model (defaults):
- `extension_data/reports/step4_pooled.json`
  - Accuracy: **64.26% ± 14.70%**
  - Precision: **68.10% ± 16.57%**
  - Recall: **79.00% ± 20.95%**
  - F1: **70.06% ± 14.17%**
  - AUC: **72.38% ± 13.98%**

### Big win: switch baseline model → GraphSAGE (L=2)

GraphSAGE, 2 layers (defaults for `threshold=0.5`, `pos_weight=2.0`):
- `extension_data/reports/step4_sage_baseline_l2.json`
  - Accuracy: **71.94% ± 11.12%** (**+7.68 pts**)
  - Precision: **75.23% ± 16.74%** (**+7.13 pts**)
  - Recall: **79.80% ± 13.56%** (≈ same)
  - F1: **75.75% ± 10.66%** (**+5.69 pts**)
  - AUC: **82.25% ± 9.52%** (**+9.87 pts**)

### Additional win: tune for recall-first behavior

Recall-first sweep (priority: recall, then precision):
- `extension_data/sweeps/step4_sweep_recall.csv`
  - Config: `model=sage`, `num_layers=2`, `pos_weight=6`, `threshold=0.30`
  - Recall: **0.8480**
  - Precision: **0.7552**
  - F1: **0.7809**
  - AUC: **0.8288**

Compared to the SAGE default run (pos_weight=2, thr=0.5):
- Recall improves (≈ **0.798 → 0.848**).
- F1 improves (≈ **0.758 → 0.781**).
- Precision stays similar.

Win summary (Step 4):
- The project has a clear “win” if you define success as improved Step 4 performance:
  - **pooled → SAGE(2)** gives a large jump in both F1 and AUC.
  - tuning `pos_weight/threshold` improves recall further for mistake detection.

---

## Bottom line

- **Yes, it’s a win** if your headline metric is Step 4 F1/AUC (graph classification), because the tuned SAGE model clearly outperforms the pooled baseline.
- Step 2 also improves via operating point tuning (recall-first vs precision-first), but Step 2 is a parallel track (not used by Step 3/4).
- Step 3’s “unmatched nodes” is a useful analysis feature; it didn’t improve Step 4 yet, so it’s best presented as an attempted improvement with mixed outcome.

