# Extension Pipeline Results Summary

This file records the **baseline** and **tuned/optimal** outcomes for each extension step, referencing saved artifacts in `extension_data/` and the write-ups in `latex_report/report.tex`.

## Step 1 - ActionFormer step localization
- **GT/oracle baseline**: `extension_data/step_embeddings_gt.pkl` provides perfect step boundaries, so it defines the upper bound for localization-sensitive downstream work.
- **Tuned ActionFormer** (`extension_data/step_embeddings_actionformer.pkl`, checkpoint `extension_data/actionformer_best.pt`):
  - **Greedy segment IoU vs. GT**: approx 0.603 on the test split.
  - **Boundary F1 at plus/minus 2 seconds**: approx 0.256 (best tuned run still struggles with very precise boundaries).
  - **Loss recipe**: focal classification + DIoU regression (see `extension_steps/step1/actionformer.py`).
  - **Command**: refer to `extension_steps/optimal_extension_configs/README.md` for the positive-weight, threshold, and smoothing arguments that produced this run.
  - **Evaluation helper**: `extension_steps/step1/compare_segments.py` supports multi-tolerance IoU/F1 evaluations.

## Step 2 - Task verification baselines
- **CaptainCook4D baselines (V1/V2)**:
  - **MLP (V1)**: accuracy approx 58-60%, F1 approx 0.49 (see `stats/error_recognition/Transformer/egovlp/None_training_performance.txt`).
  - **Transformer (V2)**: similar metrics with more variation across error types (`docs/analysis/error_type_analysis.md`).
  - **LSTM (custom "V3")**: F1 approx 58.41% reported in `latex_report/report.tex`.
- **Optimal recall-first run** (`extension_data/reports/step2_actionformer_mlp_recall_first.json`):
  - F1 0.7448, accuracy 0.6802, AUC 0.8198 (leave-one-recipe-out) using ActionFormer embeddings.
  - Command uses `--model mlp`, `--pos-weight 6`, `--threshold 0.50`.
- **Precision-first alternative**: check `extension_data/reports/step2_actionformer_mlp_precision_first.json` (higher threshold, lower recall).
- **Sweeps**: `extension_data/sweeps/step2_sweep.csv` shows how `pos_weight` and threshold affect F1 and AUC.

## Step 3 - Task graph matching
- **Baseline Hungarian matching**: `extension_data/realized_task_graphs.pkl` stores the realized graphs consumed by Step 4.
  - Cosine similarities and match behavior are summarized in `extension_data/plots/matching_analysis.png` and the LaTeX paper.
  - Optional low-sim variant (`--min-match-sim 0.2`) writes `extension_data/realized_task_graphs_minsim0.2.pkl` and flags unmatched nodes.
- **Text projection**: ridge weights cached at `extension_data/step3_text_to_visual_W.npy` map node texts into the Step 1 embedding space.
- **Inspection**: `extension_steps/step3/summarize_realized_graphs.py --pkl extension_data/realized_task_graphs.pkl --top-n 10` prints match counts, similarities, and recipe labels.

## Step 4 - Graph classification (GNN)
- **Pooled baseline (`--model pooled`)**: recorded in `extension_data/reports/step4_pooled.json` (no torch-geometric required).
- **Optimal GraphSAGE** (`extension_data/reports/step4_sage_baseline_l2.json`):
  - GraphSAGE with 2 layers, `pos_weight 6`, threshold 0.30: F1 approx 75.75%, accuracy approx 72.33%, AUC approx 82.25%.
  - Recall-first threshold tuning boosts F1 approx 78.09% with recall approx 0.848 (`latex_report/report.tex`).
  - Sweeps in `extension_data/sweeps/step4_sweep_recall.csv` cover threshold/pos_weight trade-offs.
- **Comparisons**: GraphSAGE beats pooled, GCN, and GAT (see `extension_data/plots/gnn_comparison.png` and Table 7 in the paper).

## References
- Artifacts: `extension_data/reports/`, `extension_data/sweeps/`, `extension_data/plots/`.
- Command recipes: `extension_steps/optimal_extension_configs/README.md`.
- Narrative: `latex_report/report.tex`.
