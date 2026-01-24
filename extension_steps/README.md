# Extension Steps Overview

Each subfolder under `extension_steps/` corresponds to one stage of the task verification extension pipeline described in the AML-2025 brief:

1. **`step1/` - Step Localization (ActionFormer).** Contains notebooks, configs, and helpers for temporal action localization. This stage outputs `(start,end)` timestamps per detected step and the associated feature averages (EgoVLP features are expected) used by downstream stages.
2. **`step2/` - Simple Task-Verification Baselines.** Houses training scripts and configs for recipe-level baselines (e.g., Transformer over step embeddings) with leave-one-recipe-out evaluation.
3. **`step3/` - Task-Graph Matching.** Provides the Hungarian-matching logic, text encoding helpers, and realized graph serialization used to align detected steps with task-graph nodes.
4. **`step4/` - GNN Classification.** Implements graph neural networks (GCN/GAT/GraphSAGE variants) that operate on the realized task graphs to predict recipe correctness.

At the root of `extension_steps/` there are lightweight driver scripts (`step1_actionformer.py`, `step1_pipeline.py`, `step2_verification_baseline.py`, `step3_task_graph_matching.py`, `step4_gnn_classification.py`) that orchestrate each stage when you want to run it outside the notebooks (used by `tools/run_extension_notebooks.py`).

Use this README as your starting point when you need to rerun or extend any of the four stages: each folder contains its own README/notebook plus configuration templates that control dataset paths, backbones, and hyperparameters. Follow the numbered steps to keep outputs (embeddings, realized graphs, metrics) synced with the artifacts saved under `extension_data/`.

## Optimal Commands

The `extension_steps/optimal_extension_configs/` directory collects the paste-ready commands we used to reproduce the best-known runs. Read `optimal_extension_configs/README.md` first, then:

1. Step 1 (ActionFormer): run `extension_steps/step1/actionformer.py` with the recommended arguments to get `extension_data/step_embeddings_actionformer.pkl` and `extension_data/actionformer_best.pt`. Use `step1/gt_pipeline.py` if you just want the oracle GT segments, and `step1/compare_segments.py` to evaluate IoU/boundary metrics.
2. Step 2 (Task verification baseline): use `python -m extension_steps.step2.main` with `--embeddings-pkl extension_data/step_embeddings_actionformer.pkl` (or GT) and the `--model`, `--pos-weight`, `--threshold` hyperparameters from the optimal README to produce the recall-first or precision-first JSON reports.
3. Step 3 (Matching): run `extension_steps/step3/task_graph_matching.py` with `--step1-pkl` pointing to the embeddings you want (GT or ActionFormer) and set `--out-pkl` to one of the realized graph files. Optional flags allow you to emit low-sim unmatched nodes for experimentation.
4. Step 4 (GNN classification): run `python -m extension_steps.step4.main` using the realized graphs and the GraphSAGE configuration (2 layers, pos-weight, threshold, device) shown in `optimal_extension_configs/README.md`. Use the `step4.sweep` script for sweeps over thresholds and pos-weights if you need comparisons.

These paste-ready commands capture the "optimal" results recorded in the CONCLUSIONS documents; feel free to wrap them into batch scripts or run them from `tools/run_extension_notebooks.py` if you want to reproduce the pipeline end-to-end.
