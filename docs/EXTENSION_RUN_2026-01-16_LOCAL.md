# Extension Pipeline Local Run (2026-01-16)

This document records one full local run of the extension pipeline (Steps 1–4) and the resulting metrics/artifacts.

## Setup

- **Repo:** `aml-procedural-mistake-detection`
- **Branch:** `NadimDev3`
- **Runner:** `tools/run_extension_notebooks.py`
- **Environment:** `.venv-pyg` (CUDA-enabled; GPU observed in `nvidia-smi`)
- **Data inputs**
  - EgoVLP features: `data/features/egovlp/*_360p_224.npz` (key `video_features`)
  - Step annotations: `annotations/annotation_json/step_annotations.json`
  - Splits: `er_annotations/recordings_combined_splits.json`

### Commands used

```powershell
.\.venv-pyg\Scripts\python.exe tools\run_extension_notebooks.py --steps step1
.\.venv-pyg\Scripts\python.exe tools\run_extension_notebooks.py --steps step2
.\.venv-pyg\Scripts\python.exe tools\run_extension_notebooks.py --steps step3
.\.venv-pyg\Scripts\python.exe tools\run_extension_notebooks.py --steps step4
```

Notes:
- Step 2 has a very slow grid-search section; the runner skips those cells by default. Enable explicitly with:
  - `.\.venv-pyg\Scripts\python.exe tools\run_extension_notebooks.py --steps step2 --grid-search`
- Step 3 was executed without cloning/downloading EgoVLP inside the notebook (local patch via the runner) and used a fallback text→visual alignment derived from Step 1 GT `(description -> step_embedding)` pairs.

## Outputs (files)

- Step 1: `extension_data/step_embeddings_gt.pkl`
- Step 2: `extension_data/task_verification_results.json`, `extension_data/task_verification_baselines.png`
- Step 3: `extension_data/realized_task_graphs.pkl`, `extension_data/matching_analysis.png`, `extension_data/sample_realized_graph.png`
- Step 4: `extension_data/gnn_results.json`, `extension_data/gnn_comparison.png`

## Results

### Step 2 — Task Verification baselines (LO-Recipe-Out CV)

Source: `extension_data/task_verification_results.json`

Config:
```json
{"num_epochs":50,"learning_rate":0.0001,"batch_size":32,"pos_weight":2.0,"hidden_dim":256,"feature_dim":256}
```

Mean metrics (across folds):
- `Transformer`: F1 **0.7448**, Acc 0.6802, AUC 0.8198, Prec 0.6917, Rec 0.8661
- `LSTM`: F1 0.7036, Acc 0.6626, AUC 0.7130
- `MLP`: F1 0.7000, Acc 0.5792, AUC 0.6432 (high recall / lower precision)

### Step 3 — Task Graph Matching

Source: `extension_data/realized_task_graphs.pkl`

Config:
```json
{"feature_dim":256,"text_encoder":"sentence-transformers-fallback","text_encoder_aligned":false,"matching":"hungarian"}
```

Interpretation:
- Matching uses cosine similarity + Hungarian assignment between video step embeddings and per-recipe graph node embeddings.
- `text_encoder_aligned=false` means text embeddings were not produced by the original aligned EgoVLP text encoder in this run.

### Step 4 — Graph Classification (GNNs on realized graphs)

Source: `extension_data/gnn_results.json`

Config:
```json
{"num_epochs":50,"learning_rate":0.001,"batch_size":32,"pos_weight":2.0,"hidden_dim":128,"feature_dim":256}
```

Mean metrics (across folds):
- `GraphSAGE`: F1 **0.7564**, Acc 0.7233, AUC 0.8266, Prec 0.7628, Rec 0.7924
- `GCN`: F1 0.7329, Acc 0.6905, AUC 0.7904
- `GAT`: F1 0.7189, Acc 0.7086, AUC 0.7650
- `SimplePooling`: F1 0.7105, Acc 0.6593, AUC 0.7311

## Summary / Takeaways

- Best Step 2 baseline: **Transformer F1 0.7448**.
- Best Step 4 model: **GraphSAGE F1 0.7564**.
- Improvement from adding graph reasoning (GraphSAGE vs best baseline): **+0.0116 F1** on this run/config.
