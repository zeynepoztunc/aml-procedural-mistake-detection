# AML/DAAI 2025 — Procedural Mistake Detection on CaptainCook4D

Error recognition baselines on the CaptainCook4D dataset, plus a four-substep extension
that verifies whether a recorded execution of a recipe is correct by matching it against
the recipe's task graph and classifying the resulting graph.

## Environment setup

```
python -m venv .venv
pip install -r requirements.txt
```

Then download the pre-extracted features for 1s segments and place them in
`data/features`.

## Step 1: Baseline reproduction

Download the official checkpoints from
[here](https://utdallas.app.box.com/s/uz3s1alrzucz03sleify8kazhuc1ksl3)
(`error_recognition_best` directory) and place them in `checkpoints`, then run the
error-recognition evaluation.

```
python -m core.evaluate --variant MLP --backbone omnivore \
  --ckpt checkpoints/error_recognition_best/MLP/omnivore/error_recognition_MLP_omnivore_step_epoch_43.pt \
  --split step --threshold 0.6
```

This reproduces results close to Table 2 of the paper:

| Split | Model | F1 | AUC |
|-------|-------|----|-----|
| Step | MLP (Omnivore) | 24.26 | 75.74 |
| Recordings | MLP (Omnivore) | 55.42 | 63.03 |
| Step | Transf. (Omnivore) | 55.39 | 75.62 |
| Recordings | Transf. (Omnivore) | 40.73 | 62.27 |

Use the thresholds from the official project README when reproducing the published
checkpoints: 0.6 for `step`, 0.4 for `recordings`. Note that `core/config.py` documents
0.5 for the `recordings` split, and every training notebook in this repository uses 0.5;
figures produced here are therefore not directly comparable to a 0.4-thresholded number.

## Step 2: EgoVLP backbone

`colab_feature_extraction.ipynb` extracts EgoVLP video features, and
`train_egovlp_baseline.ipynb` trains the V1 (MLP) and V2 (Transformer) error-recognition
baselines on them, on both splits.

**Features must be extracted at a 1-second window and stride.** The dataloader indexes
feature rows with raw annotation timestamps in seconds, so any other stride misaligns the
index and silently yields empty slices for the later steps of each recording — which
produces a NaN loss and, at evaluation, fabricated targets. `SEGMENT_LENGTH = 1` in the
extraction notebook enforces this, and `base.py` asserts rather than substituting values
when a slice comes back empty.

`train_lstm_baseline_v2.ipynb` and `error_type_analysis.ipynb` cover the LSTM variant and
the per-error-category breakdown, both on Omnivore/SlowFast features.

## Extension: task-graph verification

Four notebooks, run in order. Each writes its artefacts to `extension_data/` for the next.

| Substep | Notebook | Output |
|---|---|---|
| 1. Step localization | `extension_step1_actionformer.ipynb` | `step_embeddings_gt.pkl`, `step_embeddings_actionformer.pkl` |
| 2. Verification baselines | `extension_step2_verification_baseline.ipynb` | baseline metrics |
| 3. Task-graph encoding and matching | `extension_step3_task_graph_matching.ipynb` | `realized_task_graphs.pkl` |
| 4. GNN classification | `extension_step4_gnn_classification.ipynb` | `gnn_results.json` |

**Substep 1** produces step-level embeddings under two boundary sources: ground-truth
annotations (383 recordings) and the official pre-trained CaptainCook4D ActionFormer
predictions (119 recordings). The pre-trained model is published as predictions rather
than weights, and was trained on Omnivore 4s features, so it cannot be re-run on the
EgoVLP features used here. Measured against ground truth its mean best-tIoU per step is
0.741. A from-scratch ActionFormer is also included as an alternative path.

**Substep 2** evaluates MLP, Transformer and LSTM baselines on the step embeddings, and
compares ground-truth against predicted boundaries on the 119 recordings both cover.

**Substep 3** encodes task-graph nodes with the EgoVLP text encoder and matches visual
steps to nodes with the Hungarian algorithm. Because the whole substep rests on EgoVLP's
video and text towers sharing a space, the notebook measures that alignment rather than
assuming it: video-to-text retrieval of a step's own description reaches **27.31% top-1
against a 7.15% chance rate**, with matched pairs scoring 0.112 cosine above mismatched
pairs. Genuine, but weak enough to bound what the matching can contribute downstream.

**Substep 4** classifies the realized graphs. The learnable projection of node and visual
features lives here rather than in Substep 3, because it can only be trained where there
is a loss to train it — the graph classification objective supplies the gradient.

### Results

Pooled nested leave-one-recipe-out over 24 recipes, 383 recordings. The epoch is selected
on a held-out validation split of four recipes; no metric is selected on the test fold.

| Model | Bal. Acc | AUC | F1 |
|---|---|---|---|
| Majority baseline | 50.0 | 50.0 | 72.8 |
| SimplePooling (no edges) | **78.2** | 80.9 | 73.3 |
| GCN | 75.1 | **83.9** | 75.9 |
| GAT | 74.9 | 82.2 | 73.4 |
| GraphSAGE | 75.0 | 80.1 | 72.4 |

57% of recordings are labelled incorrect, so balanced accuracy and AUC are the metrics to
read; a constant "incorrect" predictor already scores 72.8 F1.

`SimplePooling` receives identical node features but no edges, and is competitive
throughout — the task-graph topology contributes about 3 AUC points via GCN and costs
about 3 points of balanced accuracy. Most of the signal lies in the node features.

An ablation on GraphSAGE with identical folds and seeds gives the learnable projection
**+3 AUC and +4 balanced accuracy** over a fixed 0.5/0.5 average of the visual and text
features, consistent across two runs.

A depth study over 2, 4, 6 and 8 layers finds oversmoothing to be architecture-dependent:
GCN degrades monotonically (−3 to −4 AUC from 2 to 8 layers) while GAT and GraphSAGE stay
flat, in both runs.

Repeated runs of an identical configuration differ by roughly 0.4–0.7 AUC points, since
the graph convolutions use non-deterministic CUDA scatter operations. Differences smaller
than that should not be interpreted.

## Notes

The notebooks are written for Google Colab: the first cell of each mounts Drive, clones
this repository and installs dependencies. They also run locally if `extension_data/` and
the feature directories are present.

## Acknowledgements

Builds on the CaptainCook4D release. See the original codebases for details.

- Error recognition: https://github.com/CaptainCook4D/error_recognition
- Feature extraction: https://github.com/CaptainCook4D/feature_extractors
- Step localization: https://github.com/CaptainCook4D/multi_step_localization
