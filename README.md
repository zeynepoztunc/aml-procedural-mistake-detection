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

Run in order. Each notebook writes its artefacts to `extension_data/` for the next.

| Substep | Notebook | Output |
|---|---|---|
| 1a. ActionFormer training | `extension_step1a_actionformer_train.ipynb` | `step1/checkpoints/cc4d_egovlp_1s/` |
| 1. Step localization | `extension_step1_actionformer.ipynb` | `step_embeddings_gt.pkl`, `step_embeddings_actionformer.pkl` |
| 2. Verification baselines | `extension_step2_verification_baseline.ipynb` | baseline metrics |
| 3. Task-graph encoding and matching | `extension_step3_task_graph_matching.ipynb` | `realized_task_graphs.pkl`, `realized_task_graphs_actionformer.pkl` |
| 4. GNN classification | `extension_step4_gnn_classification.ipynb` | `gnn_results.json` |

**Substep 1a** trains ActionFormer on our EgoVLP 1-second features, using the
CaptainCook4D fork `rohithpeddi/actionformer_release` pinned to a commit. It is a separate
notebook because it is a long GPU job that should not re-run every time the pipeline runs;
it promotes one checkpoint, its resolved config and a `provenance.json` to Drive. Validation
mAP over the fork's 353 step classes averages 36.4%.

**Substep 1** loads that checkpoint and produces step-level embeddings under two boundary
sources, both covering all 384 recordings: the ground-truth annotations (an oracle — the
annotation lists only the steps actually performed, so the segment count is itself
informative) and the trained detector. Post-processing — score threshold, class-agnostic
NMS overlap, minimum duration — is selected on the validation split alone, then frozen; the
test split is scored once.

**Substep 2** evaluates MLP, Transformer and LSTM baselines on the step embeddings under
nested leave-one-recipe-out, and compares the two boundary sources on identical recordings
and labels.

**Substep 3** encodes task-graph nodes with the EgoVLP text encoder and matches visual
steps to nodes with the Hungarian algorithm. Because the whole substep rests on EgoVLP's
video and text towers sharing a space, the notebook measures that alignment rather than
assuming it: video-to-text retrieval of a step's own description reaches **27.31% top-1
against a 7.15% chance rate**, with matched pairs scoring 0.112 cosine above mismatched
pairs. Genuine, but weak enough to bound what the matching can contribute downstream.

The notebook also scores the assignment itself. The match ratio is not a measure of
matching quality — Hungarian pads the cost matrix to a square and always returns
`min(num_visual_steps, num_nodes)` pairs — so Section 6a compares each matched segment's
own `step_id` against the `step_id` of the node it was assigned, alongside unconstrained
top-1 retrieval and the chance rate.

Matching alone loses the order of execution: visual features are written onto canonical
nodes and the canonical edges never change, so "A then B" and "B then A" realize
identically. Order Error is the largest error category in CaptainCook4D (795 instances
across 117 of 384 recordings), so every matched node also carries its position in the
observed sequence, its segment boundaries, and the count of canonical edges its position
contradicts. The canonical DAG remains the structure being judged; the observed chain is
stored alongside it, never in place of it.

**Substep 4** classifies the realized graphs with GraphSAGE, DAGNN — the layer the brief
names for DAGs — and a no-edge pooling control. The learnable projection of node and
visual features lives here rather than in Substep 3, because it can only be trained where
there is a loss to train it; the graph classification objective supplies the gradient. The
projection fuses the visual and text blocks and passes the order channels through
untouched.

### Results

Pooled nested leave-one-recipe-out over 24 recipes, 384 recordings under both boundary
sources. The epoch is selected on a held-out validation split of four recipes; no metric is
selected on the test fold. 57% of recordings are labelled incorrect, so balanced accuracy
and AUC are the metrics to read — a constant "incorrect" predictor already scores 72.8 F1.

**Substep 2 — sequence models over step embeddings.**

| Boundaries | Model | Bal. Acc | F1 | AUC |
|---|---|---|---|---|
| Annotated | MLP | 56.5 | 62.4 | 58.0 |
| Annotated | Transformer | 58.2 | 63.6 | 59.3 |
| Annotated | LSTM | 58.3 | 62.6 | 59.6 |
| ActionFormer | MLP | 56.0 | 62.8 | 62.2 |
| ActionFormer | Transformer | 58.4 | 63.1 | 62.6 |
| ActionFormer | LSTM | **60.9** | **64.9** | **64.1** |

Every one of the six sits below the majority F1 of 72.8. Verification from a flat sequence
of step embeddings is close to unlearnable at this data scale, which is the argument for
the graph formulation rather than a defect in it.

**Substep 4 — graph classification.**

| Boundaries | Model | Bal. Acc | AUC | F1 |
|---|---|---|---|---|
| Annotated | SimplePooling (no edges) | 75.6 | **80.8** | 73.2 |
| Annotated | GraphSAGE | **75.9** | 79.6 | 73.4 |
| Annotated | DAGNN | 75.4 | 79.0 | 74.2 |
| ActionFormer | SimplePooling (no edges) | **65.9** | **70.6** | 67.3 |
| ActionFormer | GraphSAGE | 63.6 | 66.6 | 63.1 |
| ActionFormer | DAGNN | 62.0 | 66.0 | 65.1 |
| — | Majority baseline | 50.0 | 50.0 | 72.8 |

`SimplePooling` receives identical node features and no edges at all, and it wins on AUC
under both boundary sources. DAGNN — the layer the brief names for DAGs, and the only one
here that consumes the edge directions — is last of the three. Neither margin clears the
run-to-run noise floor under annotated boundaries, so the honest reading is that the
task-graph topology contributes nothing measurable on top of the node features; under
predicted boundaries the no-edge control wins by a margin that does clear it.

Swapping annotated boundaries for predicted ones costs the graph models 10.2 (SimplePooling),
13.0 (GraphSAGE) and 13.0 (DAGNN) points of AUC, while the Substep 2 sequence models are
unaffected or slightly better under predicted boundaries. The asymmetry is the point: the
graph pipeline is reading the number of steps performed, which annotated boundaries hand it
for free. A deterministic rule — "a step is absent from the match" — reaches 79.1 AUC and
79.0 balanced accuracy under annotated boundaries with no features and no training, which is
every graph model in the table.

An ablation on GraphSAGE with identical folds and seeds gives the learnable projection
**+4.9 AUC and +7.0 balanced accuracy** over a fixed 0.5/0.5 average of the visual and text
features.

A depth study over 2, 4, 6 and 8 layers puts GraphSAGE at its best with 2 layers and
−8.5 AUC by 8, consistent with oversmoothing on graphs this small.

A nine-setting sweep over learning rate and hidden size spans 5.5 AUC points, and two runs
of an identical configuration have differed by 2.0. Treat roughly 3 AUC points as the floor
below which differences in these tables should not be interpreted.

## Notes

The notebooks are written for Google Colab: the first cell of each mounts Drive, clones
this repository and installs dependencies. They also run locally if `extension_data/` and
the feature directories are present.

## Acknowledgements

Builds on the CaptainCook4D release. See the original codebases for details.

- Error recognition: https://github.com/CaptainCook4D/error_recognition
- Feature extraction: https://github.com/CaptainCook4D/feature_extractors
- Step localization: https://github.com/CaptainCook4D/multi_step_localization
