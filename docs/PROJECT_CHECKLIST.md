# AML-2025 Project Checklist (PDF-only)

This checklist is derived exclusively from
`aml-procedural-mistake-detection/docs/spec/AML-2025_Mistake_Detection_Project.pdf`
(v1.1, 5 Dec 2025). It intentionally includes only items explicitly requested in
that PDF.

## Step 1: Literature Review

- [x] Read CaptainCook4D [PL1] and grasp the dataset + task setup (evidence in
      report writeup:
      `aml-procedural-mistake-detection/latex_report/report.tex`)
- [x] Read recent Procedure Learning papers [PL2, PL3] and grasp main concepts
      (evidence in report writeup:
      `aml-procedural-mistake-detection/latex_report/report.tex`)

## Step 2: Mistake Detection baselines (offline, SupervisedER)

- [x] Download the pre-extracted features from the dataset release (Omnivore,
      SlowFast) (used for baseline reproduction per
      `aml-procedural-mistake-detection/latex_report/report.tex`)
- [x] Review the CaptainCook4D paper feature-extraction details (dimensions and
      usage documented in
      `aml-procedural-mistake-detection/latex_report/report.tex`):
  - [x] What are the inputs and outputs of the pre-trained models in the feature
        extraction phase?
- [x] Reproduce CaptainCook4D SupervisedER baselines (V1 and V2) (code +
      artifacts exist):
  - [x] V1: MLP head on top of pre-extracted sub-segment features
        (`aml-procedural-mistake-detection/base.py`, checkpoints under
        `aml-procedural-mistake-detection/checkpoints/`)
  - [x] V2: Transformer layer combining cues from all sub-segments in a recipe
        step (`aml-procedural-mistake-detection/core/models/er_former.py`)
  - [x] Use the same metrics as the paper: Accuracy, Precision, Recall, F1, AUC
        (metrics logged under `aml-procedural-mistake-detection/stats/` and
        reported in `aml-procedural-mistake-detection/latex_report/report.tex`)
- [x] Analyze model performance on different error types (in addition to the
      paper metrics) (`aml-procedural-mistake-detection/error_type_analysis.md`,
      `aml-procedural-mistake-detection/docs/analysis/error_type_analysis.md`)
- [x] Propose a new baseline and compare with V1/V2 (example from PDF: RNN/LSTM
      over the per-step sequence) (LSTM code + analysis exist:
      `aml-procedural-mistake-detection/core/models/lstm.py`,
      `aml-procedural-mistake-detection/LSTM_results_analysis.md`,
      `aml-procedural-mistake-detection/docs/analysis/LSTM_results_analysis.md`)
- [x] Extend the baselines to a new feature extraction backbone (EgoVLP present
      in repo):
  - [x] Adapt the CaptainCook4D feature extraction code to support a new
        backbone (feature extraction notebook present:
        `aml-procedural-mistake-detection/colab_feature_extraction.ipynb`)
  - [x] Consider EgoVLP [VB4] or PerceptionEncoder [VB5] (EgoVLP artifacts
        present: `aml-procedural-mistake-detection/data/features/egovlp/`,
        `aml-procedural-mistake-detection/checkpoints/egovlp.pth`)
  - [ ] Download the resized dataset version (per the PDF link) (not verifiable
        from repo)

## Extension: From Mistake Detection to Task Verification (recipe-level)

- [ ] Discuss the extension with the TA before starting (required by PDF) (not
      verifiable from repo)
- [x] Use EgoVLP or PerceptionEncoder features extracted in Step 2 (aligned
      video/text spaces) (EgoVLP features exist:
      `aml-procedural-mistake-detection/data/features/egovlp/`)
- [x] Formulation (documented in report): (evidence:
      `aml-procedural-mistake-detection/latex_report/report.tex`)
  - [x] Predict a single recipe-level binary label (correct vs incorrect
        execution)
  - [x] Do not use step-level correctness annotations (not required in this
        setting)

### Substep 1: Recipe step localization

- [x] Use a step localization approach to segment a recipe video into steps
      (ActionFormer-based pipeline exists):
  - [x] Pre-trained ActionFormer model provided as part of CaptainCook4D
        (artifacts:
        `aml-procedural-mistake-detection/extension_step1_actionformer.ipynb`,
        `aml-procedural-mistake-detection/extension_data/actionformer_best.pt`),
        or
  - [ ] *OPTIONAL* Zero-shot clustering-based approach like HiERO [PL4] *OPTIONAL*
- [x] Output a list of `(start, end)` timestamps for each step in the video
      (produced as part of Step 1 outputs; see run log:
      `aml-procedural-mistake-detection/docs/EXTENSION_RUN_2026-01-16_LOCAL.md`)
- [x] Compute a step-level embedding for each detected step by averaging video
      features inside `(start, end)` (outputs:
      `aml-procedural-mistake-detection/extension_data/step_embeddings_actionformer.pkl`,
      `aml-procedural-mistake-detection/extension_data/step_embeddings_gt.pkl`)
- [x] Produce a sequence of step-level embeddings per recipe video (same outputs
      as above)

### Substep 2: Simple task-verification baselines

- [x] Train a simple task-verification baseline on recipe videos using
      recipe-level binary labels (artifacts:
      `aml-procedural-mistake-detection/extension_step2_verification_baseline.ipynb`,
      `aml-procedural-mistake-detection/extension_data/reports/task_verification_results.json`)
- [x] Example baseline from PDF: Transformer layer over the step-embedding
      sequence + binary classification head (reported in:
      `aml-procedural-mistake-detection/docs/EXTENSION_RUN_2026-01-16_LOCAL.md`)
- [x] Consider a leave-one-out evaluation setting (train on `k-1` recipes, test
      on the `k`-th recipe) (implemented as LORO/LO-Recipe-Out in:
      `aml-procedural-mistake-detection/docs/EXTENSION_RUN_2026-01-16_LOCAL.md`)

### Substep 3: Task-Graph encoding + Step matching

- [x] Encode textual step descriptions of the ground-truth task graphs using the
      EgoVLP/PE text encoder (described in
      `aml-procedural-mistake-detection/latex_report/report.tex`)
- [x] Match detected steps to graph nodes with the Hungarian matching algorithm
      (artifacts:
      `aml-procedural-mistake-detection/extension_step3_task_graph_matching.ipynb`,
      `aml-procedural-mistake-detection/extension_data/realized_task_graphs.pkl`):
  - [x] Assume each visual step matches at most one node (and vice versa)
        (documented in report/run log)
- [x] Update matched node features using a learnable projection of node features
      and visual features (evidence of learned projection:
      `aml-procedural-mistake-detection/extension_data/step3_text_to_visual_W.npy`)

### Substep 4: Classification of the observed task-graph

- [x] Train a GNN-based classifier to predict recipe correctness from the
      realized task graph (artifacts:
      `aml-procedural-mistake-detection/extension_step4_gnn_classification.ipynb`,
      `aml-procedural-mistake-detection/extension_data/reports/gnn_results.json`)
- [x] Use the provided notebook as reference; optionally experiment with
      different GNN layers (example in PDF: DAGNN [O1]) (GCN/GAT/GraphSAGE
      experiments reported in
      `aml-procedural-mistake-detection/latex_report/report.tex`)

## Project Deliverables

- [ ] Produce an 8-page report using the CVPR template (LaTeX source exists but
      no built PDF in repo snapshot:
      `aml-procedural-mistake-detection/latex_report/report.tex`)
- [x] Follow a paper-like structure: abstract, introduction, related works,
      method, experiments, conclusion (present in
      `aml-procedural-mistake-detection/latex_report/report.tex`)
- [x] Clearly explain everything you did in the project, focusing on the
      implementation of the extension (present in
      `aml-procedural-mistake-detection/latex_report/report.tex`)
