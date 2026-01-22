# Run Extension Notebooks Locally

This branch’s extension work (Step 1–4) is primarily implemented in notebooks:

- `extension_step1_pipeline.ipynb` (GT step boundaries → step embeddings)
- `extension_step1_actionformer.ipynb` (ActionFormer-based step localization experiments)
- `extension_step2_verification_baseline.ipynb`
- `extension_step3_task_graph_matching.ipynb`
- `extension_step4_gnn_classification.ipynb`

## 1) Environment

Create and activate a virtualenv in the repo root:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

**Extra deps used by the extension notebooks** (not listed in `requirements.txt`):

- `transformers`, `scipy`, `networkx`
- (Step 4) `torch-geometric` (+ its platform-specific wheels)

If you already have `.venv-ext` / `.venv-ext-cuda` in this repo, prefer using that environment for the extension notebooks.

You can sanity-check your local environment with:

```bash
python tools/run_extension_local.py --check-only
```

## 1b) Run everything via a script (no UI)

This executes the notebooks in order (and writes logs + executed notebooks under `tools/_runs/extension_local_notebooks/`):

```bash
python tools/run_extension_notebooks.py
```

Note: Step 2 contains an expensive grid-search section; the runner skips those cells by default.
Enable them explicitly with:

```bash
python tools/run_extension_notebooks.py --steps step2 --grid-search
```

If you want to run only some steps:

```bash
python tools/run_extension_notebooks.py --steps step1,step2
```

## 2) Data layout expected by notebooks

From the repo root:

- EgoVLP features: `data/features/egovlp/` (the notebooks load `*.npz` from here)
- Annotations: `annotations/annotation_json/step_annotations.json`
- Splits: `er_annotations/recordings_combined_splits.json`

The notebooks write outputs to:

- `extension_data/` (created automatically by Step 1; later notebooks expect it to exist)

## 3) Recommended local run order

### A) Ground-truth boundaries (simpler baseline)

1. Run `extension_step1_pipeline.ipynb` (produces `extension_data/step_embeddings_gt.pkl`)
2. Run `extension_step2_verification_baseline.ipynb`
3. Run `extension_step3_task_graph_matching.ipynb` (produces `extension_data/realized_task_graphs.pkl`)
4. Run `extension_step4_gnn_classification.ipynb`

### B) ActionFormer-style boundaries (automatic segmentation)

1. Run `extension_step1_actionformer.ipynb` (produces ActionFormer-derived boundaries and step embeddings, depending on the method you select in the notebook)
2. Ensure downstream notebooks point to the ActionFormer output (if they currently load `extension_data/step_embeddings_gt.pkl`, switch to the ActionFormer feature file you produced).

## 4) Common local issues

- If you start from Step 2+ without running Step 1 first, create the output folder:
  - `mkdir extension_data`
- If a notebook errors on missing packages, install the missing deps in your active environment.

## 5) Recording runs

Keep a short record of your completed runs (commands + key metrics) under `runs/`.
