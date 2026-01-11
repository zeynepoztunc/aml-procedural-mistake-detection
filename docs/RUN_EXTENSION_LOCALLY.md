# Run the Full Extension Locally (No Jupyter)

The extension in this repo is implemented as notebooks (`extension_step*.ipynb`). This runner executes them locally without installing Jupyter by converting code cells into a temporary `.py` script and running it.

## Prerequisites

- Run from the `aml-procedural-mistake-detection/` folder.
- You need the local data/features the notebooks expect (at minimum: `data/features/egovlp/`, `data/video/omnivore/`, and `annotations/annotation_json/step_annotations.json`).
- Recommended env:
  - GPU: `.\.venv-ext-cuda\Scripts\python.exe` (Python 3.12 + CUDA torch + PyG)
  - CPU fallback: `.\.venv-ext\Scripts\python.exe`

## Run All Steps (1 → 4)

From `aml-procedural-mistake-detection/`:

```powershell
.\.venv-ext-cuda\Scripts\python.exe tools\run_extension_local.py --steps 1 2 3 4
```

Outputs (generated scripts + logs) are written under `aml-procedural-mistake-detection/tools/_runs/extension_local/`.

## Outputs

After a successful full run you should have:

- Step 1: `extension_data/step_embeddings_gt.pkl`
- Step 2: `extension_data/task_verification_results.json` (+ plots like `extension_data/task_verification_baselines.png`)
- Step 3: `extension_data/realized_task_graphs.pkl` (+ plots like `extension_data/matching_analysis.png`)
- Step 4: `extension_data/gnn_results.json` (+ plots like `extension_data/gnn_comparison.png`)

## Notes / Caveats / Troubleshooting

- Recommended: use `.\.venv-ext-cuda\Scripts\python.exe` to run on GPU (CUDA).
- Fallback: `.\.venv-ext\Scripts\python.exe` is CPU-only.
- The default repo venv uses Python 3.13; PyTorch Geometric dependencies (e.g. `torch_scatter`) don’t have wheels there, so Step 4 won’t install/run.
- Cells that are Colab-only (Drive mounting, `google.colab`) are skipped automatically.
- Notebook magics like `%cd` and shell lines like `!pip install ...` are skipped.
- You still need the same local data/features the notebooks expect under `aml-procedural-mistake-detection/data/` (and any other paths referenced inside the notebooks).
- Step 3 uses HuggingFace `from_pretrained(...)` to load a text encoder; the first run may download weights.

## Speeding Up Step 2 (Optional)

The Step 2 notebook is expensive by default (3 models × leave-one-recipe-out × many epochs). For a quick smoke test:

```powershell
$env:EXT_NUM_EPOCHS="3"
$env:EXT_MAX_RECIPES="5"
.\.venv-ext-cuda\Scripts\python.exe tools\run_extension_local.py --steps 2
```

## Dry-run (generate scripts only)

```powershell
.\.venv-ext-cuda\Scripts\python.exe tools\run_extension_local.py --dry-run --steps 1 2 3 4
```
