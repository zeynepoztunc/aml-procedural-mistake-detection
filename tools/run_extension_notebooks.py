from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


NOTEBOOKS = [
    ("step1", "extension_step1_pipeline.ipynb"),
    ("step2", "extension_step2_verification_baseline.ipynb"),
    ("step3", "extension_step3_task_graph_matching.ipynb"),
    ("step4", "extension_step4_gnn_classification.ipynb"),
]

_STEP2_GRID_SEARCH_MARKERS = (
    "GRID SEARCH CONFIGURATION",
    "COMPLETE SYSTEMATIC GRID SEARCH",
    "GRID SEARCH SUMMARY",
    "GRID SEARCH RESULTS SUMMARY",
    "VISUALIZATION OF GRID SEARCH RESULTS",
    "SAVE GRID SEARCH RESULTS TO JSON",
    "run_grid_experiment(",
    "grid_search_results",
)


_STEP3_LOCAL_PATCH_MARKERS = (
    "Strict EgoVLP Loading with Helper Checks",
    "Cloning EgoVLP repository",
    "FrozenInTime",
    "vit_checkpoint_path",
    "Downloading missing ViT backbone",
)


def _maybe_strip_step2_grid_search(*, notebook: Path, scratch_notebook: Path, enabled: bool) -> Path:
    if enabled:
        return notebook

    try:
        import nbformat
    except Exception:
        return notebook

    nb = nbformat.read(str(notebook), as_version=4)
    new_cells = []
    removed = 0
    for cell in nb.cells:
        if cell.get("cell_type") == "code":
            src = cell.get("source") or ""
            if any(marker in src for marker in _STEP2_GRID_SEARCH_MARKERS):
                removed += 1
                continue
        new_cells.append(cell)

    if removed == 0:
        return notebook

    nb.cells = new_cells
    scratch_notebook.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(scratch_notebook))
    return scratch_notebook


def _maybe_patch_step3_for_local_run(*, notebook: Path, scratch_notebook: Path) -> Path:
    try:
        import nbformat
    except Exception:
        return notebook

    nb = nbformat.read(str(notebook), as_version=4)
    patched = False

    replacement_source = r"""
import os
import numpy as np
import torch
import torch.nn.functional as F

# Step3 local run: avoid cloning/downloading EgoVLP during notebook execution.
# Instead, learn a lightweight text->visual linear alignment from your Step1 GT data
# (step descriptions paired with EgoVLP-pooled step embeddings).

EGOVLP_AVAILABLE = False

sample_key = list(processed_data.keys())[0]
FEATURE_DIM = processed_data[sample_key]["step_embeddings"].shape[1]
print(f"[step3] Visual feature dimension: {FEATURE_DIM}")


def _hf_offline_enabled() -> bool:
    return bool(os.environ.get("HF_HUB_OFFLINE")) or bool(os.environ.get("TRANSFORMERS_OFFLINE"))


def _text_features_hashing(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import HashingVectorizer

    vec = HashingVectorizer(n_features=768, alternate_sign=False, norm=None)
    x = vec.transform(texts).toarray().astype(np.float32)
    return x


def _text_features_transformers(texts: list[str], *, batch_size: int = 32) -> np.ndarray:
    from transformers import AutoTokenizer, AutoModel

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=_hf_offline_enabled())
    model = AutoModel.from_pretrained(model_name, local_files_only=_hf_offline_enabled()).to(device)
    model.eval()

    all_emb = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
            last = out.last_hidden_state  # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
            pooled = (last * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1.0))
        all_emb.append(pooled.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(all_emb, axis=0) if all_emb else np.zeros((0, 768), dtype=np.float32)


def _get_text_features(texts: list[str]) -> np.ndarray:
    try:
        return _text_features_transformers(texts)
    except Exception as e:
        print(f"[step3] transformers text encoder unavailable ({type(e).__name__}: {e}); falling back to hashing.")
        return _text_features_hashing(texts)


def _fit_text_to_visual_projection(processed_data: dict, *, max_pairs: int = 6000, ridge_lambda: float = 10.0):
    texts = []
    visuals = []

    for rec in processed_data.values():
        descs = rec.get("descriptions") or []
        vis = rec.get("step_embeddings", rec.get("embeddings"))
        if vis is None:
            continue
        vis = np.asarray(vis)
        if vis.ndim != 2 or vis.shape[0] != len(descs):
            continue
        for d, v in zip(descs, vis):
            if not d:
                continue
            texts.append(str(d))
            visuals.append(v.astype(np.float32))

    if not texts:
        raise RuntimeError("No (description, embedding) pairs found to fit text->visual projection.")

    if len(texts) > max_pairs:
        idx = np.linspace(0, len(texts) - 1, max_pairs).astype(int).tolist()
        texts = [texts[i] for i in idx]
        visuals = [visuals[i] for i in idx]

    X = _get_text_features(texts)  # (N, 768)
    Y = np.stack(visuals, axis=0).astype(np.float32)  # (N, D)

    # Ridge-regularized least squares: W = (X^T X + λI)^-1 X^T Y
    XtX = X.T @ X
    XtX += ridge_lambda * np.eye(XtX.shape[0], dtype=np.float32)
    XtY = X.T @ Y
    W = np.linalg.solve(XtX, XtY).astype(np.float32)  # (768, D)

    return torch.from_numpy(W)


print("[step3] Fitting text->visual projection from Step1 GT pairs...")
_TEXT_TO_VISUAL_W = _fit_text_to_visual_projection(processed_data)
print(f"[step3] Projection fitted: W shape={tuple(_TEXT_TO_VISUAL_W.shape)}")


def encode_texts(texts, normalize: bool = True):
    if isinstance(texts, str):
        texts = [texts]

    X = _get_text_features([str(t) for t in texts])  # (N, 768)
    X_t = torch.from_numpy(X).to(device)
    W = _TEXT_TO_VISUAL_W.to(device)
    Z = X_t @ W  # (N, D)
    if normalize:
        Z = F.normalize(Z, dim=1)
    return Z


# Quick sanity check
print("\n[step3] Testing text encoder...")
test_texts = ["Break the eggs", "Mix the ingredients", "Heat the pan"]
with torch.no_grad():
    test_emb = encode_texts(test_texts)
print(f"[step3] Test embeddings shape: {tuple(test_emb.shape)}")
""".lstrip()

    data_load_replacement_source = r"""
import os
import json
import pickle
from pathlib import Path

# Load processed step embeddings from Substep 1
DEFAULT_REL = Path("extension_data") / "step_embeddings_gt.pkl"
DATA_PATH = None

if "IN_COLAB" in globals() and IN_COLAB:
    drive_path = Path("/content/drive/MyDrive/AML_Project/extension_data/step_embeddings_gt.pkl")
    if drive_path.exists():
        DATA_PATH = str(drive_path)

if DATA_PATH is None:
    # Robust local lookup: search current directory and its parents.
    cwd = Path.cwd().resolve()
    for base in [cwd] + list(cwd.parents)[:8]:
        cand = base / DEFAULT_REL
        if cand.exists():
            DATA_PATH = str(cand)
            break

if DATA_PATH is None:
    raise FileNotFoundError(f"Could not find step1 output at '{DEFAULT_REL}'. CWD={Path.cwd()}")

REPO_ROOT = Path(DATA_PATH).resolve().parent.parent
os.chdir(REPO_ROOT)
print(f"[step3] Using repo root: {REPO_ROOT}")

print(f"[step3] Loading Step1 output from: {DATA_PATH}")
with open(DATA_PATH, "rb") as f:
    loaded_data = pickle.load(f)

processed_data = loaded_data["data"]
splits = loaded_data["splits"]
print(f"Loaded {len(processed_data)} recordings")

# Load step annotations for textual descriptions
ann_path = REPO_ROOT / "annotations" / "annotation_json" / "step_annotations.json"
with open(ann_path, "r", encoding="utf-8") as f:
    step_annotations = json.load(f)
""".lstrip()

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source") or ""
        if any(marker in src for marker in _STEP3_LOCAL_PATCH_MARKERS):
            cell["source"] = replacement_source
            patched = True
            continue
        if "DATA_PATH = \"extension_data/step_embeddings_gt.pkl\"" in src or "Load processed step embeddings from Substep 1" in src:
            cell["source"] = data_load_replacement_source
            patched = True
            continue

    if not patched:
        return notebook

    scratch_notebook.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(scratch_notebook))
    return scratch_notebook


def _repo_root_from_cwd() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "requirements.txt").exists():
        return cwd
    raise SystemExit("Run this from the repo root (the folder containing requirements.txt).")


def _parse_steps(raw: str) -> set[str]:
    normalized = raw.strip().lower()
    if normalized in {"all", "*"}:
        return {name for name, _ in NOTEBOOKS}

    steps = {s.strip().lower() for s in normalized.split(",") if s.strip()}
    valid = {name for name, _ in NOTEBOOKS}
    unknown = sorted(steps - valid)
    if unknown:
        raise SystemExit(f"Unknown steps: {unknown}. Valid: {sorted(valid)}")
    return steps


def _run_nbconvert(
    *,
    repo_root: Path,
    notebook: Path,
    output_notebook: Path,
    log_path: Path,
    timeout_sec: int,
    env: dict[str, str],
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        f"--ExecutePreprocessor.timeout={timeout_sec}",
        "--output",
        str(output_notebook),
        str(notebook),
    ]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n\n")
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute the extension notebooks (Step 1–4) using nbconvert."
    )
    parser.add_argument(
        "--steps",
        default="all",
        help="Comma-separated steps to run: step1,step2,step3,step4 (default: all).",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=3600,
        help="Per-notebook execution timeout in seconds (default: 3600).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running subsequent notebooks even if one fails.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Set HF/Transformers offline env vars (Step 3 will require cached models).",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run the (very slow) Step 2 grid-search cells.",
    )
    parser.add_argument(
        "--run-id",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Run id used under tools/_runs/extension_local_notebooks/<run-id>/",
    )
    args = parser.parse_args()

    repo_root = _repo_root_from_cwd()
    os.chdir(repo_root)

    if importlib.util.find_spec("jupyter") is None:
        raise SystemExit(
            "Missing dependency 'jupyter'. Install it in your active venv:\n"
            "  pip install jupyter nbconvert\n"
            "or:\n"
            "  pip install jupyterlab\n"
        )

    selected = _parse_steps(args.steps)
    run_root = repo_root / "tools" / "_runs" / "extension_local_notebooks" / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    if args.offline:
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"

    # Preflight check (data paths + imports)
    preflight_log = run_root / "preflight.log"
    with preflight_log.open("w", encoding="utf-8") as log:
        log.write("$ python tools/run_extension_local.py --check-only\n\n")
        subprocess.run(
            [sys.executable, "tools/run_extension_local.py", "--check-only"],
            cwd=str(repo_root),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    failures: list[str] = []
    for step_name, nb_name in NOTEBOOKS:
        if step_name not in selected:
            continue

        notebook = repo_root / nb_name
        if not notebook.exists():
            failures.append(f"{step_name} (missing notebook: {nb_name})")
            if not args.continue_on_error:
                break
            continue

        step_dir = run_root / step_name
        step_dir.mkdir(parents=True, exist_ok=True)
        out_nb = step_dir / nb_name
        log_path = step_dir / f"{notebook.stem}.log"

        nb_to_run = notebook
        if step_name == "step2":
            nb_to_run = _maybe_strip_step2_grid_search(
                notebook=notebook,
                scratch_notebook=step_dir / f"{notebook.stem}.no_grid_search.ipynb",
                enabled=args.grid_search,
            )
        if step_name == "step3":
            nb_to_run = _maybe_patch_step3_for_local_run(
                notebook=nb_to_run,
                scratch_notebook=step_dir / f"{notebook.stem}.local.ipynb",
            )

        print(f"[{step_name}] executing {nb_name} ...")
        rc = _run_nbconvert(
            repo_root=repo_root,
            notebook=nb_to_run,
            output_notebook=out_nb,
            log_path=log_path,
            timeout_sec=args.timeout_sec,
            env=env,
        )
        if rc != 0:
            failures.append(f"{step_name} (exit {rc})")
            print(f"[{step_name}] FAILED (see {log_path})")
            if not args.continue_on_error:
                break
        else:
            print(f"[{step_name}] OK (output {out_nb})")

    summary_path = run_root / "SUMMARY.txt"
    summary_lines = [
        f"run_id: {args.run_id}",
        f"steps: {', '.join(sorted(selected))}",
        f"offline: {args.offline}",
        f"timeout_sec: {args.timeout_sec}",
        f"failures: {', '.join(failures) if failures else 'none'}",
        f"logs/output: {run_root}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
