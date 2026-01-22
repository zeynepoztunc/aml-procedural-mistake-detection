from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from extension_steps.step3.config import default_paths
from extension_steps.step3.io import load_json, load_pickle, save_pickle
from extension_steps.step3.realize import RealizeConfig, create_realized_graph
from extension_steps.step3.task_graphs import (
    build_task_graphs_from_step_annotations,
    normalize_task_graphs,
)
from extension_steps.step3.text_encoder import TextToVisualProjection, fit_text_to_visual_projection


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 3: Task-graph encoding + matching (refactored)")
    p.add_argument("--step1-pkl", type=str, default=None, help="Path to Step1 output .pkl")
    p.add_argument("--step-annotations-json", type=str, default=None, help="Path to step_annotations.json")
    p.add_argument("--task-graphs-json", type=str, default=None, help="Path to task_graphs.json (optional)")
    p.add_argument("--out-pkl", type=str, default=None, help="Output .pkl path")

    p.add_argument("--projection-cache", type=str, default=None, help="Cache path for text->visual W (.npy)")
    p.add_argument("--refit-projection", action="store_true", help="Force refit projection even if cache exists")
    p.add_argument(
        "--no-transformers",
        action="store_true",
        help="Disable transformers text features; use hashing features only",
    )
    p.add_argument("--max-projection-pairs", type=int, default=6000, help="Max (text,visual) pairs to fit W")
    p.add_argument("--ridge-lambda", type=float, default=10.0, help="Ridge regularization strength")

    p.add_argument("--alpha-visual", type=float, default=0.5, help="Mixing weight for matched nodes: alpha*visual+(1-alpha)*text")
    p.add_argument(
        "--min-match-sim",
        type=float,
        default=0.0,
        help=(
            "Minimum cosine similarity required to treat a Hungarian assignment as a valid match. "
            "Assignments below this threshold are kept for analysis (match_similarities) but the node is marked "
            "unmatched (node_matched=False)."
        ),
    )
    p.add_argument(
        "--unmatched-node-features",
        type=str,
        default="text",
        choices=("text", "zero"),
        help="What node_features should be for unmatched nodes: keep text embedding ('text') or set to zeros ('zero').",
    )
    p.add_argument("--limit-recordings", type=int, default=0, help="If >0, only process first N recordings (debug)")

    return p.parse_args(argv)


def _infer_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _get_feature_dim(processed_data: dict) -> int:
    sample_key = next(iter(processed_data.keys()))
    step_emb = processed_data[sample_key].get("step_embeddings", processed_data[sample_key].get("embeddings"))
    step_emb = np.asarray(step_emb)
    if step_emb.ndim != 2:
        raise ValueError("Step1 embeddings have unexpected shape.")
    return int(step_emb.shape[1])


def _load_or_build_task_graphs(*, task_graphs_json: Path, step_annotations: dict) -> dict:
    if task_graphs_json.exists():
        raw = load_json(task_graphs_json)
        return normalize_task_graphs(raw)
    return build_task_graphs_from_step_annotations(step_annotations)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    repo_root = _infer_repo_root()
    paths = default_paths(repo_root=repo_root)

    step1_pkl = Path(args.step1_pkl) if args.step1_pkl else paths.step1_pkl
    step_annotations_json = (
        Path(args.step_annotations_json) if args.step_annotations_json else paths.step_annotations_json
    )
    task_graphs_json = Path(args.task_graphs_json) if args.task_graphs_json else paths.task_graphs_json
    out_pkl = Path(args.out_pkl) if args.out_pkl else paths.out_pkl
    projection_cache = Path(args.projection_cache) if args.projection_cache else paths.projection_cache

    os.chdir(repo_root)

    print(f"[step3] Repo root: {repo_root}")
    print(f"[step3] Step1 pkl: {step1_pkl}")
    print(f"[step3] Step annotations: {step_annotations_json}")
    print(f"[step3] Task graphs: {task_graphs_json} ({'exists' if task_graphs_json.exists() else 'missing -> build'})")
    print(f"[step3] Output pkl: {out_pkl}")

    step1_payload = load_pickle(step1_pkl)
    processed_data = step1_payload["data"]
    splits = step1_payload.get("splits", {})
    feature_dim = int(step1_payload.get("feature_dim") or _get_feature_dim(processed_data))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[step3] Device: {device}")
    print(f"[step3] Visual feature dim: {feature_dim}")

    step_annotations = load_json(step_annotations_json)
    task_graphs = _load_or_build_task_graphs(task_graphs_json=task_graphs_json, step_annotations=step_annotations)
    print(f"[step3] Task graphs: {len(task_graphs)} recipe types")

    prefer_transformers = not bool(args.no_transformers)
    if projection_cache.exists() and not args.refit_projection:
        print(f"[step3] Loading cached projection: {projection_cache}")
        projection = TextToVisualProjection.load(
            projection_cache,
            feature_dim=feature_dim,
            prefer_transformers=prefer_transformers,
        )
    else:
        print("[step3] Fitting text->visual projection from Step1 GT (description, embedding) pairs...")
        projection = fit_text_to_visual_projection(
            processed_data,
            feature_dim=feature_dim,
            device=device,
            max_pairs=int(args.max_projection_pairs),
            ridge_lambda=float(args.ridge_lambda),
            prefer_transformers=prefer_transformers,
        )
        projection.save(projection_cache)
        print(f"[step3] Saved projection cache: {projection_cache}")

    print("[step3] Encoding task-graph nodes...")
    task_graph_embeddings: dict[str, dict] = {}
    for recipe_id, graph in tqdm(task_graphs.items(), desc="Encode graphs"):
        descs = [str(n.get("description", "")) for n in graph.nodes]
        with torch.no_grad():
            z = projection.encode(descs, device=device, normalize=True).detach().cpu().numpy().astype(np.float32)
        task_graph_embeddings[str(recipe_id)] = {"embeddings": z, "nodes": graph.nodes, "edges": graph.edges}

    realized_graphs: dict[str, dict] = {}
    missing_graphs: list[str] = []
    cfg = RealizeConfig(
        alpha_visual=float(args.alpha_visual),
        min_match_similarity=float(args.min_match_sim),
        unmatched_node_features=str(args.unmatched_node_features),
    )

    rec_items = list(processed_data.items())
    if int(args.limit_recordings) > 0:
        rec_items = rec_items[: int(args.limit_recordings)]

    print("[step3] Creating realized graphs...")
    for recording_id, rec in tqdm(rec_items, desc="Realize"):
        recipe_id = str(rec.get("recipe_id"))
        if recipe_id not in task_graph_embeddings:
            missing_graphs.append(str(recording_id))
            continue

        visual_emb = np.asarray(rec.get("step_embeddings", rec.get("embeddings")), dtype=np.float32)
        tg = task_graph_embeddings[recipe_id]
        realized = create_realized_graph(
            visual_embeddings=visual_emb,
            text_embeddings=np.asarray(tg["embeddings"], dtype=np.float32),
            edges=tg["edges"],
            cfg=cfg,
        )
        realized["recipe_label"] = int(rec.get("recipe_label", 0))
        realized["recipe_id"] = recipe_id
        realized_graphs[str(recording_id)] = realized

    print(f"[step3] Realized graphs: {len(realized_graphs)}")
    print(f"[step3] Missing task graphs: {len(missing_graphs)}")

    payload = {
        "realized_graphs": realized_graphs,
        "task_graph_embeddings": task_graph_embeddings,
        "splits": splits,
        "config": {
            "feature_dim": feature_dim,
            "text_encoder": "projection(distilbert->visual)" if prefer_transformers else "projection(hashing->visual)",
            "text_encoder_aligned": False,
            "matching": "hungarian",
            "alpha_visual": float(cfg.alpha_visual),
            "min_match_similarity": float(cfg.min_match_similarity),
            "unmatched_node_features": str(cfg.unmatched_node_features),
        },
    }

    save_pickle(out_pkl, payload)
    print(f"[step3] Saved: {out_pkl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
