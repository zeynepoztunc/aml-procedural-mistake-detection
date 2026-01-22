from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Step3Paths:
    repo_root: Path
    step1_pkl: Path
    step_annotations_json: Path
    task_graphs_json: Path
    out_pkl: Path
    projection_cache: Path


def default_paths(*, repo_root: Path) -> Step3Paths:
    return Step3Paths(
        repo_root=repo_root,
        step1_pkl=repo_root / "extension_data" / "step_embeddings_gt.pkl",
        step_annotations_json=repo_root
        / "annotations"
        / "annotation_json"
        / "step_annotations.json",
        task_graphs_json=repo_root / "annotations" / "annotation_json" / "task_graphs.json",
        out_pkl=repo_root / "extension_data" / "realized_task_graphs.pkl",
        projection_cache=repo_root / "extension_data" / "step3_text_to_visual_W.npy",
    )

