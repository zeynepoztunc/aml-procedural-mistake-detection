from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_features


@dataclass(frozen=True)
class StepEmbeddingRecord:
    step_embeddings: np.ndarray  # (num_steps, D)
    step_labels: list[int]
    recipe_id: int
    recipe_label: int
    descriptions: list[str]
    segments: list[tuple[float, float]]
    step_ids: list[int]


def pool_features_from_frames(features: np.ndarray, start_frame: int, end_frame: int) -> np.ndarray:
    # Pipeline stage (export): mean-pool EgoVLP features inside a predicted step segment to get one embedding.
    start_frame = max(0, start_frame)
    end_frame = min(int(features.shape[0]), end_frame)
    if start_frame >= end_frame:
        return np.zeros((features.shape[1],), dtype=np.float32)
    return np.mean(features[start_frame:end_frame], axis=0).astype(np.float32)


def _iou_1d(a: tuple[float, float], b: tuple[float, float]) -> float:
    # Pipeline helper: temporal IoU used only to align predicted segments to the closest GT step (optional).
    a0, a1 = a
    b0, b1 = b
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(1e-8, (a1 - a0) + (b1 - b0) - inter)
    return inter / union


def align_segments_to_gt(
    *,
    pred_segments: list[tuple[float, float]],
    gt_steps: list[dict[str, Any]],
) -> tuple[list[str], list[int], list[int]]:
    # Pipeline option: attach GT metadata (description/step_id/has_errors) to predicted segments by best IoU.
    """
    For each predicted segment, copy metadata from the best-overlapping GT step.
    Returns (descriptions, step_ids, step_labels).
    """
    gt_segments = [(float(s["start_time"]), float(s["end_time"])) for s in gt_steps]
    descs: list[str] = []
    ids: list[int] = []
    labels: list[int] = []

    for seg in pred_segments:
        best_iou = -1.0
        best_idx = -1
        for j, gt_seg in enumerate(gt_segments):
            iou = _iou_1d(seg, gt_seg)
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_idx >= 0:
            gt = gt_steps[best_idx]
            descs.append(str(gt.get("description", "")))
            ids.append(int(gt.get("step_id", -1)))
            labels.append(1 if gt.get("has_errors", False) else 0)
        else:
            descs.append("")
            ids.append(-1)
            labels.append(0)

    return descs, ids, labels


def build_step_embeddings_actionformer(
    *,
    recording_ids: list[str],
    annotations: dict[str, Any],
    feature_dir: Path,
    segments_by_rec_frames: dict[str, list[tuple[int, int]]],
    feature_fps: float,
    align_to_gt: bool,
) -> dict[str, dict[str, Any]]:
    # Pipeline stage (export): convert predicted segments (frames) into Step-2-compatible records + embeddings.
    out: dict[str, dict[str, Any]] = {}
    for rec_id in recording_ids:
        seg_frames = segments_by_rec_frames.get(rec_id)
        if not seg_frames:
            continue
        features = load_features(rec_id, feature_dir)
        if features is None:
            continue

        segments_sec = [(sf / feature_fps, ef / feature_fps) for sf, ef in seg_frames]
        step_embs = [pool_features_from_frames(features, sf, ef) for sf, ef in seg_frames]
        step_embeddings = np.stack(step_embs, axis=0) if step_embs else np.zeros((0, features.shape[1]), dtype=np.float32)

        recipe_id = 0
        try:
            recipe_id = int(str(rec_id).split("_")[0])
        except Exception:
            recipe_id = int(annotations.get(rec_id, {}).get("recipe_type", 0) or 0)

        ann = annotations.get(rec_id, {})
        recipe_label = 1 if any(bool(s.get("has_errors", False)) for s in ann.get("steps", [])) else 0

        descriptions: list[str] = ["" for _ in segments_sec]
        step_ids: list[int] = [-1 for _ in segments_sec]
        step_labels: list[int] = [0 for _ in segments_sec]
        if align_to_gt and "steps" in ann:
            descriptions, step_ids, step_labels = align_segments_to_gt(pred_segments=segments_sec, gt_steps=ann["steps"])

        out[rec_id] = {
            "step_embeddings": step_embeddings,
            "step_labels": step_labels,
            "recipe_id": recipe_id,
            "recipe_label": recipe_label,
            "embeddings": step_embeddings,  # legacy alias used by some cells
            "labels": step_labels,  # legacy alias
            "descriptions": descriptions,
            "segments": segments_sec,
            "step_ids": step_ids,
            "num_steps": int(step_embeddings.shape[0]),
        }

    return out


def save_step_embeddings_pkl(
    *,
    out_path: Path,
    method: str,
    feature_dim: int,
    data: dict[str, dict[str, Any]],
    splits: dict[str, Any],
) -> None:
    # Pipeline output: saves the final artifact consumed by Step 2 (`extension_data/step_embeddings_*.pkl`).
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"method": method, "feature_dim": feature_dim, "data": data, "splits": splits}
    with out_path.open("wb") as f:
        pickle.dump(payload, f)
