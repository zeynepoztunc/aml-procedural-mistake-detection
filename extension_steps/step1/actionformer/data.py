from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def load_json(path: Path) -> dict[str, Any]:
    # Pipeline input: reads annotations/splits JSON used by dataset creation and export.
    return json.loads(path.read_text(encoding="utf-8"))


def load_features(recording_id: str, feature_dir: Path) -> np.ndarray | None:
    # Pipeline input: loads per-recording EgoVLP feature sequence X ∈ R^(T×D) from .npz.
    path = feature_dir / f"{recording_id}_360p_224.npz"
    if not path.exists():
        return None
    data = np.load(str(path))
    # Some extracts store `video_features`; others store an unnamed array as the first key.
    if "video_features" in data:
        return data["video_features"]
    keys = list(data.keys())
    return data[keys[0]] if keys else None


@dataclass(frozen=True)
class Batch:
    recording_ids: list[str]
    features: torch.Tensor  # (B, T, C)
    boundary_labels: torch.Tensor  # (B, T)
    masks: torch.Tensor  # (B, T)


class StepLocalizationDataset(Dataset):
    """
    Training dataset for boundary detection.
    Label is 1 at step boundary timesteps (start/end), else 0.
    """

    def __init__(
        self,
        *,
        recording_ids: list[str],
        annotations: dict[str, Any],
        feature_dir: Path,
        max_len: int,
        feature_fps: float,
        boundary_tolerance: int = 3,
    ) -> None:
        self.annotations = annotations
        self.feature_dir = feature_dir
        self.max_len = max_len
        self.feature_fps = feature_fps
        self.boundary_tolerance = boundary_tolerance

        valid: list[str] = []
        for rec_id in recording_ids:
            if load_features(rec_id, feature_dir) is not None:
                valid.append(rec_id)
        self.valid_ids = valid

    def __len__(self) -> int:
        return len(self.valid_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec_id = self.valid_ids[idx]
        features = load_features(rec_id, self.feature_dir)
        if features is None:
            raise KeyError(f"Missing features for {rec_id}")

        seq_len = int(features.shape[0])
        boundary_labels = np.zeros(seq_len, dtype=np.float32)

        steps = self.annotations[rec_id]["steps"]
        for step in steps:
            start_frame = int(float(step["start_time"]) * self.feature_fps)
            end_frame = int(float(step["end_time"]) * self.feature_fps)
            for offset in range(-self.boundary_tolerance, self.boundary_tolerance + 1):
                if 0 <= start_frame + offset < seq_len:
                    boundary_labels[start_frame + offset] = 1.0
                if 0 <= end_frame + offset < seq_len:
                    boundary_labels[end_frame + offset] = 1.0

        if seq_len > self.max_len:
            features = features[: self.max_len]
            boundary_labels = boundary_labels[: self.max_len]
            seq_len = self.max_len

        return {
            "recording_id": rec_id,
            "features": torch.tensor(features, dtype=torch.float32),
            "boundary_labels": torch.tensor(boundary_labels, dtype=torch.float32),
            "seq_len": seq_len,
        }


def collate_fn(items: list[dict[str, Any]]) -> Batch:
    # Pipeline wiring: pads variable-length recordings into a batch tensor + mask for training/eval.
    max_len = max(int(it["seq_len"]) for it in items)

    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    rec_ids: list[str] = []

    for it in items:
        seq_len = int(it["seq_len"])
        feat = it["features"]
        lab = it["boundary_labels"]
        if seq_len < max_len:
            pad = max_len - seq_len
            feat = F.pad(feat, (0, 0, 0, pad))
            lab = F.pad(lab, (0, pad))
        features.append(feat)
        labels.append(lab)
        m = torch.zeros(max_len, dtype=torch.float32)
        m[:seq_len] = 1.0
        masks.append(m)
        rec_ids.append(it["recording_id"])

    return Batch(
        recording_ids=rec_ids,
        features=torch.stack(features),
        boundary_labels=torch.stack(labels),
        masks=torch.stack(masks),
    )
