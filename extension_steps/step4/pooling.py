from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def pooled_graph_features(realized_graph: dict[str, Any]) -> np.ndarray:
    """Mean+max pool node features -> (2D,) feature vector."""
    node_features = np.asarray(realized_graph.get("node_features"))
    if node_features.ndim != 2:
        raise ValueError("Expected realized_graph['node_features'] to be a 2D array-like")

    mean = node_features.mean(axis=0)
    max_ = node_features.max(axis=0)
    pooled = np.concatenate([mean, max_], axis=0).astype(np.float32, copy=False)
    return pooled


class PooledGraphDataset(Dataset[dict[str, Any]]):
    """Graph dataset that pre-pools node features (no PyG required)."""

    def __init__(self, recording_ids: list[str], realized_graphs: dict[str, dict[str, Any]]):
        self.recording_ids = recording_ids
        self.realized_graphs = realized_graphs

    def __len__(self) -> int:
        return len(self.recording_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        recording_id = self.recording_ids[idx]
        g = self.realized_graphs[recording_id]
        x = pooled_graph_features(g)
        y = float(g["recipe_label"])
        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor([y], dtype=torch.float32),
            "recording_id": recording_id,
            "recipe_id": int(g["recipe_id"]),
        }


def collate_pooled(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "recording_id": [b["recording_id"] for b in batch],
        "recipe_id": [b["recipe_id"] for b in batch],
    }

