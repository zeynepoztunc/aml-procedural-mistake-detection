from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StepEmbeddings:
    """In-memory view of a Step-1 output file used by Step 2."""

    method: str
    feature_dim: int
    splits: dict[str, list[str]]
    data: dict[str, dict[str, Any]]


def load_step_embeddings_pkl(path: Path) -> StepEmbeddings:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("rb") as f:
        loaded = pickle.load(f)

    if not isinstance(loaded, dict):
        raise TypeError(f"Unexpected pickle type: {type(loaded).__name__}")
    if "data" not in loaded or "splits" not in loaded:
        raise KeyError("Expected keys: 'data' and 'splits'")

    data = loaded["data"]
    if not isinstance(data, dict):
        raise TypeError(f"Expected 'data' to be dict, got {type(data).__name__}")

    # Normalize a couple of legacy keys to keep Step 2 robust.
    for rec_id, rec in data.items():
        if not isinstance(rec, dict):
            continue
        if "step_embeddings" not in rec and "embeddings" in rec:
            rec["step_embeddings"] = rec["embeddings"]
        if "recipe_label" not in rec and "label" in rec:
            rec["recipe_label"] = rec["label"]

        if "step_embeddings" in rec and isinstance(rec["step_embeddings"], np.ndarray):
            if rec["step_embeddings"].ndim != 2:
                raise ValueError(f"{rec_id}: expected step_embeddings to be 2D array")

    sample_key = next(iter(data.keys()))
    sample = data[sample_key]
    emb = sample.get("step_embeddings")
    if not isinstance(emb, np.ndarray) or emb.ndim != 2:
        raise ValueError("Could not infer feature_dim from data['step_embeddings']")

    method = str(loaded.get("method", "unknown"))
    feature_dim = int(emb.shape[1])

    splits = loaded["splits"]
    if not isinstance(splits, dict):
        raise TypeError(f"Expected 'splits' to be dict, got {type(splits).__name__}")

    return StepEmbeddings(method=method, feature_dim=feature_dim, splits=splits, data=data)

