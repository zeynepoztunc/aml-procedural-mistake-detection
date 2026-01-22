from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RealizedTaskGraphs:
    realized_graphs: dict[str, dict[str, Any]]
    splits: dict[str, list[str]] | None
    config: dict[str, Any] | None
    feature_dim: int


def load_realized_task_graphs_pkl(path: Path) -> RealizedTaskGraphs:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("rb") as f:
        loaded = pickle.load(f)

    if not isinstance(loaded, dict):
        raise TypeError(f"Unexpected pickle type: {type(loaded).__name__}")
    if "realized_graphs" not in loaded:
        raise KeyError("Expected key 'realized_graphs'")

    graphs = loaded["realized_graphs"]
    if not isinstance(graphs, dict):
        raise TypeError(f"Expected realized_graphs to be dict, got {type(graphs).__name__}")

    feature_dim = None
    cfg = loaded.get("config")
    if isinstance(cfg, dict) and "feature_dim" in cfg:
        feature_dim = int(cfg["feature_dim"])
    else:
        sample_key = next(iter(graphs.keys()))
        sample = graphs[sample_key]
        nf = np.asarray(sample.get("node_features"))
        if nf.ndim != 2:
            raise ValueError("Could not infer feature_dim from realized_graphs[*]['node_features']")
        feature_dim = int(nf.shape[1])

    splits = loaded.get("splits") if isinstance(loaded.get("splits"), dict) else None
    config = cfg if isinstance(cfg, dict) else None

    return RealizedTaskGraphs(realized_graphs=graphs, splits=splits, config=config, feature_dim=feature_dim)

