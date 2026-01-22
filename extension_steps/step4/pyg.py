from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class PyG:
    Data: Any
    PyGDataLoader: Any
    GCNConv: Any
    GATConv: Any
    SAGEConv: Any
    global_mean_pool: Any
    global_max_pool: Any


def try_import_pyg() -> PyG | None:
    try:
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader as PyGDataLoader
        from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_max_pool, global_mean_pool

        return PyG(
            Data=Data,
            PyGDataLoader=PyGDataLoader,
            GCNConv=GCNConv,
            GATConv=GATConv,
            SAGEConv=SAGEConv,
            global_mean_pool=global_mean_pool,
            global_max_pool=global_max_pool,
        )
    except Exception:
        return None


def create_pyg_data(pyg: PyG, realized_graph: dict[str, Any], recording_id: str):
    x = torch.tensor(np.asarray(realized_graph["node_features"]), dtype=torch.float32)

    edge_index = np.asarray(realized_graph.get("edge_index"))
    if edge_index.size == 0:
        edge_index = edge_index.reshape(2, 0)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("Expected realized_graph['edge_index'] to have shape (2, E)")

    if edge_index.shape[1] > 0:
        rev = np.array([edge_index[1], edge_index[0]])
        edge_index = np.concatenate([edge_index, rev], axis=1)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long)

    y = torch.tensor([float(realized_graph["recipe_label"])], dtype=torch.float32)

    return pyg.Data(
        x=x,
        edge_index=edge_index_t,
        y=y,
        recording_id=recording_id,
        recipe_id=int(realized_graph["recipe_id"]),
    )

