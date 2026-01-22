from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pyg import PyG


class PooledMLPClassifier(nn.Module):
    """Baseline: classify pooled node features (mean+max)."""

    def __init__(self, input_dim: int, *, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class GnnModels:
    GCNClassifier: Any
    GATClassifier: Any
    GraphSAGEClassifier: Any


def build_gnn_models(pyg: PyG) -> GnnModels:
    GCNConv = pyg.GCNConv
    GATConv = pyg.GATConv
    SAGEConv = pyg.SAGEConv
    global_mean_pool = pyg.global_mean_pool
    global_max_pool = pyg.global_max_pool

    class GCNClassifier(nn.Module):
        def __init__(self, input_dim: int, *, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.3):
            super().__init__()
            self.convs = nn.ModuleList([GCNConv(input_dim, hidden_dim)])
            self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim)])
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
                x = self.dropout(x)
            x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
            return self.classifier(x)

    class GATClassifier(nn.Module):
        def __init__(
            self,
            input_dim: int,
            *,
            hidden_dim: int = 128,
            num_layers: int = 3,
            heads: int = 4,
            dropout: float = 0.3,
        ):
            super().__init__()
            self.convs = nn.ModuleList([GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=dropout)])
            self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim)])
            for _ in range(num_layers - 1):
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.elu(x)
                x = self.dropout(x)
            x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
            return self.classifier(x)

    class GraphSAGEClassifier(nn.Module):
        def __init__(self, input_dim: int, *, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.3):
            super().__init__()
            self.convs = nn.ModuleList([SAGEConv(input_dim, hidden_dim)])
            self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim)])
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
                x = self.dropout(x)
            x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
            return self.classifier(x)

    return GnnModels(GCNClassifier=GCNClassifier, GATClassifier=GATClassifier, GraphSAGEClassifier=GraphSAGEClassifier)

