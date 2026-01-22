from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPTaskVerifier(nn.Module):
    """MLP baseline: masked-mean pool step embeddings then classify."""

    def __init__(self, feature_dim: int, *, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_exp = mask.unsqueeze(-1)
        pooled = (embeddings * mask_exp).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        x = F.relu(self.layer_norm(self.fc1(pooled)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class TransformerTaskVerifier(nn.Module):
    """Transformer encoder baseline with a learned CLS token."""

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_dim: int = 512,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        max_steps: int = 50,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_steps, hidden_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = embeddings.shape
        x = self.input_proj(embeddings)
        x = x + self.pos_encoding[:, :seq_len, :]

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        cls_mask = torch.ones(batch_size, 1, device=mask.device, dtype=mask.dtype)
        full_mask = torch.cat([cls_mask, mask], dim=1)
        key_padding_mask = full_mask == 0  # True = pad

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return self.classifier(x[:, 0, :])


class LSTMTaskVerifier(nn.Module):
    """BiLSTM baseline (packed sequences) with last hidden state classification."""

    def __init__(self, feature_dim: int, *, hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).long().clamp(min=1)
        sorted_lengths, sort_idx = lengths.sort(descending=True)
        sorted_emb = embeddings[sort_idx]

        packed = nn.utils.rnn.pack_padded_sequence(
            sorted_emb, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        _, (h_n, _) = self.lstm(packed)
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)

        _, unsort_idx = sort_idx.sort()
        hidden = hidden[unsort_idx]
        return self.classifier(hidden)

