from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class TaskVerificationDataset(Dataset[dict[str, Any]]):
    """Recipe-level verification dataset (variable-length sequences -> padded tensors)."""

    def __init__(self, recording_ids: list[str], processed_data: dict[str, dict[str, Any]], *, max_steps: int = 50):
        self.recording_ids = recording_ids
        self.processed_data = processed_data
        self.max_steps = max_steps

    def __len__(self) -> int:
        return len(self.recording_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        recording_id = self.recording_ids[idx]
        data = self.processed_data[recording_id]

        step_emb = data["step_embeddings"]  # (num_steps, feature_dim)
        label = float(data["recipe_label"])

        if not isinstance(step_emb, np.ndarray) or step_emb.ndim != 2:
            raise TypeError(f"{recording_id}: expected step_embeddings to be a 2D numpy array")

        num_steps, feature_dim = step_emb.shape
        out_steps = min(num_steps, self.max_steps)

        padded = np.zeros((self.max_steps, feature_dim), dtype=np.float32)
        padded[:out_steps] = step_emb[:out_steps].astype(np.float32, copy=False)

        mask = np.zeros((self.max_steps,), dtype=np.float32)
        mask[:out_steps] = 1.0

        return {
            "embeddings": torch.from_numpy(padded),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor([label], dtype=torch.float32),
            "num_steps": num_steps,
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "embeddings": torch.stack([x["embeddings"] for x in batch]),
        "mask": torch.stack([x["mask"] for x in batch]),
        "label": torch.stack([x["label"] for x in batch]),
        "num_steps": [x["num_steps"] for x in batch],
    }

