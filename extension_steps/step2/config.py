from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def repo_root_from_file(path: Path) -> Path:
    """Find the repo root by walking upwards until `requirements.txt` is found."""
    for parent in [path.resolve(), *path.resolve().parents]:
        if (parent / "requirements.txt").exists():
            return parent
    raise RuntimeError(f"Could not find repo root from: {path}")


@dataclass(frozen=True)
class TrainConfig:
    max_steps: int = 50
    num_epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-4
    hidden_dim: int = 256
    dropout: float = 0.2
    pos_weight: float = 2.0
    seed: int = 1337


@dataclass(frozen=True)
class EvalConfig:
    min_train_size: int = 10
    min_test_size: int = 2
    threshold: float = 0.5

