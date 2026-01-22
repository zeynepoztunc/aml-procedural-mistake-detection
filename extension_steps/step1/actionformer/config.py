from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def repo_root_from_file(file_path: Path) -> Path:
    # Pipeline wiring: resolve the repo root so CLI runs from a stable CWD and
    # relative data paths (data/, annotations/, er_annotations/) work.
    # file_path: .../extension_steps/step1/actionformer/config.py
    return file_path.resolve().parents[3]


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    egovlp_feature_dir: Path
    annotation_path: Path
    split_path: Path
    out_dir: Path


@dataclass(frozen=True)
class TrainConfig:
    feature_fps: float = 0.5
    max_len: int = 2000
    batch_size: int = 16
    lr: float = 5e-4
    weight_decay: float = 0.05
    pos_weight: float = 2.0
    num_epochs: int = 30
    warmup_epochs: int = 5
    seed: int = 1337


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = 256
    embd_dim: int = 256
    n_head: int = 4
    arch: tuple[int, int, int] = (2, 2, 5)
    mha_win_size: int = 19
    scale_factor: int = 2
    fpn_dim: int = 256
    head_dim: int = 256
    attn_pdrop: float = 0.1
    proj_pdrop: float = 0.1
    path_pdrop: float = 0.1
