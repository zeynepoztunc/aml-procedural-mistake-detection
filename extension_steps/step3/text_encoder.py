from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _hf_offline_enabled() -> bool:
    return bool(os.environ.get("HF_HUB_OFFLINE")) or bool(os.environ.get("TRANSFORMERS_OFFLINE"))


def _text_features_hashing(texts: list[str], *, dim: int = 768) -> np.ndarray:
    from sklearn.feature_extraction.text import HashingVectorizer

    vec = HashingVectorizer(n_features=dim, alternate_sign=False, norm=None)
    x = vec.transform(texts).toarray().astype(np.float32)
    return x


def _text_features_transformers(
    texts: list[str],
    *,
    device: torch.device,
    model_name: str = "distilbert-base-uncased",
    batch_size: int = 32,
) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=_hf_offline_enabled())
    model = AutoModel.from_pretrained(model_name, local_files_only=_hf_offline_enabled()).to(device)
    model.eval()

    all_emb: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
            last = out.last_hidden_state  # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
            pooled = (last * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1.0))
        all_emb.append(pooled.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(all_emb, axis=0) if all_emb else np.zeros((0, 768), dtype=np.float32)


def get_text_features(
    texts: list[str],
    *,
    device: torch.device,
    prefer_transformers: bool = True,
) -> np.ndarray:
    if prefer_transformers:
        try:
            return _text_features_transformers(texts, device=device)
        except Exception:
            pass
    return _text_features_hashing(texts)


@dataclass
class TextToVisualProjection:
    """Encode text into the Step1 visual embedding space via a learned linear projection."""

    feature_dim: int
    W: torch.Tensor  # (H, D) in float32
    text_feature_dim: int = 768
    prefer_transformers: bool = True

    def encode(self, texts: list[str] | str, *, device: torch.device, normalize: bool = True) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        feats = get_text_features([str(t) for t in texts], device=device, prefer_transformers=self.prefer_transformers)
        x = torch.from_numpy(feats).to(device)
        z = x @ self.W.to(device)
        return F.normalize(z, dim=1) if normalize else z

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), self.W.detach().cpu().numpy())

    @staticmethod
    def load(path: Path, *, feature_dim: int, prefer_transformers: bool = True) -> "TextToVisualProjection":
        W = torch.from_numpy(np.load(str(path))).float()
        if W.ndim != 2 or W.shape[1] != feature_dim:
            raise ValueError(f"Bad projection matrix shape: {tuple(W.shape)}; expected (*, {feature_dim})")
        return TextToVisualProjection(feature_dim=feature_dim, W=W, prefer_transformers=prefer_transformers)


def fit_text_to_visual_projection(
    processed_data: dict,
    *,
    feature_dim: int,
    device: torch.device,
    max_pairs: int = 6000,
    ridge_lambda: float = 10.0,
    prefer_transformers: bool = True,
) -> TextToVisualProjection:
    texts: list[str] = []
    visuals: list[np.ndarray] = []

    for rec in processed_data.values():
        descs = rec.get("descriptions") or []
        vis = rec.get("step_embeddings", rec.get("embeddings"))
        if vis is None:
            continue
        vis = np.asarray(vis)
        if vis.ndim != 2 or vis.shape[0] != len(descs):
            continue
        for d, v in zip(descs, vis):
            if not d:
                continue
            texts.append(str(d))
            visuals.append(np.asarray(v, dtype=np.float32))

    if not texts:
        raise RuntimeError("No (description, embedding) pairs found to fit text->visual projection.")

    if len(texts) > max_pairs:
        idx = np.linspace(0, len(texts) - 1, max_pairs).astype(int).tolist()
        texts = [texts[i] for i in idx]
        visuals = [visuals[i] for i in idx]

    X = get_text_features(texts, device=device, prefer_transformers=prefer_transformers)  # (N, H)
    Y = np.stack(visuals, axis=0).astype(np.float32)  # (N, D)

    if Y.shape[1] != feature_dim:
        raise ValueError(f"Feature dim mismatch: Y has {Y.shape[1]} but expected {feature_dim}")

    # Ridge-regularized least squares: W = (X^T X + λI)^-1 X^T Y
    XtX = X.T @ X
    XtX += ridge_lambda * np.eye(XtX.shape[0], dtype=np.float32)
    XtY = X.T @ Y
    W = np.linalg.solve(XtX, XtY).astype(np.float32)  # (H, D)

    return TextToVisualProjection(
        feature_dim=feature_dim,
        W=torch.from_numpy(W),
        text_feature_dim=X.shape[1],
        prefer_transformers=prefer_transformers,
    )

