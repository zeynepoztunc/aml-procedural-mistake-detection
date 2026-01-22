"""Step 1: ActionFormer-based step localization.

This is a refactor of the exported notebook script into small modules:
- `model.py`: ActionFormer architecture
- `data.py`: feature/annotation loading + dataset/collate
- `train.py`: training + evaluation loops
- `infer.py`: boundary -> segment inference from logits
- `export.py`: pool segments into step embeddings + save artifacts
- `main.py`: CLI entrypoint
"""

