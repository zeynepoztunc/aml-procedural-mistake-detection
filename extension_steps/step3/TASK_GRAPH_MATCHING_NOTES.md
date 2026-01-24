# Step 3 Notes (Task Graph Matching)

- **Purpose:** Match each detected step from Step 1 with its corresponding node in the ground-truth task graph so that downstream GNNs operate on a “realized” recipe graph.
- **Input:** Step 1 outputs (`step_embeddings_gt.pkl` or ActionFormer embeddings), step annotations (`annotations/annotation_json/step_annotations.json`), optional prebuilt task graphs (`annotations/annotation_json/task_graphs.json`).
- **Text encoder:** Uses EgoVLP-aligned text embeddings when available; otherwise computes DistilBERT vectors and learns a ridge projection to the Step 1 embedding space. Cache file: `extension_data/step3_text_to_visual_W.npy`.
- **Matching:** Hungarian algorithm with optional `--min-match-sim` threshold. When a match is below the threshold, the node is marked “unmatched” and can keep either its text embedding or zeros.
- **Output:** `extension_data/realized_task_graphs.pkl` containing realized graphs + split metadata. Check `summarize_realized_graphs.py` for quick inspection.
- **Tips:**
  - Always fix `--step1-pkl`/`--out-pkl` paths relative to the repo root.
  - Use `--alpha-visual` to trade off text vs. visual similarity if the ridge projection seems biased.
  - When experimenting, copy one of the paste-ready commands from `optimal_extension_configs/README.md`.
