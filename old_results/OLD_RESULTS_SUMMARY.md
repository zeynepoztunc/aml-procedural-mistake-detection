# Archived (Old) Results Snapshot

This file preserves key numbers/claims from an earlier draft of `latex_report/report.tex` before we updated it to match the **current refactored implementation** and the **latest measured results** saved under `extension_data/`.


These are kept for reference only (do **not** treat as the final project results).

## Old headline claims (from the previous draft)

- **ActionFormer (Step 1)**: reported as ~**F1 = 9.19%** (grid search) and sometimes **F1 = 11.97%**.
- **Step 3 matching**: reported cosine similarity ranges roughly **0.08–0.25** for Hungarian matching.
- **Task verification (Step 2)**: reported that models converge to **~72% F1** after systematic grid search.
- **Graph classification (Step 4)**: reported GraphSAGE achieving **75.7% F1**.

## Old Step 4 table (as previously written)

| Model | Acc | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| SimplePooling | 64.3% | 68.2% | 78.0% | 69.9% |
| GCN | 67.4% | 73.0% | 73.5% | 70.5% |
| GAT | 72.9% | 80.4% | 71.7% | 73.8% |
| GraphSAGE | 72.5% | 76.7% | 78.4% | 75.7% |

## Why this is “old”

After refactoring the extension pipeline (Steps 1–4) into runnable modules and saving measured outputs to `extension_data/`,
we have newer results that differ from these numbers (notably Step 3 similarity statistics and Step 4 LORO metrics).
