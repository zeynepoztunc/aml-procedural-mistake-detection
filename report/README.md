# Report

`main.tex` is written against the **CVPR 2026 submission template**
(https://www.overleaf.com/latex/templates/cvpr-2026-submission-template/rdtrwgypxxzb).

## Building it

1. Open the Overleaf template and make a copy.
2. Replace the template's `main.tex` with this one, and add `main.bib`.
3. Keep the template's own files: `cvpr.sty`, `cvpr_eso.sty`, `ieeenat_fullname.bst`.
4. Compile with pdfLaTeX → BibTeX → pdfLaTeX ×2.

The preamble expects `\paperID`, `\confName` and `\confYear`, which the template's `cvpr.sty`
uses for the review header. Switch `\usepackage[review]{cvpr}` to `\usepackage{cvpr}` for the
camera-ready version.

## Figures

All figures now come from the notebooks. `main.tex` expects **three image files** in
`figures/`. Names matter; extensions do not (`\graphicspath` and
`\DeclareGraphicsExtensions` resolve `.pdf`, `.png` or `.jpg`, in `figures/` or beside
`main.tex`).

### Three of them are already saved as files — no screenshot needed

| Save as | Notebook | Cell (search this first line) | Where the notebook writes it |
|---|---|---|---|
| `error_categories.png` | `error_type_analysis.ipynb` | `# Per-category F1 and AUC, both splits.` | `error_category_f1_auc.png` in the Colab working dir |
| `gnn_comparison.png` | `extension_step4_gnn_classification.ipynb` | `# Visualization - one chart per boundary source` | `extension_data/gnn_comparison.png` |

Download those two, rename to the names in column 1, upload. They are 300-dpi `savefig`
output, so they are already cleaner than any screenshot of the same cell.

### One needs capturing

| Save as | Notebook | Cell (search this first line) |
|---|---|---|
| `localization_examples.png` | `extension_step1_actionformer.ipynb` | `import matplotlib.pyplot as plt` |

That cell draws **three separate figures**, one per recording (`20_48`, `25_42`, `29_22`).
Capture all three as one image, stacked in order, and crop out the `Recording: ... GT steps:
... Predicted steps: ...` text printed between them — the caption already states those numbers.

Better than a screenshot, if you want it: add `plt.savefig(f"timeline_{rec_id}.png", dpi=300,
bbox_inches="tight")` immediately before `plt.show()` inside `compare_boundaries`, re-run that
one cell, and you get three clean files to stack.

### Optional fourth

`extension_step4_gnn_classification.ipynb`, the depth-study plot cell, already writes
`extension_data/gnn_depth_study.png`. It supports the over-smoothing paragraph in the
Ablations. Add it only if the page count allows; the paragraph stands without it.

### What went back to being tables

Two results have no plotted equivalent in any notebook, so they are tables again:
`tab:loc` (per-split localization) and `tab:omission` (the omission-free subset). The
omission-free result is the paper's strongest evidence and there is no notebook cell that
charts it.

The previously generated vector figures are parked in `figures/unused/` with the scripts
`make_figures.py` and `make_timeline.py`. Nothing references them.

## Before submitting

- Fill the author block.
- **No `\TODO{}` remains.** The macro is still defined in the preamble if you want to flag
  something while editing; delete the definition before the camera-ready.
- Check the page count. The target is 8 pages excluding references. The omission-free table
  (`tab:omission`) was added last and is the most likely thing to push it over.
- The substep-4 numbers come from the run with execution count 13 in
  `extension_step4_gnn_classification.ipynb`. **Cells 19, 20, 21 and 32 of that notebook still
  display the previous run** and need re-executing before the notebook is handed in — the report
  is already on the new numbers.

## Where the numbers come from

Every figure in the report is read from a stored cell output in the notebooks, not from
`REPORT_NOTES.md` prose. `REPORT_NOTES.md` (gitignored) records which numbers were superseded
and why — consult it before changing anything here, since several plausible-looking figures in
the project's history were retracted.

Two classes of number are easy to quote wrongly, and the report deliberately uses the held-out
variant of both:

- **Step-count MAE**: 2.45 pooled over all 384 recordings, **3.28** on the 109 held-out test
  recordings. The pooled figure averages in 213 recordings the detector trained on.
- **Step-count correlation**: 0.786 pooled, **0.620** on test.

A third: **substep-4 model rankings under annotated boundaries are not stable across runs.**
DAGNN scored 79.0 AUC in one run and 81.3 in the next, with the three models spanning only
0.7 points. The report deliberately claims no ranking there, and says why. Do not "fix" this
by quoting whichever run makes a model look best.

## The handbook

`handbook.html` is the oral-exam study handbook, published as a private artifact at
https://claude.ai/code/artifact/450f977d-8e76-4e57-ba14-fb7d88c888c9 — republish that URL after
editing the file rather than creating a new one.
