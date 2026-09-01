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

Every figure is a notebook output. `main.tex` expects **three image files** in `figures/`.
Names matter; extensions do not (`\graphicspath` and `\DeclareGraphicsExtensions` resolve
`.pdf`, `.png` or `.jpg`, in `figures/` or beside `main.tex`).

| File | Notebook | Cell (search this first line) |
|---|---|---|
| `error_type.png` | `error_type_analysis.ipynb` | `# Per-category F1 and AUC, both splits.` |
| `step1_1.png`, `step1_3.png` | `extension_step1_actionformer.ipynb` | `import matplotlib.pyplot as plt` |

`figures/orig/` holds the uncropped originals of all five captured images. Two of the five
are not referenced by `main.tex`: `step1_2.png` (the median recording, `25_42`) and
`step4.png` (the substep-4 comparison chart), both dropped to meet the page limit.

### The images in `figures/` are cropped

The versions `main.tex` includes are **not** the originals. To save vertical space, the
matplotlib titles were cropped off the top: 52 px from each timeline and 30 px from
`error_type.png`, which also removed a slogan-style suptitle that did not belong above a
caption. `pngcrop.py` (beside this README) did the cropping — a dependency-free PNG row cropper,
since neither PIL nor ImageMagick is available here. Re-crop from `figures/orig/` if a
figure needs regenerating; do not crop an already-cropped file twice.

The captions carry the information the cropped titles used to show, so if you restore an
original, shorten the caption to match.

### If a figure needs to go back in

`step4.png` is the substep-4 comparison chart from the `# Visualization - one chart per
boundary source` cell, which writes `extension_data/gnn_comparison.png`. It was cut because
`tab:sub4` carries the same numbers. `extension_data/gnn_depth_study.png` exists too and
supports the over-smoothing paragraph. Add either only if the page count allows.

### What has no plotted equivalent

`tab:omission`, the omission-free subset, is the paper's strongest single piece of evidence
and no notebook cell charts it, so it is a table by necessity rather than by choice.

The previously generated vector figures are parked in `figures/unused/` with the scripts
`make_figures.py` and `make_timeline.py`. Nothing references them.

## Before submitting

- **No `\TODO{}` remains.** The macro is still defined in the preamble if you want to flag
  something while editing; delete the definition before the camera-ready.
- Check the page count. The spec says 8 pages; with the CVPR template that normally means 8
  excluding references. If it is still over, cut in this order: the substep-4 comparison
  figure (already removed, `figures/step4.png` is kept in case it goes back), then the
  qualitative timeline figure (`fig:qualitative` — the prose carries the same point), then
  `tab:er`'s SlowFast rows.
- The substep-4 numbers come from the third run of the whole notebook (execution counts 13
  through 29 of `extension_step4_gnn_classification.ipynb`). Every cell in that notebook is
  from the same session; the check is that cell 34's `auc, full set` column matches cell 17.
  Do not mix runs.

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
Across three runs of the identical configuration and seed, SimplePooling spanned 0.3 AUC
(80.8, 80.9, 81.1), GraphSAGE 1.0 (79.6, 80.6, 79.7) and DAGNN 3.3 (79.0, 81.3, 82.3) — and
the best of the three changed between runs. The report deliberately claims no ranking there,
and says why. Do not "fix" this by quoting whichever run makes a model look best.

Note also that cell 34 of the substep-4 notebook prints the hardcoded line "Every model
collapses to chance." On the final run GraphSAGE keeps 59.3 AUC on that subset, so the
report says two of three land at chance instead. The notebook's sentence predates the run;
the table below it is correct.

## The handbook

`handbook.html` is the oral-exam study handbook, published as a private artifact at
https://claude.ai/code/artifact/450f977d-8e76-4e57-ba14-fb7d88c888c9 — republish that URL after
editing the file rather than creating a new one.
