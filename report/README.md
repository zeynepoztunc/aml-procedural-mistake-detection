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

Figure 1 is drawn in TikZ, so it compiles with no external files. If you want the plots the
notebooks save, download them into `figures/` and add `\includegraphics` blocks:

| File written by | Filename |
|---|---|
| `error_type_analysis.ipynb` | `error_category_f1_auc.png` |
| `extension_step1_actionformer.ipynb` | `actionformer_tiou_performance.png` |
| `extension_step1_actionformer.ipynb` | `actionformer_postprocessing_ablation_f1.png` |

The step-count histogram (substep 1) and the model bar chart (substep 4) are displayed inline
rather than saved; add a `plt.savefig(...)` to those cells if you want them in the PDF.

## Before submitting

- Fill the author block.
- **One `\TODO{}` remains**, in §5.7 "(iv) Remove the shortcut and everything collapses to
  chance" — the per-model omission-free numbers, pending the substep 4 run. `\TODO` renders in
  red so it cannot be missed; delete the macro definition once it is unused.
- Check the page count. The target is 8 pages excluding references.

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

## The handbook

`handbook.html` is the oral-exam study handbook, published as a private artifact at
https://claude.ai/code/artifact/450f977d-8e76-4e57-ba14-fb7d88c888c9 — republish that URL after
editing the file rather than creating a new one.
