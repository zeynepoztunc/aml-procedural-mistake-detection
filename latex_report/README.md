# LaTeX report

This folder contains the LaTeX sources for the project report.

## Build

From `aml-procedural-mistake-detection/`:

```powershell
cd latex_report
latexmk -pdf -interaction=nonstopmode -file-line-error report.tex
```

Notes:
- If `cvpr.sty` is present in this folder, the report will use the CVPR style.
- If `cvpr.sty` is missing, the document compiles with a lightweight fallback.

## Cleanup (optional)

If you previously compiled `report.tex` from the repo root, you may have legacy `report.*` build artifacts there.
You can delete them with:

```powershell
./latex_report/cleanup_root_report_files.ps1
```
