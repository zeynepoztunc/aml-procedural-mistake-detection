$ErrorActionPreference = "Stop"

Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "Removing legacy LaTeX files from repo root (report.*)..." -ForegroundColor Cyan
Write-Host "If this fails with 'Access is denied', close any PDF preview / LaTeX Workshop viewer using report.pdf and rerun." -ForegroundColor Yellow

$files = @(
  "report.tex",
  "report.pdf",
  "report.aux",
  "report.bbl",
  "report.blg",
  "report.fdb_latexmk",
  "report.fls",
  "report.log",
  "report.out",
  "report.synctex",
  "report.synctex.gz",
  "report.synctex(busy)",
  "report.toc",
  "report.lof",
  "report.lot"
)

foreach ($f in $files) {
  if (Test-Path $f) {
    Remove-Item -Force $f
    Write-Host "Deleted $f"
  }
}

Write-Host "Done." -ForegroundColor Green

