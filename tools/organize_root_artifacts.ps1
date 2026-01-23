$ErrorActionPreference = "Continue"

Set-Location $PSScriptRoot
Set-Location ..

Write-Host "Organizing root-level artifacts into folders..." -ForegroundColor Cyan
Write-Host "If you see 'Access is denied', close any open VSCode tabs/PDF previews and rerun." -ForegroundColor Yellow

$moves = @(
  @{ src = "colab_feature_extraction.ipynb"; dst = "notebooks" },
  @{ src = "colab_quickstart.ipynb"; dst = "notebooks" },
  @{ src = "error_type_analysis.ipynb"; dst = "notebooks" },
  @{ src = "extension_step1_actionformer.ipynb"; dst = "notebooks" },
  @{ src = "extension_step1_pipeline.ipynb"; dst = "notebooks" },
  @{ src = "extension_step2_verification_baseline.ipynb"; dst = "notebooks" },
  @{ src = "extension_step3_task_graph_matching.ipynb"; dst = "notebooks" },
  @{ src = "extension_step4_gnn_classification.ipynb"; dst = "notebooks" },
  @{ src = "old-version-train_egovlp_baseline.ipynb"; dst = "notebooks" },
  @{ src = "train_lstm_baseline.ipynb"; dst = "notebooks" },
  @{ src = "train_lstm_baseline_v2.ipynb"; dst = "notebooks" },

  @{ src = "error_type_analysis.md"; dst = "docs\\analysis" },
  @{ src = "LSTM_results_analysis.md"; dst = "docs\\analysis" },
  @{ src = "LOCAL_BRANCH_COMPARISON.md"; dst = "docs\\analysis" },

  @{ src = "AML-2025_Mistake_Detection_Project.pdf"; dst = "docs\\spec" },
  @{ src = "gnn_oversmoothing.png"; dst = "assets\\figures" }
)

foreach ($m in $moves) {
  $src = Join-Path (Get-Location) $m.src
  $dstDir = Join-Path (Get-Location) $m.dst
  $dst = Join-Path $dstDir $m.src

  if (Test-Path $src) {
    if (!(Test-Path $dstDir)) { New-Item -ItemType Directory -Path $dstDir | Out-Null }
    try {
      Move-Item -Force $src $dst
      Write-Host "Moved $($m.src) -> $($m.dst)" -ForegroundColor Green
    } catch {
      Write-Host "Could not move $($m.src): $($_.Exception.Message)" -ForegroundColor Red
    }
  }
}

Write-Host "Done." -ForegroundColor Cyan

