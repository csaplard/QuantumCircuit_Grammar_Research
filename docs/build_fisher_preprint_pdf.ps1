# Build Fisher_Threshold_Study_Preprint.pdf from LaTeX (same style as Grammar preprint).
# Requires: pdflatex (MiKTeX / TeX Live). First run may install missing packages (e.g. kvoptions).
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
$tex = "Fisher_Threshold_Study_Preprint.tex"
if (-not (Test-Path $tex)) { throw "Missing $tex" }

$pdflatex = $null
foreach ($c in @(
        (Get-Command pdflatex -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source),
        "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe",
        "${env:ProgramFiles}\MiKTeX\miktex\bin\x64\pdflatex.exe"
    )) {
    if ($c -and (Test-Path $c)) { $pdflatex = $c; break }
}
if (-not $pdflatex) { Write-Error "pdflatex not found. Install MiKTeX or TeX Live."; exit 1 }
Write-Host "Using:" $pdflatex

# Three runs: resolve LastPage, hyperref outlines, citations
1..3 | ForEach-Object { & $pdflatex -interaction=nonstopmode -halt-on-error $tex }
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Output: $here\Fisher_Threshold_Study_Preprint.pdf"
