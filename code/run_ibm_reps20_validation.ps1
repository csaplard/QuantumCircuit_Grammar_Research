# Run after collect_ibm_shots.py --reps 20 has produced *163840shots.txt files.
# Usage (from repo root): powershell -ExecutionPolicy Bypass -File code/run_ibm_reps20_validation.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$tag = "reps20"
$substr = "163840"

foreach ($backend in @("ibm_marrakesh", "ibm_torino")) {
  Write-Host "=== Grammar learning: $backend ===" -ForegroundColor Cyan
  python code/run_ibm_grammar_learning.py --backend-prefix $backend --filename-substr $substr --output-tag $tag

  Write-Host "=== GHZ vs Hadamard+layers threshold: $backend ===" -ForegroundColor Cyan
  python code/run_ibm_ghz_vs_hadamardlayers_threshold.py --backend-prefix $backend --filename-substr $substr --output-tag $tag --only-qubits "10,20" --max-points "4000,8000,16000,50000,100000" --epochs 5

  Write-Host "=== Sycamore-style protocol (LSTM + shuffled + LSB + LR/RF): $backend ===" -ForegroundColor Cyan
  python code/run_ibm_sycamore_protocol.py --backend-prefix $backend --filename-substr $substr --output-tag $tag --epochs 50 --max-pts 100000 --modes "original,shuffled,lsb,baseline"
}

Write-Host "Done." -ForegroundColor Green
