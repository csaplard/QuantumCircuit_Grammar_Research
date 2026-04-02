# Resume IBM shot collection: skips circuits that already have a complete
# <backend>_<circuit>_<shots*reps>shots.txt file with the expected row count.
# Usage (repo root): powershell -ExecutionPolicy Bypass -File code/resume_ibm_shots.ps1
#
# Same args as collect_ibm_shots.py; defaults: --shots 8192 --reps 20

param(
  [string]$Backend = "ibm_marrakesh",
  [int]$Shots = 8192,
  [int]$Reps = 20
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "Resuming: backend=$Backend shots=$Shots reps=$Reps" -ForegroundColor Cyan
python code/collect_ibm_shots.py --backend $Backend --shots $Shots --reps $Reps --resume

Write-Host "Done." -ForegroundColor Green
