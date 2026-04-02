# Full Sycamore Fisher sweep for multiple grammar-learner seeds (sequential).
# Each run writes results\fisher_*_all_readouts_seed<N>.* and logs to results\fisher_full_run_seed<N>.log
#
# Usage (from repo root):
#   powershell -NoProfile -ExecutionPolicy Bypass -File code/run_fisher_seed_sweep.ps1
# Optional:
#   powershell -File code/run_fisher_seed_sweep.ps1 -Seeds 0,1,2

param(
    [int[]]$Seeds = @(0, 1, 2)
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Results = Join-Path $RepoRoot "results"
if (-not (Test-Path $Results)) { New-Item -ItemType Directory -Path $Results | Out-Null }

Set-Location $RepoRoot
Write-Host "Repo: $RepoRoot"
Write-Host "Seeds: $($Seeds -join ', ')"

foreach ($s in $Seeds) {
    $logOut = Join-Path $Results "fisher_full_run_seed${s}.log"
    $logErr = Join-Path $Results "fisher_full_run_seed${s}.err.log"
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$stamp] Starting Fisher full run seed=$s"
    Write-Host "  stdout: $logOut"
    Write-Host "  stderr: $logErr"

    $p = Start-Process -FilePath "python" `
        -ArgumentList @(
            (Join-Path $RepoRoot "code\fisher_information_analysis.py"),
            "--tag-with-seed",
            "--seed",
            "$s"
        ) `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $logOut `
        -RedirectStandardError $logErr `
        -WindowStyle Hidden `
        -PassThru `
        -Wait

    if ($p.ExitCode -ne 0) {
        Write-Error "Fisher seed $s exited with code $($p.ExitCode). See $logErr"
        exit $p.ExitCode
    }
    Write-Host "[$((Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))] Finished seed=$s"
}

Write-Host "All Fisher seed runs completed."
