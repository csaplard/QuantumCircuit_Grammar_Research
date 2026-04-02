# After the current fisher_information_analysis --seed 0 finishes, run seeds 1 and 2
# sequentially (same layout as run_fisher_seed_sweep.ps1). Does NOT start a second seed 0.
#
# Usage (from repo root):
#   powershell -NoProfile -ExecutionPolicy Bypass -File code/run_fisher_seeds_1_2_after_seed0.ps1
#
# Optional: set PYTHON_FISHER to full python.exe path, or pass -PythonExe.

param(
    [string]$PythonExe = $env:PYTHON_FISHER
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Results = Join-Path $RepoRoot "results"
if (-not (Test-Path $Results)) { New-Item -ItemType Directory -Path $Results | Out-Null }

if (-not $PythonExe) {
    $PythonExe = "C:\Users\csapl\AppData\Local\Python\pythoncore-3.14-64\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Error "Python not found: $PythonExe - set PYTHON_FISHER or install Python 3.14."
    exit 1
}

Set-Location $RepoRoot
$queueLog = Join-Path $Results "fisher_queue_seeds_1_2.log"

function Write-QueueLog([string]$msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Add-Content -Path $queueLog -Value $line -Encoding utf8
    Write-Host $line
}

Write-QueueLog "Waiting for fisher_information_analysis --seed 0 to finish..."
while ($true) {
    $running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'fisher_information_analysis' -and $_.CommandLine -match '--seed\s+0(\s|$)' }
    if (-not $running) { break }
    Start-Sleep -Seconds 30
}
Write-QueueLog "Seed 0 finished (or was not running); starting seeds 1 and 2."

foreach ($s in @(1, 2)) {
    $logOut = Join-Path $Results "fisher_full_run_seed${s}.log"
    $logErr = Join-Path $Results "fisher_full_run_seed${s}.err.log"
    Write-QueueLog "Starting Fisher seed=$s"
    Write-QueueLog "  stdout: $logOut"
    Write-QueueLog "  stderr: $logErr"

    $p = Start-Process -FilePath $PythonExe `
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
        Write-QueueLog "ERROR: Fisher seed $s exited with code $($p.ExitCode). See $logErr"
        exit $p.ExitCode
    }
    Write-QueueLog "Finished seed=$s"
}

Write-QueueLog "All queued Fisher seeds 1 and 2 completed."
