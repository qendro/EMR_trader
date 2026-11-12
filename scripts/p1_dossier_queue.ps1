<# Assemble the P0 comparison and M4/M5 ledger after P1 validation. #>
param([Parameter(Mandatory=$true)][int]$P1ProcessId)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$log = Join-Path $workspace 'benchmarks/results/p1-regression-queue.out.log'
Write-Output 'Waiting for P1 regression and replay'
while (-not (Test-Path $log) -or -not (Select-String -Path $log -Pattern 'P1 cross-backend regression and replay queue completed.' -Quiet)) {
    if (-not (Get-Process -Id $P1ProcessId -ErrorAction SilentlyContinue)) {
        throw 'P1 regression queue exited without completing'
    }
    Start-Sleep -Seconds 30
}
Write-Output 'Auditing all four P0 methods'
& docker run --rm --entrypoint env -e PYTHONPATH=/app/src -e JAX_PLATFORMS=cpu `
    -v "${workspace}:/app" neuroevo:m4-p0-comparison-v1 `
    python scripts/p0_compare.py --root benchmarks/results `
    --baseline benchmarks/p0-random-action-baseline-v1.json `
    --json-output benchmarks/p0-comparison-v1.json `
    --markdown-output benchmarks/P0-COMPARISON-REPORT.md
if ($LASTEXITCODE -ne 0) { throw 'P0 comparison audit failed' }
Write-Output 'Assembling M4/M5 acceptance dossier'
& docker run --rm --entrypoint env -e PYTHONPATH=/app/src -e JAX_PLATFORMS=cpu `
    -v "${workspace}:/app" neuroevo:m4-p0-comparison-v1 `
    python scripts/p1_dossier.py --root benchmarks `
    --json-output benchmarks/p1-dossier-v1.json `
    --markdown-output benchmarks/P1-DOSSIER.md
if ($LASTEXITCODE -ne 0) { throw 'P1 dossier assembly failed' }
Write-Output 'P1 dossier queue completed.'
