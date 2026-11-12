<# Run the single-process stability diagnostic after the W6 GPU queue. #>
param([Parameter(Mandatory=$true)][int]$W6ProcessId)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$log = Join-Path $workspace 'benchmarks/results/w6-queue.out.log'
Write-Output 'Waiting for the W6 queue'
while (-not (Test-Path $log) -or -not (Select-String -Path $log -Pattern 'W6 per-card fixture queue completed.' -Quiet)) {
    if (-not (Get-Process -Id $W6ProcessId -ErrorAction SilentlyContinue)) {
        throw 'W6 queue exited without completing'
    }
    Start-Sleep -Seconds 30
}
Write-Output 'Running W7 adaptive stability diagnostic on RTX 3070'
& docker run --rm --gpus all --cpus=4 --memory=6g --shm-size=1g `
    --entrypoint env -e PYTHONPATH=/app/src -e NEUROEVO_BACKEND=gpu `
    -e 'NEUROEVO_GPU_NAME=NVIDIA GeForce RTX 3070' -e NEUROEVO_GPU_MIB=8192 `
    -v "${workspace}:/app" -v neuroevo-cache:/cache `
    neuroevo:m4-p0-comparison-v1 python -m neuroevo run `
    --config configs/s2-w7-adaptive-stability-v1.json `
    --output benchmarks/results/s2-w7-adaptive-v1
if ($LASTEXITCODE -ne 0) { throw 'W7 stability run failed' }
Write-Output 'Auditing W7 stability result'
& docker run --rm --entrypoint env -e PYTHONPATH=/app/src -e JAX_PLATFORMS=cpu `
    -v "${workspace}:/app" neuroevo:m4-p0-comparison-v1 `
    python scripts/w7_audit.py --run benchmarks/results/s2-w7-adaptive-v1/seed-900 `
    --output benchmarks/w7-stability-v1-audit.json
if ($LASTEXITCODE -ne 0) { throw 'W7 audit failed' }
Write-Output 'W7 stability queue completed.'
