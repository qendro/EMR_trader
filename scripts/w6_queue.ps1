<# Run W6 on each card after P0 comparison work releases the GPUs. #>
param([Parameter(Mandatory=$true)][int]$ComparisonProcessId)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$log = Join-Path $workspace 'benchmarks/results/p0-comparison-queue.out.log'
Write-Output 'Waiting for the P0 comparison queue'
while (-not (Test-Path $log) -or -not (Select-String -Path $log -Pattern 'P0 comparison queue completed.' -Quiet)) {
    if (-not (Get-Process -Id $ComparisonProcessId -ErrorAction SilentlyContinue)) {
        throw 'P0 comparison queue exited without completing'
    }
    Start-Sleep -Seconds 30
}
$inventory = & nvidia-smi --query-gpu=uuid,name --format=csv,noheader
foreach ($card in @(@{name='NVIDIA GeForce RTX 3070'; label='rtx3070'},
                    @{name='NVIDIA GeForce RTX 3060'; label='rtx3060'})) {
    $row = $inventory | Where-Object { $_.EndsWith($card.name) }
    if (@($row).Count -ne 1) { throw "GPU inventory mismatch: $($card.name)" }
    $uuid = ($row -split ',')[0].Trim()
    $output = "benchmarks/w6-$($card.label)-v1.json"
    Write-Output "Running W6 on $($card.name)"
    & docker run --rm --gpus all --cpus=4 --memory=6g --shm-size=1g `
        --entrypoint env -e "CUDA_VISIBLE_DEVICES=$uuid" -e JAX_PLATFORMS=cuda `
        -e XLA_PYTHON_CLIENT_PREALLOCATE=false `
        -v "${workspace}:/app" neuroevo:m4-p0-comparison-v1 `
        python scripts/w6_fixture.py --output $output
    if ($LASTEXITCODE -ne 0) { throw "W6 failed on $($card.name)" }
}
& docker run --rm --entrypoint env -e PYTHONPATH=/app/src -e JAX_PLATFORMS=cpu `
    -v "${workspace}:/app" neuroevo:m4-p0-comparison-v1 `
    python scripts/w6_audit.py --results benchmarks/w6-rtx3070-v1.json `
    benchmarks/w6-rtx3060-v1.json --output benchmarks/w6-v1-audit.json
if ($LASTEXITCODE -ne 0) { throw 'W6 audit failed' }
Write-Output 'W6 per-card fixture queue completed.'
