<# Measure the S1 crossover map after the P1 dossier releases both GPUs. #>
param([Parameter(Mandatory=$true)][int]$DossierProcessId)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$log = Join-Path $workspace 'benchmarks/results/p1-dossier-queue.out.log'
Write-Output 'Waiting for the P1 dossier queue'
while (-not (Test-Path $log) -or -not (Select-String -Path $log -Pattern 'P1 dossier queue completed.' -Quiet)) {
    if (-not (Get-Process -Id $DossierProcessId -ErrorAction SilentlyContinue)) {
        throw 'P1 dossier queue exited without completing'
    }
    Start-Sleep -Seconds 30
}
Write-Output 'Measuring S1 crossover map on CPU'
& docker run --rm --cpus=4 --memory=6g --shm-size=1g `
    --entrypoint env -e PYTHONPATH=/app/src -e JAX_PLATFORMS=cpu `
    -v "${workspace}:/app" -v neuroevo-cache:/cache `
    neuroevo:m4-p0-comparison-v1 python scripts/s1_crossover_map.py `
    --output benchmarks/s1-map-cpu-v1.json
if ($LASTEXITCODE -ne 0) { throw 'S1 CPU crossover map failed' }
$inventory = & nvidia-smi --query-gpu=uuid,name --format=csv,noheader
foreach ($card in @(@{name='NVIDIA GeForce RTX 3070'; label='rtx3070'},
                    @{name='NVIDIA GeForce RTX 3060'; label='rtx3060'})) {
    $row = $inventory | Where-Object { $_.EndsWith($card.name) }
    if (@($row).Count -ne 1) { throw "S1 GPU inventory mismatch: $($card.name)" }
    $uuid = ($row -split ',')[0].Trim()
    Write-Output "Measuring S1 crossover map on $($card.name)"
    & docker run --rm --gpus all --cpus=4 --memory=6g --shm-size=1g `
        --entrypoint env -e PYTHONPATH=/app/src -e JAX_PLATFORMS=cuda `
        -e "CUDA_VISIBLE_DEVICES=$uuid" -e XLA_PYTHON_CLIENT_PREALLOCATE=false `
        -v "${workspace}:/app" -v neuroevo-cache:/cache `
        neuroevo:m4-p0-comparison-v1 python scripts/s1_crossover_map.py `
        --output "benchmarks/s1-map-$($card.label)-v1.json"
    if ($LASTEXITCODE -ne 0) { throw "S1 crossover map failed on $($card.name)" }
}
& docker run --rm --entrypoint env -e PYTHONPATH=/app/src -e JAX_PLATFORMS=cpu `
    -v "${workspace}:/app" neuroevo:m4-p0-comparison-v1 `
    python scripts/s1_map_audit.py --cpu benchmarks/s1-map-cpu-v1.json `
    --rtx3070 benchmarks/s1-map-rtx3070-v1.json `
    --rtx3060 benchmarks/s1-map-rtx3060-v1.json `
    --output benchmarks/s1-map-v1-audit.json
if ($LASTEXITCODE -ne 0) { throw 'S1 crossover map audit failed' }
Write-Output 'S1 crossover map queue completed.'
