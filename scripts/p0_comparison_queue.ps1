<# Run the frozen P0 comparison pilots and acceptance seeds after adaptive P0. #>
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$image = 'neuroevo:m4-p0-comparison-v1'
$digest = 'sha256:0e05d80d70b9847916ccbb654a62782ad3ae97b69f600291f27f28865660afdd'

Write-Output 'Waiting for neuroevo-p0-adaptive-acceptance'
$status = & docker wait neuroevo-p0-adaptive-acceptance
if ($LASTEXITCODE -ne 0 -or [int]$status -ne 0) {
    throw "Adaptive P0 acceptance failed: $status"
}

foreach ($method in @('direct', 'fixed', 'weight')) {
    $name = "p0-partial-$method-acceptance-v1"
    $hostOutput = Join-Path $workspace "benchmarks/results/$name"
    New-Item -ItemType Directory -Force -Path $hostOutput | Out-Null
    $containerOutput = "/results/$name"
    $mount = "${hostOutput}:${containerOutput}"
    $config = "configs/$name.json"
    Write-Output "Starting $method G9 pilot"
    & docker run --rm --gpus all --cpus=4 --memory=6g --shm-size=1g `
        -e NEUROEVO_BACKEND=gpu -e 'NEUROEVO_GPU_NAME=NVIDIA GeForce RTX 3070' `
        -e NEUROEVO_GPU_MIB=8192 -e "NEUROEVO_IMAGE_DIGEST=$digest" `
        -v $mount -v neuroevo-cache:/cache $image benchmark `
        --config $config --output $containerOutput
    if ($LASTEXITCODE -ne 0) { throw "$method G9 pilot failed" }
    $pilot = Get-Content (Join-Path $hostOutput 'pilot.json') -Raw | ConvertFrom-Json
    if (-not $pilot.passes) { throw "$method G9 pilot did not fit the budget" }
    Write-Output "Starting $method five-seed acceptance"
    & docker run --rm --gpus all --cpus=4 --memory=6g --shm-size=1g `
        -e NEUROEVO_BACKEND=gpu -e 'NEUROEVO_GPU_NAME=NVIDIA GeForce RTX 3070' `
        -e NEUROEVO_GPU_MIB=8192 -e "NEUROEVO_IMAGE_DIGEST=$digest" `
        -v $mount -v neuroevo-cache:/cache $image run `
        --config $config --output $containerOutput
    if ($LASTEXITCODE -ne 0) { throw "$method acceptance failed" }
}
Write-Output 'P0 comparison queue completed.'
