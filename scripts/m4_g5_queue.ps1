<# Matched three-replicate end-to-end M4 generation measurements. #>
param([Parameter(Mandatory=$true)][int]$W7ProcessId)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$log = Join-Path $workspace 'benchmarks/results/w7-queue.out.log'
Write-Output 'Waiting for the W7 queue'
while (-not (Test-Path $log) -or -not (Select-String -Path $log -Pattern 'W7 stability queue completed.' -Quiet)) {
    if (-not (Get-Process -Id $W7ProcessId -ErrorAction SilentlyContinue)) {
        throw 'W7 queue exited without completing'
    }
    Start-Sleep -Seconds 30
}
Write-Output 'Running same-shape W6 CPU stage probe'
& docker run --rm --cpus=4 --memory=6g --shm-size=1g `
    --entrypoint env -e PYTHONPATH=/app/src -e JAX_PLATFORMS=cpu `
    -v "${workspace}:/app" neuroevo:m4-p0-comparison-v1 `
    python scripts/w6_fixture.py --population 8 --microbatch 4 `
    --output benchmarks/w6-cpu-pilot-p4.json
if ($LASTEXITCODE -ne 0) { throw 'W6 CPU stage probe failed' }
& docker run --rm --entrypoint env -e PYTHONPATH=/app/src -e JAX_PLATFORMS=cpu `
    -v "${workspace}:/app" neuroevo:m4-p0-comparison-v1 `
    python scripts/w6_ratio.py --cpu benchmarks/w6-cpu-pilot-p4.json `
    --gpu benchmarks/w6-rtx3070-pilot-p4.json benchmarks/w6-rtx3060-pilot-p4.json `
    --output benchmarks/w6-cpu-gpu-ratio-v1.json
if ($LASTEXITCODE -ne 0) { throw 'W6 CPU/GPU stage ratio failed' }
foreach ($profile in @('cpu', 'rtx3070', 'rtx3060')) {
    for ($rep = 1; $rep -le 3; $rep++) {
        $output = "benchmarks/results/m4-g5-$profile-rep$rep"
        $envArgs = @('-e', 'PYTHONPATH=/app/src')
        $gpuArgs = @()
        if ($profile -eq 'cpu') {
            $envArgs += @('-e', 'NEUROEVO_BACKEND=cpu')
        } else {
            $gpuArgs += @('--gpus', 'all')
            $envArgs += @('-e', 'NEUROEVO_BACKEND=gpu')
            $envArgs += @('-e', "NEUROEVO_GPU_NAME=NVIDIA GeForce RTX $($profile.Substring(3))")
            $envArgs += @('-e', "NEUROEVO_GPU_MIB=$(if ($profile -eq 'rtx3070') { 8192 } else { 12288 })")
        }
        Write-Output "Running M4 G5 $profile replicate $rep"
        & docker run --rm @gpuArgs --cpus=4 --memory=6g --shm-size=1g `
            --entrypoint env @envArgs -v "${workspace}:/app" -v neuroevo-cache:/cache `
            neuroevo:m4-p0-comparison-v1 python -m neuroevo benchmark `
            --config configs/s2-m4-g5-evolved-v1.json --output $output
        if ($LASTEXITCODE -ne 0) { throw "M4 G5 $profile replicate $rep failed" }
    }
}
& docker run --rm --entrypoint env -e PYTHONPATH=/app/src -e JAX_PLATFORMS=cpu `
    -v "${workspace}:/app" neuroevo:m4-p0-comparison-v1 `
    python scripts/m4_g5_audit.py --root benchmarks/results `
    --output benchmarks/m4-g5-evolved-v1-audit.json
if ($LASTEXITCODE -ne 0) { throw 'M4 G5 audit failed' }
Write-Output 'M4 G5 matched generation queue completed.'
