<# Three matched G5 replicas on the bulk-extraction source, one device per process. #>
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
foreach ($profile in @('cpu', 'rtx3070', 'rtx3060')) {
    for ($rep = 1; $rep -le 3; $rep++) {
        $output = "benchmarks/results/m4-g5-v3-$profile-rep$rep"
        if (Test-Path (Join-Path $workspace "$output/pilot.json")) {
            Write-Output "Retaining completed M4 G5 v3 $profile replicate $rep"
            continue
        }
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
        Write-Output "Running M4 G5 v3 $profile replicate $rep"
        & docker run --rm @gpuArgs --cpus=4 --memory=6g --shm-size=1g `
            --entrypoint env @envArgs -v "${workspace}:/app" -v neuroevo-cache:/cache `
            neuroevo:m4-p0-comparison-v1 python -m neuroevo benchmark `
            --config configs/s2-m4-g5-evolved-v3.json --output $output
        if ($LASTEXITCODE -ne 0) { throw "M4 G5 v3 $profile replicate $rep failed" }
    }
}
& docker run --rm --entrypoint env -e PYTHONPATH=/app/src -e JAX_PLATFORMS=cpu `
    -v "${workspace}:/app" neuroevo:m4-p0-comparison-v1 `
    python scripts/m4_g5_audit.py --root benchmarks/results --prefix m4-g5-v3 `
    --output benchmarks/m4-g5-evolved-v3-audit.json
if ($LASTEXITCODE -ne 0) { throw 'M4 G5 v3 audit failed' }
Write-Output 'M4 G5 v3 queue completed.'
