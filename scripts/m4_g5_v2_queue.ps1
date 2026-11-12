<# Re-measure G5 with a versioned 64-agent physical microbatch ceiling. #>
param([Parameter(Mandatory=$true)][int]$MapProcessId)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$log = Join-Path $workspace 'benchmarks/results/s1-map-queue.out.log'
Write-Output 'Waiting for the S1 crossover map queue'
while (-not (Test-Path $log) -or -not (Select-String -Path $log -Pattern 'S1 crossover map queue completed.' -Quiet)) {
    if (-not (Get-Process -Id $MapProcessId -ErrorAction SilentlyContinue)) {
        throw 'S1 map queue exited without completing'
    }
    Start-Sleep -Seconds 30
}
foreach ($profile in @('cpu', 'rtx3070', 'rtx3060')) {
    for ($rep = 1; $rep -le 3; $rep++) {
        $output = "benchmarks/results/m4-g5-v2-$profile-rep$rep"
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
        Write-Output "Running M4 G5 v2 $profile replicate $rep"
        & docker run --rm @gpuArgs --cpus=4 --memory=6g --shm-size=1g `
            --entrypoint env @envArgs -v "${workspace}:/app" -v neuroevo-cache:/cache `
            neuroevo:m4-p0-comparison-v1 python -m neuroevo benchmark `
            --config configs/s2-m4-g5-evolved-v2.json --output $output
        if ($LASTEXITCODE -ne 0) { throw "M4 G5 v2 $profile replicate $rep failed" }
    }
}
& docker run --rm --entrypoint env -e PYTHONPATH=/app/src -e JAX_PLATFORMS=cpu `
    -v "${workspace}:/app" neuroevo:m4-p0-comparison-v1 `
    python scripts/m4_g5_audit.py --root benchmarks/results --prefix m4-g5-v2 `
    --output benchmarks/m4-g5-evolved-v2-audit.json
if ($LASTEXITCODE -ne 0) { throw 'M4 G5 v2 audit failed' }
Write-Output 'M4 G5 v2 queue completed.'
