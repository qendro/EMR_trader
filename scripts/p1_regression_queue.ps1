<# Run current-source GPU suites and portable artifact checks after M4 probes. #>
param([Parameter(Mandatory=$true)][int]$M4ProcessId)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$log = Join-Path $workspace 'benchmarks/results/m4-g5-queue.out.log'
Write-Output 'Waiting for the M4 G5 queue'
while (-not (Test-Path $log) -or -not (Select-String -Path $log -Pattern 'M4 G5 matched generation queue completed.' -Quiet)) {
    if (-not (Get-Process -Id $M4ProcessId -ErrorAction SilentlyContinue)) {
        throw 'M4 G5 queue exited without completing'
    }
    Start-Sleep -Seconds 30
}
$artifact = 'benchmarks/results/evidence/s0-development-v1/s0-memory-smoke-v1/seed-900/champion.npz'
foreach ($profile in @('cpu', 'rtx3070', 'rtx3060')) {
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
    Write-Output "Running P1 $profile regression suite"
    & docker run --rm @gpuArgs --cpus=4 --memory=6g --shm-size=1g `
        @envArgs -v "${workspace}:/app" -v neuroevo-cache:/cache `
        neuroevo:m4-p0-comparison-v1 test -q tests --disable-warnings `
        --junitxml "benchmarks/results/p1-tests-$profile.xml"
    if ($LASTEXITCODE -ne 0) { throw "P1 $profile regression failed" }
    Write-Output "Replaying P1 artifact on $profile"
    & docker run --rm @gpuArgs --cpus=4 --memory=6g --shm-size=1g `
        @envArgs -v "${workspace}:/app" -v neuroevo-cache:/cache `
        neuroevo:m4-p0-comparison-v1 replay --phenotype $artifact `
        --output "benchmarks/p1-replay-$profile-v1.json"
    if ($LASTEXITCODE -ne 0) { throw "P1 $profile replay failed" }
}
Write-Output 'Checking CPU checkpoint continuation after container replacement'
$runArgs = @('--cpus=4', '--memory=6g', '--shm-size=1g',
    '-e', 'PYTHONPATH=/app/src', '-e', 'NEUROEVO_BACKEND=cpu',
    '-v', "${workspace}:/app", '-v', 'neuroevo-cache:/cache',
    'neuroevo:m4-p0-comparison-v1', 'run',
    '--config', 'configs/s2-adaptive-memory-smoke-v1.json',
    '--output', 'benchmarks/results/p1-resume-v1', '--seed', '900')
& docker run --rm @runArgs --stop-after 1
if ($LASTEXITCODE -ne 0) { throw 'P1 pre-replacement checkpoint failed' }
& docker run --rm @runArgs --resume
if ($LASTEXITCODE -ne 0) { throw 'P1 post-replacement continuation failed' }
& docker run --rm --entrypoint env -e PYTHONPATH=/app/src -e JAX_PLATFORMS=cpu `
    -v "${workspace}:/app" neuroevo:m4-p0-comparison-v1 `
    python scripts/p1_audit.py --benchmarks-root benchmarks `
    --output benchmarks/p1-crossbackend-v1-audit.json
if ($LASTEXITCODE -ne 0) { throw 'P1 portability audit failed' }
Write-Output 'P1 cross-backend regression and replay queue completed.'
