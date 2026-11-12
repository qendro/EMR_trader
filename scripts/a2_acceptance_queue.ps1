<# Wait for the memory acceptance container, audit it, then run and audit CartPole.
   The GPU tasks are sequential so their measured times do not overlap. #>
$ErrorActionPreference = 'Stop'
$workspacePath = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$imageDigest = 'sha256:89a3b14f9695910f70f9f060a4013ebca188e890b53ec05365d875649a4fc569'
$memoryName = 'neuroevo-a2-memory-acceptance'
$cartpoleName = 'neuroevo-a2-cartpole-acceptance'

function Audit-Acceptance($taskName) {
    $resultRoot = "/results/a2-$taskName-acceptance-v1-rtx3070-image"
    $outputFile = "/app/benchmarks/a2-$taskName-acceptance-v1-audit.json"
    & docker run --rm --memory=6g --cpus=4 `
        -v 'neuroevo-results:/results:ro' -v "${workspacePath}:/app" `
        --entrypoint python neuroevo:a2-v1 scripts/a2_acceptance_audit.py `
        --root $resultRoot --output $outputFile
    if ($LASTEXITCODE -ne 0) { throw "A2 $taskName acceptance audit failed" }
}

Write-Output "Waiting for $memoryName"
$memoryStatus = & docker wait $memoryName
if ($LASTEXITCODE -ne 0 -or [int]$memoryStatus -ne 0) {
    throw "A2 memory container failed: $memoryStatus"
}
Audit-Acceptance 'memory'

$existing = & docker ps -a --filter "name=^/$cartpoleName$" --format '{{.Names}}'
if ($existing) { throw "A2 CartPole container already exists: $existing" }
Write-Output "Starting $cartpoleName"
& docker run -d --name $cartpoleName --gpus all --cpus=4 --memory=6g --shm-size=1g `
    -e NEUROEVO_BACKEND=gpu -e 'NEUROEVO_GPU_NAME=NVIDIA GeForce RTX 3070' `
    -e NEUROEVO_GPU_MIB=8192 -e "NEUROEVO_IMAGE_DIGEST=$imageDigest" `
    -v neuroevo-results:/results -v neuroevo-cache:/cache `
    neuroevo:a2-v1 run --config configs/a2-cartpole-acceptance-v1.json `
    --output /results/a2-cartpole-acceptance-v1-rtx3070-image
if ($LASTEXITCODE -ne 0) { throw 'Could not start A2 CartPole acceptance' }

Write-Output "Waiting for $cartpoleName"
$cartpoleStatus = & docker wait $cartpoleName
if ($LASTEXITCODE -ne 0 -or [int]$cartpoleStatus -ne 0) {
    throw "A2 CartPole container failed: $cartpoleStatus"
}
Audit-Acceptance 'cartpole'
Write-Output 'A2 gating-card acceptance runs and audits finished.'
