<# Run adaptive visual diagnostics after the gating-card acceptance queue finishes. #>
$ErrorActionPreference = 'Stop'
$imageDigest = 'sha256:b3d48ce6f35587811c670d3e79d8898813390eb9b02f56acd0d133c81c03de9b'
$cartpoleName = 'neuroevo-a2-cartpole-acceptance'

Write-Output "Waiting for $cartpoleName to be created"
for ($attempt = 0; $attempt -lt 480; $attempt++) {
    if (& docker ps -a --filter "name=^/$cartpoleName$" --format '{{.Names}}') { break }
    Start-Sleep -Seconds 15
}
if ($attempt -eq 480) { throw 'CartPole acceptance was not started within two hours' }
$cartpoleStatus = & docker wait $cartpoleName
if ($LASTEXITCODE -ne 0 -or [int]$cartpoleStatus -ne 0) {
    throw "CartPole acceptance failed; visual diagnostic remains pending: $cartpoleStatus"
}
Write-Output 'Running visual CPU smoke'
& docker run --rm --cpus=4 --memory=6g --shm-size=1g `
    -e NEUROEVO_BACKEND=cpu -e JAX_PLATFORMS=cpu `
    -e "NEUROEVO_IMAGE_DIGEST=$imageDigest" `
    -v neuroevo-results:/results -v neuroevo-cache:/cache `
    neuroevo:a2-visual-v1 run --config configs/a2-visual-smoke-v1.json `
    --output /results/a2-visual-cpu-smoke-v1
if ($LASTEXITCODE -ne 0) { throw 'Adaptive visual CPU smoke failed' }

Write-Output 'Running visual RTX 3070 smoke'
& docker run --rm --gpus all --cpus=4 --memory=6g --shm-size=1g `
    -e NEUROEVO_BACKEND=gpu -e 'NEUROEVO_GPU_NAME=NVIDIA GeForce RTX 3070' `
    -e NEUROEVO_GPU_MIB=8192 -e "NEUROEVO_IMAGE_DIGEST=$imageDigest" `
    -v neuroevo-results:/results -v neuroevo-cache:/cache `
    neuroevo:a2-visual-v1 run --config configs/a2-visual-smoke-v1.json `
    --output /results/a2-visual-gpu-smoke-v1
if ($LASTEXITCODE -ne 0) { throw 'Adaptive visual GPU smoke failed' }

Write-Output 'Running three-seed adaptive visual diagnostic'
& docker run --rm --gpus all --cpus=4 --memory=6g --shm-size=1g `
    -e NEUROEVO_BACKEND=gpu -e 'NEUROEVO_GPU_NAME=NVIDIA GeForce RTX 3070' `
    -e NEUROEVO_GPU_MIB=8192 -e "NEUROEVO_IMAGE_DIGEST=$imageDigest" `
    -v neuroevo-results:/results -v neuroevo-cache:/cache `
    neuroevo:a2-visual-v1 run --config configs/a2-visual-adaptive-diagnostic-v1.json `
    --output /results/a2-visual-adaptive-diagnostic-v1-rtx3070-image
if ($LASTEXITCODE -ne 0) { throw 'Adaptive visual diagnostic failed' }
Write-Output 'A2 visual smoke and three-seed diagnostic finished.'
