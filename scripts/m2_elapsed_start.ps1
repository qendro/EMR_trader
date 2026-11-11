param([string]$Image='neuroevo:elapsed-v5',
      [string]$RunRoot='/results/m2-elapsed-v5',
      [string]$Container='neuroevo-m2-elapsed')
$ErrorActionPreference='Stop'
$digest=(& docker image inspect $Image --format '{{.Id}}').Trim()
if ($LASTEXITCODE -ne 0 -or -not $digest) { throw "Image unavailable: $Image" }
& docker run -d --name $Container --restart 'on-failure:3' --gpus all --memory=6g --cpus=4 `
    --mount 'type=volume,source=neuroevo-results,target=/results' `
    --mount 'type=volume,source=neuroevo-cache,target=/cache' `
    --mount "type=bind,source=$PSScriptRoot,target=/audit,readonly" `
    -e 'NEUROEVO_BACKEND=gpu' `
    -e 'NEUROEVO_GPU_NAME=NVIDIA GeForce RTX 3070' `
    -e 'NEUROEVO_GPU_MIB=8192' `
    -e "NEUROEVO_IMAGE_DIGEST=$digest" `
    --entrypoint python $Image /audit/m2_elapsed_acceptance.py --output $RunRoot
if ($LASTEXITCODE -ne 0) { throw 'Elapsed comparison container failed to start' }
