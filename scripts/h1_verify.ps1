param([string]$Image='neuroevo:h1')
$ErrorActionPreference='Stop'
$digest=(& docker image inspect $Image --format '{{.Id}}').Trim()
$common=@('run','--rm','--memory=6g','--cpus=4','--mount','type=volume,source=neuroevo-results,target=/results',
    '--mount','type=volume,source=neuroevo-cache,target=/cache',
    '--mount',"type=bind,source=$PWD\scripts,target=/audit,readonly",'--mount',"type=bind,source=$PWD\configs,target=/manifests,readonly",
    '-e',"NEUROEVO_IMAGE_DIGEST=$digest",'-e','OPENBLAS_NUM_THREADS=4')
foreach ($profile in @('cpu','rtx3070','rtx3060')) {
    if ($profile -eq 'cpu') { $selection=@('-e','NEUROEVO_BACKEND=cpu') }
    else {
        $name=if ($profile -eq 'rtx3070') {'NVIDIA GeForce RTX 3070'} else {'NVIDIA GeForce RTX 3060'}
        $mib=if ($profile -eq 'rtx3070') {'8192'} else {'12288'}
        $selection=@('--gpus','all','-e','NEUROEVO_BACKEND=gpu','-e',"NEUROEVO_GPU_NAME=$name",'-e',"NEUROEVO_GPU_MIB=$mib")
    }
    & docker @common @selection --entrypoint python $Image /audit/fanin_calibration_v2.py --coefficient /manifests/fanin-calibration-v2.json --output "/results/h1-calibration-v2/$profile-validation.json"
    if ($LASTEXITCODE -ne 0) { throw "Frozen calibration validation failed: $profile" }
    & docker @common @selection --entrypoint python $Image /audit/h1_audit.py --coefficient /manifests/fanin-calibration-v2.json --output "/results/h1-audit-v2/$profile"
    if ($LASTEXITCODE -ne 0) { throw "Large fixture failed: $profile" }
    foreach ($task in @('cartpole','memory')) {
        $output="/results/h1-smoke-v2/$profile-$task"
        & docker @common @selection $Image run --config "configs/h1-fixed-$task-smoke-v1.json" --output $output
        if ($LASTEXITCODE -ne 0) { throw "Fixed wiring failed: $profile $task" }
        & docker @common @selection $Image replay --phenotype "$output/seed-900/champion.npz" --output "$output/replay.json"
        if ($LASTEXITCODE -ne 0) { throw "Portable fixed replay failed: $profile $task" }
    }
}
