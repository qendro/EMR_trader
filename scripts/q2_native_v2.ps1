param([string]$Image='neuroevo:current')
$ErrorActionPreference='Stop'
$digest=(& docker image inspect $Image --format '{{.Id}}').Trim()
$common=@('run','--rm','--memory=6g','--cpus=4','--mount','type=volume,source=neuroevo-results,target=/results',
    '--mount','type=volume,source=neuroevo-cache,target=/cache',
    '--mount',"type=bind,source=$PWD\scripts,target=/audit,readonly",'--mount',"type=bind,source=$PWD\configs,target=/manifests,readonly",
    '-e',"NEUROEVO_IMAGE_DIGEST=$digest",'--entrypoint','python')
foreach ($profile in @('cpu','rtx3070','rtx3060')) {
    if ($profile -eq 'cpu') { $selection=@('-e','NEUROEVO_BACKEND=cpu') }
    else {
        $name=if ($profile -eq 'rtx3070') {'NVIDIA GeForce RTX 3070'} else {'NVIDIA GeForce RTX 3060'}
        $mib=if ($profile -eq 'rtx3070') {'8192'} else {'12288'}
        $selection=@('--gpus','all','-e','NEUROEVO_BACKEND=gpu','-e',"NEUROEVO_GPU_NAME=$name",'-e',"NEUROEVO_GPU_MIB=$mib")
    }
    foreach ($mode in @('direct','fixed')) {
        foreach ($seed in 700..702) {
            & docker @common @selection -e "JAX_COMPILATION_CACHE_DIR=/cache/q2-native-v2/$profile-$mode" $Image /audit/upstream_native_v2.py --mode $mode --seed $seed --manifest /manifests/q2-native-v2.json --output "/results/q2-native-v2/$profile-$mode-$seed"
            if ($LASTEXITCODE -ne 0) { Write-Warning "Recorded failed cell: $profile $mode $seed" }
        }
    }
}
