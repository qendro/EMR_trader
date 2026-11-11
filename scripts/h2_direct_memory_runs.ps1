param([string]$Image='neuroevo:h2',
      [ValidateSet('cpu','rtx3070','rtx3060','all')][string]$Profile='rtx3070',
      [string]$RunRoot='/results/h2-direct-memory-acceptance-v1')
$ErrorActionPreference='Stop'
$digest=(& docker image inspect $Image --format '{{.Id}}').Trim()
$common=@('run','--rm','--memory=6g','--cpus=4',
    '--mount','type=volume,source=neuroevo-results,target=/results',
    '--mount','type=volume,source=neuroevo-cache,target=/cache',
    '--mount',"type=bind,source=$PWD\scripts,target=/audit,readonly",
    '-e',"NEUROEVO_IMAGE_DIGEST=$digest")
$profiles=if ($Profile -eq 'all') {@('cpu','rtx3070','rtx3060')} else {@($Profile)}
foreach ($lane in $profiles) {
    if ($lane -eq 'cpu') { $selection=@('-e','NEUROEVO_BACKEND=cpu') }
    else {
        $name=if ($lane -eq 'rtx3070') {'NVIDIA GeForce RTX 3070'} else {'NVIDIA GeForce RTX 3060'}
        $mib=if ($lane -eq 'rtx3070') {'8192'} else {'12288'}
        $selection=@('--gpus','all','-e','NEUROEVO_BACKEND=gpu',
            '-e',"NEUROEVO_GPU_NAME=$name",'-e',"NEUROEVO_GPU_MIB=$mib")
    }
    $output="$RunRoot/$lane-memory"
    $config='configs/h2-direct-memory-acceptance-v1.json'
    & docker @common @selection $Image benchmark --config $config --output $output
    if ($LASTEXITCODE -ne 0) { throw "G9 pilot failed: $lane memory" }
    & docker @common @selection $Image run --config $config --output $output
    if ($LASTEXITCODE -ne 0) { throw "Direct memory run failed: $lane" }
    & docker @common @selection --entrypoint python $Image /audit/memory_portable_replay.py --experiment $output --output "$output/task-portable-replay.json"
    if ($LASTEXITCODE -ne 0) { throw "Direct memory task replay failed: $lane" }
}
