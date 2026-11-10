param([string]$Image='neuroevo:b0',
      [ValidateSet('cpu','rtx3070','rtx3060','all')][string]$Profile='all',
      [ValidateSet('cartpole','memory','all')][string]$Task='all',
      [string]$RunRoot='/results/b0-v1')
$ErrorActionPreference='Stop'
$digest=(& docker image inspect $Image --format '{{.Id}}').Trim()
$common=@('run','--rm','--memory=6g','--cpus=4','--mount','type=volume,source=neuroevo-results,target=/results',
    '--mount','type=volume,source=neuroevo-cache,target=/cache','-e',"NEUROEVO_IMAGE_DIGEST=$digest")
$profiles=if ($Profile -eq 'all') {@('cpu','rtx3070','rtx3060')} else {@($Profile)}
$tasks=if ($Task -eq 'all') {@('cartpole','memory')} else {@($Task)}
foreach ($lane in $profiles) {
    if ($lane -eq 'cpu') { $selection=@('-e','NEUROEVO_BACKEND=cpu') }
    else {
        $name=if ($lane -eq 'rtx3070') {'NVIDIA GeForce RTX 3070'} else {'NVIDIA GeForce RTX 3060'}
        $mib=if ($lane -eq 'rtx3070') {'8192'} else {'12288'}
        $selection=@('--gpus','all','-e','NEUROEVO_BACKEND=gpu','-e',"NEUROEVO_GPU_NAME=$name",'-e',"NEUROEVO_GPU_MIB=$mib")
    }
    foreach ($experimentTask in $tasks) {
        $output="$RunRoot/$lane-$experimentTask"
        $config="configs/b0-$experimentTask-acceptance-v1.json"
        & docker @common @selection $Image benchmark --config $config --output $output
        if ($LASTEXITCODE -ne 0) { throw "G9 pilot failed: $lane $experimentTask" }
        & docker @common @selection $Image run --config $config --output $output
        if ($LASTEXITCODE -ne 0) { throw "Baseline run failed: $lane $experimentTask" }
        & docker @common @selection $Image replay --phenotype "$output/seed-0/champion.npz" --output "$output/portable-replay.json"
        if ($LASTEXITCODE -ne 0) { throw "Baseline portable replay failed: $lane $experimentTask" }
    }
}
