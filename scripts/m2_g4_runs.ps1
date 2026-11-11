param([string]$Image='neuroevo:m2-g4',
      [ValidateSet('cpu','rtx3070','rtx3060')][string]$Profile='rtx3070',
      [string]$RunRoot='/results/m2-g4-v1')
$ErrorActionPreference='Stop'
$digest=(& docker image inspect $Image --format '{{.Id}}').Trim()
$common=@('run','--rm','--memory=6g','--cpus=4',
    '--mount','type=volume,source=neuroevo-results,target=/results',
    '--mount','type=volume,source=neuroevo-cache,target=/cache',
    '-e',"NEUROEVO_IMAGE_DIGEST=$digest")
$selection=if ($Profile -eq 'cpu') {@('-e','NEUROEVO_BACKEND=cpu')} else {
    $name=if ($Profile -eq 'rtx3070') {'NVIDIA GeForce RTX 3070'} else {'NVIDIA GeForce RTX 3060'}
    $mib=if ($Profile -eq 'rtx3070') {'8192'} else {'12288'}
    @('--gpus','all','-e','NEUROEVO_BACKEND=gpu','-e',"NEUROEVO_GPU_NAME=$name",'-e',"NEUROEVO_GPU_MIB=$mib")
}
foreach ($trial in 0..2) {
    $controller=@($common+$selection+@($Image,'run','--config','configs/m2-g4-fixed-cartpole-v1.json',
        '--output',"$RunRoot/$Profile-controller-$trial"))
    $driver=@($common+$selection+@('--entrypoint','python',$Image,'scripts/m2_g4_minimal_driver.py',
        '--output',"$RunRoot/$Profile-driver-$trial"))
    $commands=if ($trial%2 -eq 0) {@($controller,$driver)} else {@($driver,$controller)}
    foreach ($command in $commands) {
        & docker @command
        if ($LASTEXITCODE -ne 0) { throw "M2 G4 failed: $Profile trial $trial" }
    }
}
