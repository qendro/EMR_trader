param([string]$Image='neuroevo:visual',
      [ValidateSet('rtx3070','rtx3060','cpu')][string]$Profile='rtx3070',
      [ValidateSet('fixed','direct','b0','all')][string]$Method='all',
      [string]$RunRoot='/results/h2-visual-diagnostic-v1')
$ErrorActionPreference='Stop'
$digest=(& docker image inspect $Image --format '{{.Id}}').Trim()
$common=@('run','--rm','--memory=6g','--cpus=4','--mount','type=volume,source=neuroevo-results,target=/results',
    '--mount','type=volume,source=neuroevo-cache,target=/cache','-e',"NEUROEVO_IMAGE_DIGEST=$digest")
$selection=if ($Profile -eq 'cpu') {@('-e','NEUROEVO_BACKEND=cpu')} else {
    $name=if ($Profile -eq 'rtx3070') {'NVIDIA GeForce RTX 3070'} else {'NVIDIA GeForce RTX 3060'}
    $mib=if ($Profile -eq 'rtx3070') {'8192'} else {'12288'}
    @('--gpus','all','-e','NEUROEVO_BACKEND=gpu','-e',"NEUROEVO_GPU_NAME=$name",'-e',"NEUROEVO_GPU_MIB=$mib")
}
$methods=if ($Method -eq 'all') {@('b0','direct','fixed')} else {@($Method)}
foreach ($methodName in $methods) {
    $config="configs/h2-visual-$methodName-diagnostic-v1.json"
    $output="$RunRoot/$Profile-$methodName"
    & docker @common @selection $Image run --config $config --output $output
    if ($LASTEXITCODE -ne 0) { throw "Visual diagnostic failed: $Profile $methodName" }
}
