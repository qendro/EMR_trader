param([string]$Image='neuroevo:m2-g2',
      [ValidateSet('cpu','rtx3070','rtx3060','all')][string]$Profile='all',
      [string]$RunRoot='/results/m2-g2-v1')
$ErrorActionPreference='Stop'
$digest=(& docker image inspect $Image --format '{{.Id}}').Trim()
$common=@('run','--rm','--memory=6g','--cpus=4',
    '--mount','type=volume,source=neuroevo-results,target=/results',
    '--mount','type=volume,source=neuroevo-cache,target=/cache',
    '-e',"NEUROEVO_IMAGE_DIGEST=$digest")
$profiles=if ($Profile -eq 'all') {@('cpu','rtx3070','rtx3060')} else {@($Profile)}
foreach ($lane in $profiles) {
    $selection=if ($lane -eq 'cpu') {@('-e','NEUROEVO_BACKEND=cpu')} else {
        $name=if ($lane -eq 'rtx3070') {'NVIDIA GeForce RTX 3070'} else {'NVIDIA GeForce RTX 3060'}
        $mib=if ($lane -eq 'rtx3070') {'8192'} else {'12288'}
        @('--gpus','all','-e','NEUROEVO_BACKEND=gpu','-e',"NEUROEVO_GPU_NAME=$name",'-e',"NEUROEVO_GPU_MIB=$mib")
    }
    & docker @common @selection --entrypoint python $Image scripts/m2_g2_compilation.py --output "$RunRoot/$lane"
    if ($LASTEXITCODE -ne 0) { throw "M2 G2 failed: $lane" }
}
