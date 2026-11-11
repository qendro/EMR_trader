param([string]$Image='neuroevo:w8',
      [ValidateSet('cpu','rtx3070','rtx3060','all')][string]$Profile='all',
      [string]$RunRoot='/results/w8-v2')
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
    & docker @common @selection --entrypoint python $Image /audit/w8_check.py --output "$RunRoot/$lane.json"
    if ($LASTEXITCODE -ne 0) { throw "W8 contract check failed: $lane" }
}
