param([ValidateSet('cpu','rtx3070','rtx3060')][string]$Profile='cpu',
      [string]$Image='neuroevo:current', [string]$RunPrefix='m1-current')
$ErrorActionPreference='Stop'
$digest=(& docker image inspect $Image --format '{{.Id}}').Trim()
if ($LASTEXITCODE -ne 0) { throw 'Build the pinned image first' }
$common=@('run','--rm','--memory=6g','--cpus=4','--mount','type=volume,source=neuroevo-results,target=/results',
          '--mount','type=volume,source=neuroevo-cache,target=/cache','-e',"NEUROEVO_IMAGE_DIGEST=$digest")
if ($Profile -eq 'cpu') { $selection=@('-e','NEUROEVO_BACKEND=cpu') }
else {
    $name=if ($Profile -eq 'rtx3070') {'NVIDIA GeForce RTX 3070'} else {'NVIDIA GeForce RTX 3060'}
    $mib=if ($Profile -eq 'rtx3070') {'8192'} else {'12288'}
    $selection=@('--gpus','all','-e','NEUROEVO_BACKEND=gpu','-e',"NEUROEVO_GPU_NAME=$name",'-e',"NEUROEVO_GPU_MIB=$mib")
}
$root="/results/$RunPrefix-$Profile"
& docker @common @selection $Image benchmark --config configs/m1-acceptance-v1.json --output $root
if ($LASTEXITCODE -ne 0) { throw "G9 failed on $Profile" }
& docker @common @selection $Image run --config configs/m1-acceptance-v1.json --output $root
if ($LASTEXITCODE -ne 0) { throw "Acceptance failed on $Profile" }
