param([string]$Image='neuroevo:release', [string]$RunRoot='/results/qualification-reproduction',
      [string[]]$Profiles=@('cpu','rtx3070','rtx3060'))
$ErrorActionPreference='Stop'
$digest=(& docker image inspect $Image --format '{{.Id}}').Trim()
if ($LASTEXITCODE -ne 0) { throw 'Build the pinned image first' }
$common=@('run','--rm','--memory=6g','--cpus=4','--mount','type=volume,source=neuroevo-results,target=/results',
          '--mount','type=volume,source=neuroevo-cache,target=/cache','-e',"NEUROEVO_IMAGE_DIGEST=$digest")
foreach ($profile in $Profiles) {
    if ($profile -notin @('cpu','rtx3070','rtx3060')) { throw "Unknown profile: $profile" }
    if ($profile -eq 'cpu') { $selection=@('-e','NEUROEVO_BACKEND=cpu') }
    else {
        $name=if ($profile -eq 'rtx3070') {'NVIDIA GeForce RTX 3070'} else {'NVIDIA GeForce RTX 3060'}
        $mib=if ($profile -eq 'rtx3070') {'8192'} else {'12288'}
        $selection=@('--gpus','all','-e','NEUROEVO_BACKEND=gpu','-e',"NEUROEVO_GPU_NAME=$name",'-e',"NEUROEVO_GPU_MIB=$mib")
    }
    & docker @common @selection $Image qualify --output "$RunRoot/q1-$profile.json"
    if ($LASTEXITCODE -ne 0) { throw "Q1 failed on $profile" }
    & docker @common @selection $Image test -q "--junitxml=$RunRoot/tests-$profile.xml"
    if ($LASTEXITCODE -ne 0) { throw "Contract tests failed on $profile" }
    & docker @common @selection $Image oracle --output "$RunRoot/oracle-$profile.json"
    if ($LASTEXITCODE -ne 0) { throw "Oracle failed on $profile" }
}
