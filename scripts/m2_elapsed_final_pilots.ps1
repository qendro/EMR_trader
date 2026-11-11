param([string]$Image='neuroevo:elapsed-v1',
      [string]$RunRoot='/results/m2-elapsed-v1')
$ErrorActionPreference='Stop'
$digest=(& docker image inspect $Image --format '{{.Id}}').Trim()
$common=@('run','--rm','--gpus','all','--memory=6g','--cpus=4',
    '--mount','type=volume,source=neuroevo-results,target=/results',
    '--mount','type=volume,source=neuroevo-cache,target=/cache',
    '-e','NEUROEVO_BACKEND=gpu',
    '-e','NEUROEVO_GPU_NAME=NVIDIA GeForce RTX 3070',
    '-e','NEUROEVO_GPU_MIB=8192',
    '-e',"NEUROEVO_IMAGE_DIGEST=$digest")
foreach ($method in @('b0','fixed')) {
    foreach ($task in @('cartpole','memory')) {
        $version=if ($method -eq 'b0' -and $task -eq 'cartpole') {'v2'} else {'v1'}
        $config="configs/m2-elapsed-$method-$task-acceptance-$version.json"
        $output="$RunRoot/$method-$task"
        & docker @common $Image benchmark --config $config --output $output
        if ($LASTEXITCODE -ne 0) { throw "Final elapsed pilot failed: $method $task" }
    }
}
