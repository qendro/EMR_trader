param([string]$Image='neuroevo:current')
$ErrorActionPreference='Stop'
$digest=(& docker image inspect $Image --format '{{.Id}}').Trim()
$common=@('run','--rm','--memory=6g','--cpus=4','--mount','type=volume,source=neuroevo-results,target=/results',
          '--mount','type=volume,source=neuroevo-cache,target=/cache','--mount',"type=bind,source=$PWD\configs,target=/manifests,readonly",
          '-e',"NEUROEVO_IMAGE_DIGEST=$digest",'-e','NEUROEVO_BACKEND=cpu')
foreach ($task in @('memory','dag-control','synchronous-cartpole','fixed-scenario')) {
    $config="/manifests/m1-$task-diagnostic-v1.json"
    $output="/results/diagnostics/current-$task"
    & docker @common $Image benchmark --config $config --output $output
    if ($LASTEXITCODE -ne 0) { throw "Diagnostic pilot failed: $task" }
    & docker @common $Image run --config $config --output $output
    if ($LASTEXITCODE -ne 0) { throw "Diagnostic failed: $task" }
}
