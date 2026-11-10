param([string]$Image='neuroevo:current',
      [ValidateSet('cpu','rtx3070','rtx3060','all')][string]$Profile='all')
$ErrorActionPreference='Stop'
$digest=(& docker image inspect $Image --format '{{.Id}}').Trim()
$common=@('run','--rm','--memory=6g','--cpus=4','--mount','type=volume,source=neuroevo-results,target=/results',
          '--mount','type=volume,source=neuroevo-cache,target=/cache','--mount',"type=bind,source=$PWD\scripts,target=/audit,readonly",
          '-e',"NEUROEVO_IMAGE_DIGEST=$digest")
$profiles=if ($Profile -eq 'all') {@('cpu','rtx3070','rtx3060')} else {@($Profile)}
foreach ($lane in $profiles) {
    if ($lane -eq 'cpu') { $selection=@('-e','NEUROEVO_BACKEND=cpu') }
    else {
        $name=if ($lane -eq 'rtx3070') {'NVIDIA GeForce RTX 3070'} else {'NVIDIA GeForce RTX 3060'}
        $mib=if ($lane -eq 'rtx3070') {'8192'} else {'12288'}
        $selection=@('--gpus','all','-e','NEUROEVO_BACKEND=gpu','-e',"NEUROEVO_GPU_NAME=$name",'-e',"NEUROEVO_GPU_MIB=$mib")
    }
    foreach ($trial in 0..2) {
        # Same seed, production dimensions, diagnostics and checkpoint interval.
        # G4 compares generations 2..10 inclusive; terminal selection/checkpoint excluded.
        $generations=if ($trial -eq 0) {'102'} else {'12'}
        $root="/results/g4-$lane"
        $controller=@($common+$selection+@($Image,'benchmark','--config','configs/m1-acceptance-v1.json','--output',"$root/controller-$trial"))
        $driver=@($common+$selection+@('--entrypoint','python',$Image,'/audit/minimal_driver.py','--output',"$root/driver-$trial",'--generations',$generations))
        if ($trial % 2 -eq 0) { $commands=@($controller,$driver) } else { $commands=@($driver,$controller) }
        foreach ($command in $commands) {
            & docker @command
            if ($LASTEXITCODE -ne 0) { throw "Performance cell failed: $lane trial $trial" }
        }
    }
}
