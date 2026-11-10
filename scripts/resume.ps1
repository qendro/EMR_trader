param([string]$Image='neuroevo:current')
$ErrorActionPreference='Stop'
$digest=(& docker image inspect $Image --format '{{.Id}}').Trim()
$common=@('run','--rm','--memory=6g','--cpus=4','--mount','type=volume,source=neuroevo-results,target=/results',
          '--mount','type=volume,source=neuroevo-cache,target=/cache','--mount',"type=bind,source=$PWD\scripts,target=/audit,readonly",
          '-e',"NEUROEVO_IMAGE_DIGEST=$digest",'-e','NEUROEVO_BACKEND=cpu')
$root='/results/m1-resume-current-cpu'
& docker @common --entrypoint mkdir $Image -p $root
& docker @common --entrypoint cp $Image /results/m1-current-cpu/pilot.json "$root/pilot.json"
if ($LASTEXITCODE -ne 0) { throw 'Matching CPU pilot missing' }
& docker @common $Image run --config configs/m1-acceptance-v1.json --seed 0 --stop-after 40 --output $root
if ($LASTEXITCODE -ne 0) { throw 'First 40 generations failed' }
& docker @common $Image run --config configs/m1-acceptance-v1.json --seed 0 --resume --output $root
if ($LASTEXITCODE -ne 0) { throw 'Continuation failed' }
& docker @common --entrypoint python $Image /audit/compare_resume.py /results/m1-current-cpu/seed-0 "$root/seed-0" --output /results/qualification-current/resume-40-100.json
if ($LASTEXITCODE -ne 0) { throw 'Complete state replay differs' }
& docker @common --entrypoint python $Image /audit/episode_replay.py /results/m1-current-cpu/seed-0/champion.npz --output /results/qualification-current/closed-loop-replay-cpu.json
if ($LASTEXITCODE -ne 0) { throw 'Standalone replay differs' }
& docker @common --entrypoint python $Image /audit/archive_check.py /results/m1-current-cpu/seed-0
if ($LASTEXITCODE -ne 0) { throw 'Evaluated archive differs' }
