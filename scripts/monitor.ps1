param([string]$Output='benchmarks/results/current-gpu.csv',
      [string]$StopPath='artifacts/stop-monitor')
$ErrorActionPreference='Stop'
$query='timestamp,uuid,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu,power.draw,clocks.sm'
'timestamp,uuid,name,total_mib,used_mib,free_mib,temperature_c,utilization_percent,power_w,clock_mhz' | Set-Content -Encoding utf8 $Output
while (-not (Test-Path -LiteralPath $StopPath)) {
    & nvidia-smi "--query-gpu=$query" --format=csv,noheader,nounits | Add-Content -Encoding utf8 $Output
    if ($LASTEXITCODE -ne 0) { throw 'GPU telemetry failed' }
    Start-Sleep -Seconds 1
}
