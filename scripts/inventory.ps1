param([string]$OutputPath = 'benchmarks/results/host-inventory.json')
$ErrorActionPreference = 'Stop'
$machine = Get-CimInstance Win32_ComputerSystem
$operating = Get-CimInstance Win32_OperatingSystem
$record = [ordered]@{
  schema = 1
  date = '2026-11-10'
  windows = $operating.Caption
  windows_build = $operating.BuildNumber
  host_ram_bytes = $machine.TotalPhysicalMemory
  logical_cpus = $machine.NumberOfLogicalProcessors
  gpus_csv = (& nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader,nounits | Out-String).Trim()
  docker = (& docker version --format '{{json .}}' | ConvertFrom-Json)
  docker_resources = (& docker info --format '{{json .}}' | ConvertFrom-Json | Select-Object KernelVersion,OperatingSystem,Architecture,NCPU,MemTotal)
  wsl = ((& wsl --list --verbose | Out-String) -replace "`0", '').Trim()
  gating_card = 'NVIDIA GeForce RTX 3070'
  gating_card_mib = 8192
  host_container_limit_bytes = 6442450944
  status = 'inventory-and-passthrough-only-not-JAX-qualification'
}
$parent = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Force -Path $parent | Out-Null
[System.IO.File]::WriteAllText((Join-Path (Get-Location) $OutputPath), ($record | ConvertTo-Json -Depth 15))
