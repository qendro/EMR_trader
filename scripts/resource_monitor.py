"""One-second Linux VM memory/swap samples from inside Docker, without JAX."""
import json
from pathlib import Path
import time

stop=Path('/control/stop-monitor')
output=Path('/reports/current-vm.jsonl')
with output.open('w') as f:
    while not stop.exists():
        fields={}
        for line in Path('/proc/meminfo').read_text().splitlines():
            name,value=line.split(':',1)
            if name in ('MemTotal','MemAvailable','SwapTotal','SwapFree'):
                fields[name+'_bytes']=int(value.split()[0])*1024
        counters=dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
        fields.update(unix_seconds=time.time(),pswpin_pages=int(counters['pswpin']),
                      pswpout_pages=int(counters['pswpout']))
        f.write(json.dumps(fields)+'\n')
        f.flush()
        time.sleep(1)
