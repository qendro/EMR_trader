"""Supported memory observations; unavailable counters remain null."""
from pathlib import Path
import resource


def read_int(path):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def memory_record():
    import jax
    return {"process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            "cgroup_peak_bytes": read_int("/sys/fs/cgroup/memory.peak"),
            "cgroup_current_bytes": read_int("/sys/fs/cgroup/memory.current"),
            "cgroup_swap_bytes": read_int("/sys/fs/cgroup/memory.swap.current"),
            "device_allocator": jax.devices()[0].memory_stats()}
