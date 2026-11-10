import os
import time
os.environ["NEUROEVO_PROCESS_STARTED"] = str(time.perf_counter())
from .backend import initialize

hardware = initialize()
from .cli import main
main(hardware=hardware)
