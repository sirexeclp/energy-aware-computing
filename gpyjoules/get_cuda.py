"""Print cuda version number."""
import os
import re
import pynvml3
from pathlib import Path

with pynvml3.NVMLLib() as lib:
    supported = lib.system.get_cuda_driver_version()
    print("supported cuda version: ", supported)


def parse_cuda_version_string(ver_str):
    regex = "^CUDA Version (?P<major>\d+)\.(?P<minor>\d+)\.?(?P<build>\d*)"
    return tuple(re.match(regex, ver_str).groupdict().values())


CUDA_VERSION_FILE = Path("/usr/local/cuda/version.txt")
if CUDA_VERSION_FILE.exists():
    print(parse_cuda_version_string(CUDA_VERSION_FILE.read_text()))
else:
    print(os.environ.get("CUDA_VERSION"))
