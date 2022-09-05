import re
import pynvml3

with pynvml3.NVMLLib() as lib:
	supported = lib.system.get_cuda_driver_version()
	print("supported cuda version: ", supported)

def parse_cuda_version_string(ver_str):
	regex = "^CUDA Version (?P<major>\d+)\.(?P<minor>\d+)\.?(?P<build>\d*)"
	return tuple(re.match(regex, ver_str).groupdict().values())

with open("/usr/local/cuda/version.txt", "r") as f:
	print(parse_cuda_version_string(f.read()))
