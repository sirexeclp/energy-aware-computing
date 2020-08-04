from dataclasses import dataclass, asdict
import json
from system import System
import platform
import distro
from typing import List
import pip
import sys

@dataclass
class SystemInfo:
    """Dataclass with System Information"""
    device_name: str
    cuda_capability: str
    driver_version: str
    cuda_version: str
    nvml_version: str
    platform: str
    os_release: str
    os_version: str
    machine: str
    distro: str
    python_version: str
    pip_packages: List[str]

    @classmethod
    def gather(cls, device):
        system = System()
        info = cls(
            device_name=device.get_name(),
            cuda_capability=device.get_cuda_compute_capability(),
            # gather system infos
            driver_version=system.get_driver_version(),
            cuda_version=system.get_cuda_driver_version(),
            nvml_version=system.get_nvml_version(),
            platform=platform.platform(),
            os_release=platform.release(),
            os_version=platform.version(),
            machine=platform.machine(),
            distro=distro.lsb_release_info(),
            python_version=sys.version,
            pip_packages=list(pip.get_installed_distributions(local_only=True))
        )
        return info

    def save(self, path):
        with open(path,"w") as f:
            json.dump(asdict(self), f)
