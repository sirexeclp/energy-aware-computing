import json
import platform
import sys
from dataclasses import dataclass, asdict
from typing import List, Tuple, NamedTuple

import distro
import pkg_resources


class SystemInfo(NamedTuple):
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
    pip_packages: List[Tuple[str, str]]
    # supported_sampling_types: List[str]

    @classmethod
    def gather(cls, device):
        system = device.lib.system
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
            pip_packages=[(x.project_name, x.version) for x in pkg_resources.working_set]
        )
        return info

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self._asdict(), f)
