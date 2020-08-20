import json
import platform
import sys
from typing import List, Tuple, NamedTuple

import distro
import pkg_resources
from pynvml3 import Device, ClockType, ClockId


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
    effective_power_limit: int
    applications_clock_target_sm: int
    max_boost_clock_sm: int

    applications_clock_target_mem: int
    max_boost_clock_mem: int

    @classmethod
    def gather(cls, device: Device):
        system = device.lib.system
        info = cls(
            device_name=device.get_name(),
            cuda_capability=device.get_cuda_compute_capability(),
            effective_power_limit=device.get_enforced_power_limit(),
            applications_clock_target_sm=device.get_clock(ClockType.SM, ClockId.APP_CLOCK_TARGET),
            max_boost_clock_sm=device.get_clock(ClockType.SM, ClockId.CUSTOMER_BOOST_MAX),
            applications_clock_target_mem=device.get_clock(ClockType.MEM, ClockId.APP_CLOCK_TARGET),
            max_boost_clock_mem=device.get_clock(ClockType.MEM, ClockId.CUSTOMER_BOOST_MAX),
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
