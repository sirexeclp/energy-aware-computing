from dataclasses import dataclass

@dataclass
class SystemInfo:
    """Dataclass with System Information"""
    device_name: str
    cuda_capability: str
    driver_version: str
    cuda_version: str
    nvml_version: str