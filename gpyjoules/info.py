"""Provide info about available power- and clock-limits."""
import pynvml3
import numpy as np


def print_info4gpu(gpu: pynvml3.Device):
    """Print power and clock limits for the given GPU-Device."""
    print("GPU:", gpu.get_name())

    pl_min, pl_max = (
        np.array(gpu.get_power_management_limit_constraints()) / 1_000
    )  # mW 2 W
    print("Power Limits:")
    print("\tmin:", pl_min)
    print("\tmax:", pl_max)

    memory_clocks = gpu.get_supported_memory_clocks()
    print("Clocks:")
    for mem_clock in memory_clocks:
        gpu_clocks = np.array(gpu.get_supported_graphics_clocks(mem_clock))
        print("\tmemory:", mem_clock)
        print("\t\tgpu:", gpu_clocks)


def print_info():
    """Print available power and clock limits for all GPUs."""
    with pynvml3.NVMLLib() as nvml:
        for gpu in nvml.device:
            print_info4gpu(gpu)
            print()


if __name__ == "__main__":
    print_info()
