import pynvml3
import numpy as np

lib = pynvml3.NVMLLib()
num_total_runs = 10
gpu_index = 0
with lib:
    dev = lib.device.from_index(gpu_index)
    name = dev.get_name()
    memory_clocks = dev.get_supported_memory_clocks()

    print("Power Limit Constraints: ", dev.get_power_management_limit_constraints())

    print(memory_clocks, "for", name)
    run_per_mem = np.ceil(num_total_runs / len(memory_clocks))
    for mem in memory_clocks:
        gpu_clocks = np.array(dev.get_supported_graphics_clocks(mem))
        selected_idx = np.linspace(0, len(gpu_clocks), run_per_mem)
        selected = gpu_clocks[selected_idx]

        print("\t", gpu_clocks)
        print("selected for", mem, "\t", selected)