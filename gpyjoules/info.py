import pynvml3
import numpy as np

lib = pynvml3.NVMLLib()
num_total_runs = 10
gpu_index = 0
with lib:
    dev = lib.device.from_index(gpu_index)
    name = dev.get_name()
    memory_clocks = dev.get_supported_memory_clocks()

    pl_constraints = dev.get_power_management_limit_constraints()
    pl_constraints = np.array(pl_constraints) / 1_000  # mW 2 W

    print("Power Limit Constraints: ", pl_constraints)

    print("Memory Clocks: ", memory_clocks, "for", name)
    run_per_mem = int(np.ceil(num_total_runs / len(memory_clocks)))
    for mem in memory_clocks:
        gpu_clocks = np.array(dev.get_supported_graphics_clocks(mem))
        selected_idx = np.linspace(0, len(gpu_clocks) - 1, run_per_mem, dtype=np.int)
        selected = gpu_clocks[selected_idx]

        print("\t", gpu_clocks)
        print("selected for", mem, "\t", selected)
        tuples = [f"- [ {mem}, {x} ]" for x in selected]
        print("\n".join(tuples))
