
import nvidia_smi as smi
from datetime import datetime
from collections import namedtuple
from pathlib import Path


Utilization = namedtuple('Utilization', ['gpu', 'mem'])


class GPU():
    def __init__(self, index):
        self.index = index
        self.handle = smi.nvmlDeviceGetHandleByIndex(index)

        """
        enable persistence mode for potential performance gain
        https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries
        """
        if not self.get_persistence_mode():
            self.set_persistence_mode(True)
        
        if not self.get_persistence_mode():
            raise Exception("Could not enable persistence mode!")
        
        try:
            self.power_limit_constraints = smi.nvmlDeviceGetPowerManagementLimitConstraints(self.handle)
        except smi.NVMLError_NotSupported as e:
            self.power_limit_constraints = None
    
    def set_persistence_mode(self, enabled):
        smi.nvmlDeviceSetPersistenceMode(self.handle, 1 if enabled else 0)
    
    def get_persistence_mode(self):
        return smi.nvmlDeviceGetPersistenceMode(self.handle) == 1

    def get_utilization(self):
        """
https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

utilization.gpu:
    Percent of time over the past sample period during which one or more kernels was executing on the GPU.
    The sample period may be between 1 second and 1/6 second depending on the product.

utilization.memory:
    Percent of time over the past sample period during which global (device) memory was being read or written.
    The sample period may be between 1 second and 1/6 second depending on the product.
        """
        res = smi.nvmlDeviceGetUtilizationRates(self.handle)
        return Utilization(res.gpu, res.memory)
    
    def get_memory_clock(self):
        return smi.nvmlDeviceGetClockInfo(self.handle,smi.NVML_CLOCK_MEM)
    
    def get_sm_clock(self):
        return smi.nvmlDeviceGetClockInfo(self.handle,smi.NVML_CLOCK_SM)

    def get_power_state(self):
        return smi.nvmlDeviceGetPowerState(self.handle)
    
    def get_power_usage(self):
        return smi.nvmlDeviceGetPowerUsage(self.handle)
    
    def get_temperature(self):
        return smi.nvmlDeviceGetTemperature(self.handle, smi.NVML_TEMPERATURE_GPU)
    
    def get_pci_tx(self):
        return smi.nvmlDeviceGetPcieThroughput(self.handle, smi.NVML_PCIE_UTIL_TX_BYTES)
    
    def get_pci_rx(self):
        return smi.nvmlDeviceGetPcieThroughput(self.handle, smi.NVML_PCIE_UTIL_RX_BYTES)

    def get_power_limit_constraints(self):
        return self.power_limit_constraints
    
    def get_current_power_limit(self):
        return smi.nvmlDeviceGetPowerManagementLimit(self.handle)

    def set_power_limit(self, limit):
        if self.power_limit_constraints is None:
            raise smi.NVMLError_NotSupported()
        if limit < self.power_limit_constraints[0]:
            raise ValueError(f"Given power limit is to low. Limit must be in range of {self.power_limit_constraints} (inclusive), but was {limit}!")
        if limit > self.power_limit_constraints[1]:
            raise ValueError(f"Given power limit is to high. Limit must be in range of {self.power_limit_constraints} (inclusive), but was {limit}!")
        
        smi.nvmlDeviceSetPowerManagementLimit(self.handle, limit)
        return self.get_current_power_limit() == limit
    
    def reset_power_limit(self):
        if self.power_limit_constraints is None:
            return True
        
        _, p_max = self.power_limit_constraints
        try:
            if p_max == self.get_current_power_limit():
                return True
            else:
                return self.set_power_limit(p_max)
        except Exception as e:
            return False
        

    def get_stats(self):
        util = self.get_utilization()
        data = {
            "gpu-index": self.index
            ,"timestamp": str(datetime.now())
            ,"util-gpu": util.gpu
            ,"util-mem": util.mem

            ,"clock-mem": self.get_memory_clock()
            ,"clock-sm": self.get_sm_clock()

            ,"power-state": self.get_power_state()

            ,"power": self.get_power_usage()
            ,"tmp": self.get_temperature()
            ,"pci-tx": self.get_pci_tx()
            ,"pci-rx": self.get_pci_rx()
        }
        return data


class SMIWrapper():
    def __init__(self):
        self.gpus = []
        self.num_devices = 0
    
    def init_handles(self):
        self.num_devices = smi.nvmlDeviceGetCount()
        for id in range(self.num_devices):
            self.gpus.append(GPU(id))

    def __enter__(self):
        smi.nvmlInit()
        self.init_handles()
        return self
    
    def __exit__(self, *argc, **kwargs):
        smi.nvmlShutdown()
    
    def get_all_stats(self, data):
        for gpu in self.gpus:
            data.append(gpu.get_stats())
    
    def get_baseline(self, length, data):
        start = time.time()
        while (time.time()-start) < length:
            self.get_all_stats(data)
        end = time.time()
        print (len(data)/(end-start))

    def set_all_powerlimit(self, limit):
        result = True
        for gpu in self.gpus:
            result &= gpu.set_power_limit(limit)
        return result

    def reset_all_powerlimit(self):
        result = True
        for gpu in self.gpus:
             result &= gpu.reset_power_limit()
        return result