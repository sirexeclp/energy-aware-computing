import time
from enum import Enum

import nvidia_smi as smi
from datetime import datetime
from collections import namedtuple
from pathlib import Path

Utilization = namedtuple('Utilization', ['gpu', 'mem'])


class SampleType(Enum):
    TotalPower = smi.NVML_TOTAL_POWER_SAMPLES
    GpuUtilization = smi.NVML_GPU_UTILIZATION_SAMPLES
    MemoryUtilization = smi.NVML_MEMORY_UTILIZATION_SAMPLES
    GpuClock = smi.NVML_PROCESSOR_CLK_SAMPLES
    MemoryClock = smi.NVML_MEMORY_CLK_SAMPLES
    EncoderUtilization = smi.NVML_ENC_UTILIZATION_SAMPLES
    DecoderUtilization = smi.NVML_DEC_UTILIZATION_SAMPLES


def resolve_type(union: smi.c_nvmlSample_t, value_type):
    sample_value = union.sampleValue
    mapping = {
        ValueType.double: sample_value.dVal,
        ValueType.u_int: sample_value.uiVal,
        ValueType.u_long: sample_value.ulVal,
        ValueType.u_long_long: sample_value.ullVal
    }
    return mapping[value_type]


class ValueType(Enum):
    double = 0
    u_int = 1
    u_long = 2
    u_long_long = 3
    long_long = 4


class GPU:
    def __init__(self, index):
        self.index = index
        self.handle = smi.nvmlDeviceGetHandleByIndex(index)
        self.last_sample_timestamps = {}

        """
        enable persistence mode for potential performance gain
        https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries
        """
        if not self.get_persistence_mode():
            self.set_persistence_mode(True)

        if not self.get_persistence_mode():
            pass
            # raise Exception("Could not enable persistence mode!")

        try:
            self.power_limit_constraints = smi.nvmlDeviceGetPowerManagementLimitConstraints(self.handle)
        except smi.NVML_ERROR_NOT_SUPPORTED:
            self.power_limit_constraints = None

        try:
            self.supported_clocks = self._get_supported_clocks()
        except smi.NVML_ERROR_NOT_SUPPORTED:
            self.supported_clocks = None

    def set_persistence_mode(self, enabled):
        try:
            smi.nvmlDeviceSetPersistenceMode(self.handle, 1 if enabled else 0)
        except smi.NVML_ERROR_NO_PERMISSION:
            return False
        return True

    def get_persistence_mode(self):
        return smi.nvmlDeviceGetPersistenceMode(self.handle) == 1

    def get_sample(self, sample_type):
        last_timestamp = self.last_sample_timestamps.get(sample_type, 0)

        sample_value_type, buffer = smi.nvmlDeviceGetSamples(self.handle, sample_type.value, last_timestamp)
        sample_value_type = ValueType(sample_value_type)

        self.last_sample_timestamps[sample_type] = buffer[-1].timeStamp

        data = [resolve_type(x, sample_value_type) for x in buffer]
        timestamp = [i.timeStamp for i in buffer]

        return timestamp, data

    # def get_all_samples(self):
    #     exclude = [SampleType.DecoderUtilization, SampleType.EncoderUtilization]
    #     sample_types = set(SampleType) - set(exclude)
    #     for sample_type in sample_types:
    #         self.get_sample(sample_type)

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
        return smi.nvmlDeviceGetClockInfo(self.handle, smi.NVML_CLOCK_MEM)

    def get_memory_clock_limit(self):
        return smi.nvmlDeviceGetApplicationsClock(self.handle, smi.NVML_CLOCK_MEM)

    def get_sm_clock(self):
        return smi.nvmlDeviceGetClockInfo(self.handle, smi.NVML_CLOCK_SM)

    def get_sm_clock_limit(self):
        return smi.nvmlDeviceGetApplicationsClock(self.handle, smi.NVML_CLOCK_SM)

    def get_power_state(self):
        return smi.nvmlDeviceGetPowerState(self.handle)

    def get_power_usage(self):
        return smi.nvmlDeviceGetPowerUsage(self.handle)

    def get_temperature(self):
        """:returns int"""
        return smi.nvmlDeviceGetTemperature(self.handle, smi.NVML_TEMPERATURE_GPU)

    def get_pci_tx(self):
        """
        Retrieve PCIe utilization information.
        This function is querying a byte counter over a 20ms interval
        and thus is the PCIe throughput over that interval.

        MAXWELL_OR_NEWER
        This method is not supported in virtual machines running virtual GPU (vGPU).

        :raises smi.NVML_ERROR_UNINITIALIZED: if the library has not been successfully initialized
        :raises smi.NVML_ERROR_INVALID_ARGUMENT: if device or counter is invalid, or value is NULL
        :raises smi.NVML_ERROR_NOT_SUPPORTED: if the device does not support this feature
        :raises smi.NVML_ERROR_GPU_IS_LOST:  if the target GPU has fallen off the bus or is otherwise inaccessible
        :raises smi.NVML_ERROR_UNKNOWN: on any unexpected error

        :returns int
        """
        return smi.nvmlDeviceGetPcieThroughput(self.handle, smi.NVML_PCIE_UTIL_TX_BYTES)

    def get_pci_rx(self):
        return smi.nvmlDeviceGetPcieThroughput(self.handle, smi.NVML_PCIE_UTIL_RX_BYTES)

    def get_power_limit_constraints(self):
        return self.power_limit_constraints

    def get_current_power_limit(self):
        return smi.nvmlDeviceGetPowerManagementLimit(self.handle)

    def set_power_limit(self, limit):
        if self.power_limit_constraints is None:
            raise smi.NVML_ERROR_NOT_SUPPORTED()
        if limit < self.power_limit_constraints[0]:
            raise ValueError(
                f"Given power limit is to low. Limit must be in range of {self.power_limit_constraints} (inclusive), "
                f"but was {limit}!")
        if limit > self.power_limit_constraints[1]:
            raise ValueError(
                f"Given power limit is to high. Limit must be in range of {self.power_limit_constraints} (inclusive), "
                f"but was {limit}!")

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

    def _get_supported_clocks(self):
        """
        Returns a dictionary of valid memory: graphics clock combinations.
        """
        supported_clocks = {}
        mem_clocks = smi.nvmlDeviceGetSupportedMemoryClocks(self.handle)
        for mem_c in mem_clocks:
            graphic_clocks = smi.nvmlDeviceGetSupportedGraphicsClocks(self.handle, mem_c)
            supported_clocks[mem_c] = graphic_clocks
        return supported_clocks

    def reset_clocks(self):
        smi.nvmlDeviceResetApplicationsClocks(self.handle)
        return True

    def set_clocks(self, memory_clock, graphic_clock):
        supported_graphic_clocks = self.supported_clocks.get(memory_clock, None)
        if supported_graphic_clocks is None or graphic_clock not in supported_graphic_clocks:
            raise ValueError(f"Given Clock configuration is not supported! mem:{memory_clock} gpu: {graphic_clock}")

        smi.nvmlDeviceSetApplicationsClocks(self.handle, memory_clock, graphic_clock)
        return memory_clock == self.get_memory_clock_limit() and graphic_clock == self.get_sm_clock_limit()

    def get_stats(self):
        util = self.get_utilization()
        data = {
            "gpu-index": self.index
            , "timestamp": str(datetime.now())
            , "util-gpu": self.get_sample(SampleType.GpuUtilization)
            , "util-mem": self.get_sample(SampleType.MemoryUtilization)

            , "clock-mem": self.get_sample(SampleType.MemoryClock)
            , "clock-gpu": self.get_sample(SampleType.GpuClock)

            # , "power-state": self.get_power_state()

            , "power": self.get_sample(SampleType.TotalPower)
            , "tmp": self.get_temperature()
            , "pci-tx": self.get_pci_tx()
            , "pci-rx": self.get_pci_rx()
        }
        return data


class SMIWrapper:
    def __init__(self, selected_devices=None):
        self.gpus = []
        self.selected_devices = selected_devices

    def open(self):
        self.__enter__()

    def close(self):
        self.__exit__()

    def __enter__(self):
        smi.nvmlInit()
        if self.selected_devices is None:
            self.selected_devices = range(smi.nvmlDeviceGetCount())
        self.gpus = [GPU(i) for i in self.selected_devices]

        return self

    def __exit__(self, *argc, **kwargs):
        smi.nvmlShutdown()
        self.gpus = []
        self.num_devices = 0

    def get_all_stats(self, data):
        for gpu in self.gpus:
            data.append(gpu.get_stats())

    def get_baseline(self, length, data):
        start = time.time()
        while (time.time() - start) < length:
            self.get_all_stats(data)
        end = time.time()
        print(len(data) / (end - start))

    def set_all_power_limit(self, limit):
        result = True
        for gpu in self.gpus:
            result &= gpu.set_power_limit(limit)
        return result

    def reset_all_power_limit(self):
        result = True
        for gpu in self.gpus:
            result &= gpu.reset_power_limit()
        return result

    def set_all_clocks(self, memory, graphic):
        result = True
        for gpu in self.gpus:
            result &= gpu.set_clocks(memory, graphic)
        return result

    def reset_all_clocks(self):
        result = True
        for gpu in self.gpus:
            result &= gpu.reset_clocks()
        return result

    @staticmethod
    def set_clocks(clocks):
        with SMIWrapper() as sw:
            if clocks is not None:
                success = sw.set_all_clocks(*clocks)
                print(f"[{'Success' if success else 'Failed'}] set clocks to {clocks}")
            else:
                success = sw.reset_all_clocks()
                print(f"[{'Success' if success else 'Failed'}] reset clocks to default")

            assert success, f"Failed setting/resetting clocks ({clocks}), aborting experiment!"

    @staticmethod
    def set_power_limit(power_limit):
        with SMIWrapper() as sw:
            if power_limit is not None:
                success = sw.set_all_power_limit(power_limit * 1000)  # convert to mW
                print(f"[{'Success' if success else 'Failed'}] set power-limit to {power_limit}")
            else:
                success = sw.reset_all_power_limit()
                print(f"[{'Success' if success else 'Failed'}] reset power-limit to default (max)")

            assert success, "Failed setting/resetting power-limit, aborting experiment!"
