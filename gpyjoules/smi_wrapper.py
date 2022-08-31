import time
from enum import Enum
from typing import Tuple

import pynvml as nvml
from datetime import datetime
from collections import namedtuple
from pathlib import Path
import pint

ureg = pint.UnitRegistry()

Utilization = namedtuple("Utilization", ["gpu", "mem"])
SampleData = namedtuple("SampleData", ["timestamp", "data"])


class TemperatureThresholdType(Enum):
    SHUTDOWN = nvml.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN
    SLOWDOWN = nvml.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN


class SampleType(Enum):
    TotalPower = nvml.NVML_TOTAL_POWER_SAMPLES
    GpuUtilization = nvml.NVML_GPU_UTILIZATION_SAMPLES
    MemoryUtilization = nvml.NVML_MEMORY_UTILIZATION_SAMPLES
    GpuClock = nvml.NVML_PROCESSOR_CLK_SAMPLES
    MemoryClock = nvml.NVML_MEMORY_CLK_SAMPLES
    EncoderUtilization = nvml.NVML_ENC_UTILIZATION_SAMPLES
    DecoderUtilization = nvml.NVML_DEC_UTILIZATION_SAMPLES


class NvLinkCapability(Enum):
    P2P_SUPPORTED = 0
    SYSMEM_ACCESS = 1
    P2P_ATOMICS = 2
    SYSMEM_ATOMICS = 3
    SLI_BRIDGE = 4
    VALID = 5


class ValueType(Enum):
    DOUBLE = 0
    U_INT = 1
    U_LONG = 2
    U_LONG_LONG = 3
    LONG_LONG = 4

    def extract_value(self, union: nvml.c_nvmlSample_t):
        sample_value = union.sampleValue
        mapping = {
            ValueType.DOUBLE: sample_value.dVal,
            ValueType.U_INT: sample_value.uiVal,
            ValueType.U_LONG: sample_value.ulVal,
            ValueType.U_LONG_LONG: sample_value.ullVal,
        }
        return mapping[self]


class GPU:
    """
    https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html
    """

    def __init__(self, index, force_persistence_mode=False):
        self.index = index
        self.handle = nvml.nvmlDeviceGetHandleByIndex(index)
        self.last_sample_timestamps = {}

        if force_persistence_mode:
            self.check_persistence_mode()

        try:
            self.power_limit_constraints = (
                nvml.nvmlDeviceGetPowerManagementLimitConstraints(self.handle)
            )
        except nvml.NVMLError_NotSupported:
            self.power_limit_constraints = None

        try:
            self.supported_clocks = self._get_supported_clocks()
        except nvml.NVMLError_NotSupported:
            self.supported_clocks = None

    def check_persistence_mode(self):
        """
        Enable persistence mode for potential performance gain.
        https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries
        """
        if not self.get_persistence_mode():
            self.set_persistence_mode(True)
        if not self.get_persistence_mode():
            raise Exception("Could not enable persistence mode!")

    def set_persistence_mode(self, enabled):
        try:
            nvml.nvmlDeviceSetPersistenceMode(self.handle, 1 if enabled else 0)
        except nvml.NVMLError_NoPermission:
            return False
        return True

    def get_persistence_mode(self) -> bool:
        return nvml.nvmlDeviceGetPersistenceMode(self.handle) == 1

    def get_sample(self, sample_type: SampleType) -> SampleData:
        """
        Gets recent samples for the GPU.
        KEPLER_OR_NEWER
        Based on type, this method can be used to fetch the **power**,
        **utilization** or **clock** samples maintained in the buffer by the driver.
        CPU timestamp in microseconds.
        The advantage of using this method for samples in contrast to polling via existing methods
        is to get get higher frequency data at lower polling cost.
        https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1gb7d2a6d2a9b4584cd985765d1ff46c94
        :param sample_type: Type of sampling event
        :type sample_type: SampleType
        :return: timestamp and data in a SampleData Tuple
        :rtype: SampleData
        """
        last_timestamp = self.last_sample_timestamps.get(sample_type, 0)

        sample_value_type, buffer = nvml.nvmlDeviceGetSamples(
            self.handle, sample_type.value, last_timestamp
        )
        sample_value_type = ValueType(sample_value_type)

        self.last_sample_timestamps[sample_type] = buffer[-1].timeStamp

        data = [sample_value_type.extract_value(x) for x in buffer]
        timestamp = [i.timeStamp for i in buffer]

        return SampleData(timestamp, data)

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
        res = nvml.nvmlDeviceGetUtilizationRates(self.handle)
        return Utilization(res.gpu, res.memory)

    def get_memory_clock(self):
        """
        Retrieves the current memory clock speed for the device.
        :return: the clock speed in MHz
        :rtype: int
        """
        return (
            nvml.nvmlDeviceGetClockInfo(self.handle, nvml.NVML_CLOCK_MEM)
            * ureg.megaHertz
        )

    def get_memory_clock_limit(self):
        """
        Retrieves the current setting of the memory clock that applications will use unless an overspec situation occurs.
        Can be changed using nvmlDeviceSetApplicationsClocks.
        KEPLER_OR_NEWER
        :return: the clock speed in MHz
        :rtype: int
        """
        return (
            nvml.nvmlDeviceGetApplicationsClock(self.handle, nvml.NVML_CLOCK_MEM)
            * ureg.megaHertz
        )

    def get_max_memory_clock(self):
        """
        Retrieves the maximum memory clock speed for the device.
        *FERMI_OR_NEWER*
        :return:
        :rtype:
        """
        return nvml.nvmlDeviceGetMaxClockInfo(self.handle, nvml.NVML_CLOCK_MEM)

    def get_max_sm_clock(self):
        """
        Retrieves the maximum sm clock speed for the device.
        *FERMI_OR_NEWER*

        :return:
        :rtype:
        """
        return nvml.nvmlDeviceGetMaxClockInfo(self.handle, nvml.NVML_CLOCK_SM)

    def get_sm_clock(self):
        """
        Retrieves the current sm clock speed for the device.
        :return: the clock speed in MHz
        :rtype: int
        """
        return (
            nvml.nvmlDeviceGetClockInfo(self.handle, nvml.NVML_CLOCK_SM)
            * ureg.megaHertz
        )

    def get_sm_clock_limit(self):
        """
        Retrieves the default sm clock that GPU boots with
        or defaults to after nvmlDeviceResetApplicationsClocks call.

        :return: the clock speed in MHz
        :rtype: int
        """
        return (
            nvml.nvmlDeviceGetApplicationsClock(self.handle, nvml.NVML_CLOCK_SM)
            * ureg.megaHertz
        )

    def get_performance_state(self):
        """
        Retrieves the current performance state for the device.

        Performance state 0 -- Maximum Performance.
        ...
        Performance state 15 -- Minimum Performance.
        Performance state 32 -- Unknown performance state.

        FERMI_OR_NEWER

        :return: the performance state reading
        :rtype: int
        """
        return nvml.nvmlDeviceGetPowerState(self.handle)

    def get_power_usage(self):
        """
        Retrieves power usage for this GPU in milliwatts and its associated circuitry (e.g. memory)

        FERMI_OR_NEWER

        On Fermi and Kepler GPUs the reading is accurate to within +/- 5% of current power draw.
        It is only available if power management mode is supported. See nvmlDeviceGetPowerManagementMode.

        :return: power usage
        :rtype: int
        """
        return nvml.nvmlDeviceGetPowerUsage(self.handle) * ureg.milliwatt

    def get_temperature(self):
        """
        Retrieves the current temperature readings for the device, in degrees C.

        :return: temperature in °C
        :rtype: int
        """
        return (
            nvml.nvmlDeviceGetTemperature(self.handle, nvml.NVML_TEMPERATURE_GPU)
            * ureg.degC
        )

    def get_temperature_threshold(self, threshold_type: TemperatureThresholdType):
        """
        Retrieves the temperature threshold for the GPU with the specified threshold type in degrees C.
        KEPLER_OR_NEWER

        :return: temperature threshold in °C
        :rtype: int
        """
        return (
            nvml.nvmlDeviceGetTemperatureThreshold(self.handle, threshold_type.value)
            * ureg.degC
        )

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
        return (
            nvml.nvmlDeviceGetPcieThroughput(self.handle, nvml.NVML_PCIE_UTIL_TX_BYTES)
            * ureg.byte
            / (20 * ureg.milliseconds)
        )

    def get_pci_rx(self):
        """
        Retrieve PCIe utilization information.
        This function is querying a byte counter over a 20ms interval
        and thus is the PCIe throughput over that interval.

        MAXWELL_OR_NEWER
        This method is not supported in virtual machines running virtual GPU (vGPU).

        :return:
        :rtype:
        """
        return (
            nvml.nvmlDeviceGetPcieThroughput(self.handle, nvml.NVML_PCIE_UTIL_RX_BYTES)
            * ureg.byte
            / (20 * ureg.milliseconds)
        )

    def get_power_limit_constraints(self) -> Tuple[int, int]:
        """
        Retrieves information about possible values of power management limits on this device.
        KEPLER_OR_NEWER

        :return: tuple of (minimum power management limit,maximum power management limit) in milliwatts
        :rtype: (int, int)
        """
        return self.power_limit_constraints

    def get_current_power_limit(self):
        """
        Retrieves the power management limit associated with this device.

        FERMI_OR_NEWER

        The power limit defines the upper boundary for the card's power draw.
        If the card's total power draw reaches this limit the power management algorithm kicks in.
        This reading is only available if power management mode is supported.

        :return: power management limit in milliwatts
        :rtype: int
        """
        return nvml.nvmlDeviceGetPowerManagementLimit(self.handle)

    def set_power_limit(self, limit):
        """
        Set new power limit of this device.
        KEPLER_OR_NEWER, Requires root/admin permissions.

        Note:
        Limit is not persistent across reboots or driver unloads.
        Enable persistent mode to prevent driver from unloading when no application is using the device.

        :param limit: Power management limit in milliwatts to set
        :type limit: int
        :return: True if limit has been set, False otherwise.
        :rtype: bool
        :raises smi.NVML_ERROR_NOT_SUPPORTED: if the device does not support this feature
        """

        if self.power_limit_constraints is None:
            raise nvml.NVML_ERROR_NOT_SUPPORTED()
        if limit < self.power_limit_constraints[0]:
            raise ValueError(
                f"Given power limit is to low. Limit must be in range of {self.power_limit_constraints} (inclusive), "
                f"but was {limit}!"
            )
        if limit > self.power_limit_constraints[1]:
            raise ValueError(
                f"Given power limit is to high. Limit must be in range of {self.power_limit_constraints} (inclusive), "
                f"but was {limit}!"
            )

        nvml.nvmlDeviceSetPowerManagementLimit(self.handle, limit)
        return self.get_current_power_limit() == limit

    def reset_power_limit(self):
        """
        Resets power-limit to maximum value.
        :return:
        :rtype:
        """
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
        mem_clocks = nvml.nvmlDeviceGetSupportedMemoryClocks(self.handle)
        for mem_c in mem_clocks:
            graphic_clocks = nvml.nvmlDeviceGetSupportedGraphicsClocks(
                self.handle, mem_c
            )
            supported_clocks[mem_c] = graphic_clocks
        return supported_clocks

    def get_default_memory_clock(self) -> int:
        """
        Retrieves the default applications clock that GPU boots with
        or defaults to after nvmlDeviceResetApplicationsClocks call.

        KEPLER_OR_NEWER

        :return: default clock in MHz
        :rtype: int
        """
        return nvml.nvmlDeviceGetDefaultApplicationsClock(
            self.handle, nvml.NVML_CLOCK_MEM
        )

    def get_default_sm_clock(self) -> int:
        """
        Retrieves the default applications clock that GPU boots with
        or defaults to after nvmlDeviceResetApplicationsClocks call.

        KEPLER_OR_NEWER

        :return: default clock in MHz
        :rtype: int
        """
        return nvml.nvmlDeviceGetDefaultApplicationsClock(
            self.handle, nvml.NVML_CLOCK_SM
        )

    def reset_clocks(self):
        """
        Resets the application clock to the default value
        This is the applications clock that will be used after system reboot or driver reload.
        Default value is constant, but the current value an be changed using nvmlDeviceSetApplicationsClocks.
        On Pascal and newer hardware, if clocks were previously locked with nvmlDeviceSetApplicationsClocks,
         this call will unlock clocks.
        This returns clocks their default behavior of automatically boosting above base clocks as thermal limits allow.

        :return: True if reset was successful, False otherwise
        :rtype: bool
        """
        nvml.nvmlDeviceResetApplicationsClocks(self.handle)
        return (
            self.get_sm_clock_limit() == self.get_default_sm_clock()
            and self.get_memory_clock_limit() == self.get_default_memory_clock()
        )

    def set_clocks(self, memory_clock, graphic_clock):
        """
        Set clocks that applications will lock to.
        Sets the clocks that compute and graphics applications will be running at.
        e.g. CUDA driver requests these clocks during context creation
        which means this property defines clocks at which CUDA applications will be running
        unless some overspec event occurs (e.g. over power, over thermal or external HW brake).

        Can be used as a setting to request constant performance.

        On Pascal and newer hardware, this will automatically disable automatic boosting of clocks.

        On K80 and newer Kepler and Maxwell GPUs, users desiring fixed performance should also call nvmlDeviceSetAutoBoostedClocksEnabled to prevent clocks from automatically boosting above the clock value being set.

        KEPLER_OR_NEWER_GF, Requires root/admin permissions.

        See nvmlDeviceGetSupportedMemoryClocks and nvmlDeviceGetSupportedGraphicsClocks
        for details on how to list available clocks combinations.

        After system reboot or driver reload applications clocks go back to their default value.
        See nvmlDeviceResetApplicationsClocks.

        :param memory_clock: Requested memory clock in MHz
        :type memory_clock: int
        :param graphic_clock: Requested graphics clock in MHz
        :type graphic_clock: int
        :return: True if clocks, were set successfully, False otherwise
        :rtype: bool
        """
        supported_graphic_clocks = self.supported_clocks.get(memory_clock, None)
        if (
            supported_graphic_clocks is None
            or graphic_clock not in supported_graphic_clocks
        ):
            raise ValueError(
                f"Given Clock configuration is not supported! mem:{memory_clock} gpu: {graphic_clock}"
            )

        nvml.nvmlDeviceSetApplicationsClocks(self.handle, memory_clock, graphic_clock)
        return (
            memory_clock == self.get_memory_clock_limit()
            and graphic_clock == self.get_sm_clock_limit()
        )

    def get_nv_counter(self, link: int, capability: NvLinkCapability):
        _nvmlNvLinkCapability_t = nvml.c_uint
        cap_result = nvml.c_uint()
        fn = nvml._nvmlGetFunctionPointer("nvmlDeviceGetNvLinkCapability")
        ret = fn(
            self.handle,
            nvml.c_uint(link),
            _nvmlNvLinkCapability_t(capability.value),
            nvml.byref(cap_result),
        )
        nvml._nvmlCheckReturn(ret)
        return cap_result.value

    # def get_stats(self):
    #     util = self.get_utilization()
    #     data = {
    #         "gpu-index": self.index
    #         , "timestamp": str(datetime.now())
    #         # , "util-gpu": self.get_sample(SampleType.GpuUtilization)
    #         # , "util-mem": self.get_sample(SampleType.MemoryUtilization)
    #         #
    #         # , "clock-mem": self.get_sample(SampleType.MemoryClock)
    #         # , "clock-gpu": self.get_sample(SampleType.GpuClock)
    #         #
    #         # # , "power-state": self.get_power_state()
    #         #
    #         # , "power": self.get_sample(SampleType.TotalPower)
    #         , "power": self.get_power_usage()
    #         , "tmp": self.get_temperature()
    #         , "pci-tx": self.get_pci_tx()
    #         , "pci-rx": self.get_pci_rx()
    #     }
    #     return data


class SMIWrapper:
    def __init__(self, selected_devices=None):
        self.gpus = []
        self.selected_devices = selected_devices

    def open(self):
        self.__enter__()

    def close(self):
        self.__exit__()

    def __enter__(self):
        nvml.nvmlInit()
        if self.selected_devices is None:
            self.selected_devices = range(nvml.nvmlDeviceGetCount())
        self.gpus = [GPU(i) for i in self.selected_devices]

        return self

    def __exit__(self, *argc, **kwargs):
        nvml.nvmlShutdown()
        self.gpus = []
        self.num_devices = 0
