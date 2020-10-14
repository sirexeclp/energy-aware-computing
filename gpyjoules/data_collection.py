import abc
import atexit
import multiprocessing
import time
from datetime import datetime
from multiprocessing.queues import Queue
from multiprocessing import Event
from pathlib import Path
from typing import Optional, Union, List

import pandas
from pynvml3 import Device, NVMLLib, SamplingType, NVMLErrorNotFound, ClockType, ClockId, TemperatureSensors, \
    PcieUtilCounter, NVMLErrorNotSupported


class ProcessTimer(abc.ABC):

    def __init__(self, interval: float, args=None, kwargs=None):
        self.interval = interval
        self.args = () if args is None else args
        self.kwargs = {} if kwargs is None else kwargs
        self._stop = multiprocessing.Event()
        self.last_time = None
        self.process = multiprocessing.Process(target=self._run)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self) -> None:
        self.last_time = time.time()
        self._stop.clear()
        self.process.start()

    def stop(self) -> None:
        self._stop.set()

    @abc.abstractmethod
    def _on_start(self) -> None:
        pass

    @abc.abstractmethod
    def _on_stop(self) -> None:
        pass

    @abc.abstractmethod
    def _on_tick(self, args, kwargs) -> None:
        pass

    def _run(self) -> None:
        self._on_start()
        while not self._stop.is_set():
            iter_start = time.time()
            self._on_tick(self.args, self.kwargs)
            now = time.time()
            sleep_time = self.interval - (now - self.last_time)
            if (now - iter_start) > self.interval:
                print(f"WARNING: can't keep up {(now - iter_start)}s")
                # TODO: log that we are running late
            self.last_time = now
            time.sleep(max(sleep_time, 0))
        self._on_stop()


class Collector(ProcessTimer):
    def __init__(self, device_id: Optional[int], interval: float, path: Union[str, Path], args=None, kwargs=None):
        super(Collector, self).__init__(interval, args, kwargs)
        self.data = []
        self.device_id = device_id
        self.lib = None
        self.device: Optional[Device] = None
        self.path = Path(path)

    def _on_start(self) -> None:
        self.lib = NVMLLib()
        self.lib.open()
        self.device = self.lib.device.from_index(self.device_id)

    def _on_stop(self) -> None:
        self.device = None
        self.lib.close()
        self._save()

    @abc.abstractmethod
    def _get_save_path(self) -> Path:
        pass

    @abc.abstractmethod
    def test(self, device: Device) -> bool:
        pass

    def _save(self) -> None:
        # print(f"[saving] {self.__class__.__name__}")
        # print(self.get_len())
        path = self._get_save_path()
        df = pd.DataFrame(self.data)
        df.to_csv(path)

    def get_len(self):
        return len(self.data)


class SampleCollector(Collector):

    def __init__(self, sample_type: SamplingType, device_id: int, interval: float, path: Union[str, Path]):
        super(SampleCollector, self).__init__(device_id, interval, path)
        self.sample_type = sample_type
        self.last_sample_time = 0

    def _on_tick(self, args, kwargs) -> None:
        samples = self.device.try_get_samples(self.sample_type, self.last_sample_time)
        if samples:
            self.data.extend(samples)
            self.last_sample_time = samples[-1].timestamp

    def _get_save_path(self) -> Path:
        return self.path / (self.sample_type.get_filename() + ".csv")

    def test(self, device: Device) -> bool:
        try:
            # first call sometimes works but then the second will fail; idk why.
            self.device = device
            self.device.get_samples(self.sample_type, self.last_sample_time)
            time.sleep(self.interval)
            self.device.get_samples(self.sample_type, self.last_sample_time)
            print(f"{self.sample_type}: OK")
            self.device = None
            return True
        except NVMLErrorNotFound:
            print(f"{self.sample_type} not supported on this device!")
            self.device = None
            return False


class SlowCollector(Collector):

    def __init__(self, device_id: int, interval: float, path: str, args=None, kwargs=None):
        super(SlowCollector, self).__init__(device_id, interval, path, args, kwargs)
        self.data_functions = {
            "timestamp": lambda: str(datetime.now())
            , "util": lambda: self.device.get_utilization_rates()
            , "clock-mem": lambda: self.device.get_clock(ClockType.MEM, ClockId.CURRENT)
            , "clock-gpu": lambda: self.device.get_clock(ClockType.SM, ClockId.CURRENT)
            , "app-clock-mem": lambda: self.device.get_applications_clock(ClockType.MEM)
            , "app-clock-gpu": lambda: self.device.get_applications_clock(ClockType.SM)
            , "enforced-power-limit": lambda: self.device.get_enforced_power_limit()
            , "total-energy": lambda: self.device.get_total_energy_consumption()
            # int representation to save on storage size
            , "power-state": lambda: self.device.get_power_state().value
            , "power": lambda: self.device.get_power_usage()
            , "tmp": lambda: self.device.get_temperature(TemperatureSensors.TEMPERATURE_GPU)
            , "pci-tx": lambda: self.device.get_pcie_throughput(PcieUtilCounter.TX_BYTES)
            , "pci-rx": lambda: self.device.get_pcie_throughput(PcieUtilCounter.RX_BYTES)
        }

    def _get_save_path(self) -> Path:
        return self.path / "gpu-power.csv"

    def _on_tick(self, args, kwargs) -> None:
        data_event, data_queue = args
        data = {key: function() for key, function in self.data_functions.items()}
        if "util" in data:
            data["util-gpu"] = data["util"].gpu
            data["util-mem"] = data["util"].memory
            del data["util"]

        self.data.append(data)

        if data_event.is_set():
            data_queue.put(self.data)
            data_event.clear()

    @staticmethod
    def _is_supported(key: str, function: callable) -> bool:
        try:
            function()
            return True
        except NVMLErrorNotSupported:
            print(f"W: {key} is not supported on this system!")
            return False

    def test(self, device: Device) -> bool:
        self.device = device
        self.data_functions = {key: value for key, value in
                               self.data_functions.items() if self._is_supported(key, value)}
        self.device = None
        return bool(self.data_functions)


class ExternalCollector(Collector):

    def __init__(self, interval, path ):
        super().__init__(None, interval=interval, path=path)
        self.serial_devices = [
            "/dev/ttyACM0",
            "/dev/ttyACM1"
        ]
        self.handles = []

    def test(self, device: Optional[Device]) -> bool:
        try:
            for dev in self.serial_devices:
                self.handles.append(pynpoint.MCP(dev))
            return True
        except Exception:
            return False

    def _on_start(self):
        pass

    def _on_stop(self):
        self._save()

    def _on_tick(self, args, kwargs) -> None:
        tmp = {"timestamp": str(datetime.now())}
        for index, h in enumerate(self.handles):
            c0, c1 = h.get_power()
            tmp[f"d{index}c0"] = c0
            tmp[f"d{index}c1"] = c1
        self.data.append(tmp)

    def _get_save_path(self) -> Path:
        return self.path / "power-external.csv"


class CollectorManager:
    def __init__(self, data_path: Union[str, Path], device_id: int):
        self.collectors: List[Collector] = []
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.device_id = device_id

    def add(self, collector: Union[Collector, List[Collector]]) -> None:
        if not isinstance(collector, list):
            collector = [collector]

        with NVMLLib() as lib:
            device = lib.device.from_index(self.device_id)
            for col in collector:
                if col.test(device):
                    self.collectors.append(col)

    def add_by_sampling_type(self, sampling_type: Union[SamplingType,
                                                        List[SamplingType]]
                             , interval):
        if isinstance(sampling_type, SamplingType):
            sampling_type = [sampling_type]

        collectors = [SampleCollector(st, self.device_id, interval, self.data_path) for st in sampling_type]
        self.add(collectors)

    def start(self):
        for col in self.collectors:
            col.start()

    def stop(self):
        for col in self.collectors:
            col.stop()


def start_collecting(data_root: str, visible_devices: int, data_event: Event, data_queue: Queue):

    sampling_types = [
        SamplingType.GPU_UTILIZATION_SAMPLES,
        SamplingType.MEMORY_UTILIZATION_SAMPLES,
        # SamplingType.MEMORY_CLK_SAMPLES,
        # SamplingType.PROCESSOR_CLK_SAMPLES,
        SamplingType.TOTAL_POWER_SAMPLES
    ]

    sampling_manager = CollectorManager(data_root, visible_devices)
    sampling_manager.add(SlowCollector(visible_devices,
                                       interval=0.15,
                                       path=data_root,
                                       args=(data_event, data_queue)))

    external_collector = ExternalCollector(interval=0.02, path=data_root)
    sampling_manager.add(external_collector)

    sampling_manager.add_by_sampling_type(sampling_types, interval=1.5)

    @atexit.register
    def on_exit():
        sampling_manager.stop()

    sampling_manager.start()