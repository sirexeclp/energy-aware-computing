import abc
import atexit
import multiprocessing
import time
import os
import warnings
# with warnings.catch_warnings():
from typing import Union, Optional
from multiprocessing import Event, Queue

from pynvml3 import Device, ClockType, ClockId
from pynvml3 import NVMLLib
from pynvml3 import TemperatureSensors, PcieUtilCounter, SamplingType
from pynvml3.errors import NVMLErrorNotFound, NVMLErrorNotSupported
from util import *

warnings.filterwarnings('ignore')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow
import tensorflow.keras as keras


class TimestampLogger:

    def __init__(self, log_path: Union[str, Path]):
        self.timestamp_log = []
        self.log_path = log_path
        self.columns = ["timestamp", "event", "data"]

    def log_event(self, name: str, index: int = 0) -> None:
        # print(f"[Info] {name} {index}")
        tmp = {
            "timestamp": str(datetime.now()),
            "event": name,
            "data": index
        }
        self.timestamp_log.append(tmp)

    def save(self):
        df = pd.DataFrame(self.timestamp_log, columns=self.columns)
        df.to_csv(self.log_path, index=False)

    def get_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.timestamp_log, columns=self.columns)


class EnergyCallback(keras.callbacks.Callback):
    """Custom Keras Callback to log events, for energy measurements."""

    def __init__(self, enable_energy_prediction: bool, num_epochs: int,
                 timestamp_logger: TimestampLogger, visible_devices: int,
                 data_event: Event, data_queue: Queue):
        super(EnergyCallback, self).__init__()
        self.timestamp_log = timestamp_logger
        self.enable_energy_prediction = enable_energy_prediction
        self.num_epochs = num_epochs

        self.total_batch = 0
        self.visible_devices = visible_devices
        self.last_time = time.time()
        self.summary_writer = tensorflow.summary.create_file_writer("/tmp/energy-board/energy")
        self.data_event = data_event
        self.data_queue = data_queue

    def on_train_batch_end(self, batch: int, logs=None) -> None:
        self.log_event("batch_end", batch)
        # self.total_batch += 1
        # if (self.total_batch % 100) == 0:
        #     with NVMLLib() as lib:
        #         device = Device.from_index(self.visible_devices)
        #         device.set_power_management_limit(device.get_power_management_limit()-50_000)
        #         t = time() - self.last_time
        #         print(f"Testing PL: {device.get_power_management_limit()}W, Time: {t}")
        #         self.last_time = time()

    def on_train_batch_begin(self, batch: int, logs=None) -> None:
        self.log_event("batch_begin", batch)

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        self.log_event("epoch_begin", epoch)

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        self.log_event("epoch_end", epoch)

        if epoch > 0 and self.enable_energy_prediction:
            self.predict_energy(epoch)

    def on_train_begin(self, logs=None) -> None:
        self.log_event("train_begin")
        print("Timestamp Logger: Train Begin")

    def on_train_end(self, logs=None) -> None:
        self.log_event("train_end")

    def log_event(self, name: str, index: int = 0):
        self.timestamp_log.log_event(name=name, index=index)

    def get_gpu_data(self) -> pd.DataFrame:
        self.data_event.set()
        while self.data_queue.empty():
            time.sleep(0.1)
        gpu_data = self.data_queue.get()
        gpu_data = pd.DataFrame(gpu_data)
        return gpu_data

    def predict_energy(self, epoch: int) -> None:
        gpu_data = self.get_gpu_data()
        timestamps = self.timestamp_log.get_df()
        power_data = PowerData(gpu_data, timestamps)
        pred = predict_energy_live(power_data, [0], self.num_epochs, epoch)
        actual = calculate_total_energy(power_data, [0])
        with self.summary_writer.as_default():
            tensorflow.summary.scalar("energy", data=actual, step=epoch)
            tensorflow.summary.scalar("energy-predicted", data=pred, step=epoch)
        self.summary_writer.flush()
        print(f"\nConsumed Energy: {actual / 1_000:.3f}/{pred / 1_000:.3f}kJ")


def get_log_path(new_path: Union[str, Path]) -> Path:
    data_root = Path(new_path)
    timestamp_log_path = data_root / "timestamps.csv"
    data_root.mkdir(parents=True, exist_ok=True)
    return timestamp_log_path


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
    def __init__(self, device_id: int, interval: float, path: Union[str, Path], args=None, kwargs=None):
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


def patch(data_root: str, enable_energy: bool, visible_devices: int):
    timestamp_log_path = get_log_path(data_root)
    logger = TimestampLogger(timestamp_log_path)
    # timestamp_log = open(timestamp_log_path , "w", buffering=1)
    # timestamp_log.write(f"timestamp,event,data\n")

    sampling_types = [
        SamplingType.GPU_UTILIZATION_SAMPLES,
        SamplingType.MEMORY_UTILIZATION_SAMPLES,
        # SamplingType.MEMORY_CLK_SAMPLES,
        # SamplingType.PROCESSOR_CLK_SAMPLES,
        SamplingType.TOTAL_POWER_SAMPLES
    ]

    sampling_manager = CollectorManager(data_root, visible_devices)

    data_event = Event()
    data_queue = Queue()

    sampling_manager.add(SlowCollector(visible_devices,
                                       interval=0.15,
                                       path=data_root,
                                       args=(data_event, data_queue)))

    sampling_manager.add_by_sampling_type(sampling_types, interval=1.5)

    sampling_manager.start()

    @atexit.register
    def on_exit():
        logger.log_event("experiment_end")
        logger.save()
        sampling_manager.stop()

    def get_patched_fit(original_function):
        def patched_fit(*args, **kwargs):
            num_epochs = kwargs.get("epochs")
            callback = EnergyCallback(enable_energy, num_epochs
                                      , logger, visible_devices
                                      , data_event, data_queue)
            kwargs.setdefault("callbacks", list()).append(callback)
            if not enable_energy:
                kwargs["verbose"] = 2
            return original_function(*args, **kwargs)

        return patched_fit

    model = tensorflow.keras.models.Model
    model.fit = get_patched_fit(model.fit)
    model.fit_generator = get_patched_fit(model.fit_generator)

    logger.log_event("experiment_begin")
