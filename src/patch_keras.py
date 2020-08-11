import abc
import multiprocessing
import warnings

# with warnings.catch_warnings():
from typing import Union

from pynvml3 import Device, ClockType, ClockId
from pynvml3 import TemperatureSensors, PcieUtilCounter, SamplingType
from pynvml3 import NVMLLib
from pynvml3.errors import NVMLErrorNotFound

warnings.filterwarnings('ignore')

import tensorflow
import tensorflow.keras as keras
import atexit
from multiprocessing import Process, Event, Queue
import time
from util import *


class EnergyCallback(keras.callbacks.Callback):
    """Costum Keras Callback to log events, for energy measurements."""

    def __init__(self, enable_energy_prediction,
                 num_epochs, log_path, visible_devices, data_event, data_queue):
        super(EnergyCallback, self).__init__()
        self.timestamp_log = []
        self.enable_energy_prediction = enable_energy_prediction
        self.num_epochs = num_epochs
        self.log_path = log_path
        self.columns = ["timestamp", "event", "data"]
        self.total_batch = 0
        self.visible_devices = visible_devices
        self.last_time = time.time()
        self.summary_writer = tensorflow.summary.create_file_writer("/tmp/energy-board/energy")
        self.data_event = data_event
        self.data_queue = data_queue

    def on_train_batch_end(self, batch, logs=None):
        pass
        # self.total_batch += 1
        # if (self.total_batch % 100) == 0:
        #     with NVMLLib() as lib:
        #         device = Device.from_index(self.visible_devices)
        #         device.set_power_management_limit(device.get_power_management_limit()-50_000)
        #         t = time() - self.last_time
        #         print(f"Testing PL: {device.get_power_management_limit()}W, Time: {t}")
        #         self.last_time = time()

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        self.log_event("epoch_begin", epoch)

    def on_epoch_end(self, epoch, logs=None):
        self.log_event("epoch_end", epoch)

        if epoch > 0 and self.enable_energy_prediction:
            self.predict_energy(epoch)

    def on_train_begin(self, logs=None):
        self.log_event("train_begin")

    def on_train_end(self, logs=None):
        self.log_event("train_end")

    def log_event(self, name, index=""):
        # print(f"[Info] {name} {index}")
        tmp = {
            "timestamp": str(datetime.now())
            , "event": name
            , "data": index
        }
        self.timestamp_log.append(tmp)

    def get_gpu_data(self):
        self.data_event.set()
        while self.data_queue.empty():
            time.sleep(0.1)
        gpu_data = self.data_queue.get()
        gpu_data = pd.DataFrame(gpu_data)
        return gpu_data

    def predict_energy(self, epoch):
        gpu_data = self.get_gpu_data()
        timestamps = pd.DataFrame(self.timestamp_log, columns=self.columns)
        power_data = PowerData(gpu_data, timestamps)
        pred = predict_energy_live(power_data, [0], self.num_epochs, epoch)
        actual = calculate_total_energy(power_data, [0])
        with self.summary_writer.as_default():
            tensorflow.summary.scalar("energy", data=actual, step=epoch)
            tensorflow.summary.scalar("energy-predicted", data=pred, step=epoch)
        self.summary_writer.flush()
        print(f"\nConsumed Energy: {actual / 1_000:.3f}/{pred / 1_000:.3f}kJ")

    def __del__(self):
        df = pd.DataFrame(self.timestamp_log, columns=self.columns)
        df.to_csv(self.log_path, index=False)


def get_log_path(new_path):
    data_root = Path(new_path)
    timestamp_log_path = data_root / "timestamps.csv"
    data_root.mkdir(parents=True, exist_ok=True)
    return timestamp_log_path


class ProcessTimer(abc.ABC):

    def __init__(self, interval, args=None, kwargs=None):
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

    def start(self):
        self.last_time = time.time()
        self._stop.clear()
        self.process.start()

    def stop(self):
        self._stop.set()

    @abc.abstractmethod
    def _on_start(self):
        pass

    @abc.abstractmethod
    def _on_stop(self):
        pass

    @abc.abstractmethod
    def _on_tick(self, args, kwargs):
        pass

    def _run(self):
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
    def __init__(self, device_id, interval, path, args=None, kwargs=None):
        super(Collector, self).__init__(interval, args, kwargs)
        self.data = []
        self.device_id = device_id
        self.lib = None
        self.device: Device = None
        self.path = Path(path)

    def _on_start(self):
        self.lib = NVMLLib()
        self.lib.open()
        self.device = self.lib.device.from_index(self.device_id)

    def _on_stop(self):
        self.device = None
        self.lib.close()
        self._save()

    @abc.abstractmethod
    def _get_save_path(self):
        pass

    def _save(self):
        # print(f"[saving] {self.__class__.__name__}")
        # print(self.get_len())
        path = self._get_save_path()
        df = pd.DataFrame(self.data)
        df.to_csv(path)

    def get_len(self):
        return len(self.data)


class SampleCollector(Collector):

    def __init__(self, sample_type, device_id, interval, path):
        super(SampleCollector, self).__init__(device_id, interval, path)
        self.sample_type = sample_type
        self.last_sample_time = 0

    def _on_tick(self, args, kwargs):
        samples = self.device.get_samples(self.sample_type, self.last_sample_time)
        self.data.extend(samples)
        self.last_sample_time = samples[-1].timestamp

    def _get_save_path(self):
        return self.path / (self.sample_type.get_filename() + ".csv")

    def test(self, device):
        try:
            # first call sometimes works but then the second will fail; idk why.
            self.device = device
            self._on_tick(None, None)
            self._on_tick(None, None)
            print(f"{self.sample_type}: OK")
            self.device = None
            return True
        except NVMLErrorNotFound as e:
            print(f"{self.sample_type} not supported on this device!")
            self.device = None
            return False


class SlowCollector(Collector):

    def _get_save_path(self):
        return self.path / "gpu-power.csv"

    def _on_tick(self, args, kwargs):
        data_event, data_queue = args
        util = self.device.get_utilization_rates()
        data = {
            "timestamp": str(datetime.now())
            , "util-gpu": util.gpu
            , "util-mem": util.memory

            , "clock-mem": self.device.get_clock(ClockType.MEM, ClockId.CURRENT)
            , "clock-gpu": self.device.get_clock(ClockType.SM, ClockId.CURRENT)
            # int representation to save on storage size
            , "power-state": self.device.get_power_state().value
            #
            , "power": self.device.get_power_usage()
            , "tmp": self.device.get_temperature(TemperatureSensors.TEMPERATURE_GPU)
            , "pci-tx": self.device.get_pcie_throughput(PcieUtilCounter.TX_BYTES)
            , "pci-rx": self.device.get_pcie_throughput(PcieUtilCounter.RX_BYTES)
        }
        self.data.append(data)
        # print(self.get_len())
        if data_event.is_set():
            data_queue.put(self.data)
            data_event.clear()


class CollectorManager:
    def __init__(self, data_path, device_id, ):
        self.collectors: List[Collector] = []
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.device_id = device_id

    def add(self, collector: Union[Collector, List[Collector]]):
        if isinstance(collector, list):
            self.collectors.extend(collector)
        else:
            self.collectors.append(collector)

    def add_by_sampling_type(self, sampling_type: Union[SamplingType,
                                                        List[SamplingType]]
                             , interval):
        if isinstance(sampling_type, SamplingType):
            sampling_type = [sampling_type]

        with NVMLLib() as lib:
            device = lib.device.from_index(self.device_id)
            for st in sampling_type:
                collector = SampleCollector(st, self.device_id, interval
                                            , self.data_path)
                if collector.test(device):
                    self.add(collector)

    def start(self):
        for col in self.collectors:
            col.start()

    def stop(self):
        for col in self.collectors:
            col.stop()


def patch(data_root, enable_energy, visible_devices):
    timestamp_log_path = get_log_path(data_root)
    # timestamp_log = open(timestamp_log_path , "w", buffering=1)
    # timestamp_log.write(f"timestamp,event,data\n")

    sampling_types = [
        SamplingType.GPU_UTILIZATION_SAMPLES,
        SamplingType.MEMORY_UTILIZATION_SAMPLES,
        SamplingType.MEMORY_CLK_SAMPLES,
        SamplingType.PROCESSOR_CLK_SAMPLES,
        SamplingType.TOTAL_POWER_SAMPLES
    ]

    sampling_manager = CollectorManager(data_root, visible_devices)

    data_event = Event()
    data_queue = Queue()

    sampling_manager.add(SlowCollector(visible_devices,
                                       interval=0.5,
                                       path=data_root,
                                       args=(data_event, data_queue)))

    sampling_manager.add_by_sampling_type(sampling_types, interval=1.5)

    sampling_manager.start()

    @atexit.register
    def on_exit():
        sampling_manager.stop()

    def get_patched_fit(original_function):
        def patched_fit(*args, **kwargs):
            num_epochs = kwargs.get("epochs")
            callback = EnergyCallback(enable_energy, num_epochs
                                      , timestamp_log_path, visible_devices
                                      , data_event, data_queue)
            kwargs.setdefault("callbacks", list()).append(callback)
            if not enable_energy:
                kwargs["verbose"] = 2
            return original_function(*args, **kwargs)

        return patched_fit

    model = tensorflow.keras.models.Model
    model.fit = get_patched_fit(model.fit)
    model.fit_generator = get_patched_fit(model.fit_generator)

    # log_event("experiment_begin")
