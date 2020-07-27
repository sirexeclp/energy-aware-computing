import warnings

# with warnings.catch_warnings():
from device import Device
from enums import TemperatureSensors, PcieUtilCounter, SamplingType
from pynvml import NVMLLib

warnings.filterwarnings('ignore')
import tensorflow
import tensorflow.keras as keras
import atexit
from multiprocessing import Process, Event, Queue
from time import sleep, time
from .util import *

stop_event = Event()
data_event = Event()
data_queue = Queue()


class EnergyCallback(keras.callbacks.Callback):
    """Costum Keras Callback to log events, for energy measurements."""

    def __init__(self, enable_energy_prediction, num_epochs, log_path, visible_devices):
        super(EnergyCallback, self).__init__()
        self.timestamp_log = []
        self.enable_energy_prediction = enable_energy_prediction
        self.num_epochs = num_epochs
        self.log_path = log_path
        self.columns = ["timestamp", "event", "data"]
        self.total_batch = 0
        self.visible_devices = visible_devices
        self.last_time = time()
        self.summary_writer = tensorflow.summary.create_file_writer("/tmp/energy-board/energy")

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

    @staticmethod
    def get_gpu_data():
        data_event.set()
        while data_queue.empty():
            sleep(0.1)
        gpu_data = data_queue.get()
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


def get_all_stats(device: Device):
    data = {
        "gpu-index": device.get_index()
        , "timestamp": str(datetime.now())
        # , "util-gpu": self.get_sample(SampleType.GpuUtilization)
        # , "util-mem": self.get_sample(SampleType.MemoryUtilization)
        #
        # , "clock-mem": self.get_sample(SampleType.MemoryClock)
        # , "clock-gpu": self.get_sample(SampleType.GpuClock)
        #
        , "power-state": device.get_power_state()
        #
        # , "power": self.get_sample(SampleType.TotalPower)
        , "power": device.get_power_usage()
        , "tmp": device.get_temperature(TemperatureSensors.TEMPERATURE_GPU)
        , "pci-tx": device.get_pcie_throughput(PcieUtilCounter.TX_BYTES)
        , "pci-rx": device.get_pcie_throughput(PcieUtilCounter.RX_BYTES)
    }
    return data


class SampleCollector:

    def __init__(self, sample_type):
        self.sample_type = sample_type
        self.data = []
        self.last_time = 0

    def collect(self, device):
        samples = device.get_samples(self.sample_type, self.last_time)
        self.data.extend(samples)
        self.last_time = samples[-1].timestamp

    def to_csv(self, path):
        path = Path(path) / (self.sample_type.get_filename() + ".csv")
        df = pd.DataFrame(self.data)
        df.to_csv(path)



def collect_power_data(data_path, done, get_data, data, visible_devices):
    gpu_data = []
    util_gpu = SampleCollector(SamplingType.GPU_UTILIZATION_SAMPLES)
    util_mem = SampleCollector(SamplingType.MEMORY_UTILIZATION_SAMPLES)
    clock_mem = SampleCollector(SamplingType.MEMORY_CLK_SAMPLES)
    clock_gpu = SampleCollector(SamplingType.PROCESSOR_CLK_SAMPLES)
    total_power = SampleCollector(SamplingType.TOTAL_POWER_SAMPLES)

    collectors = [util_gpu, util_mem, total_power, clock_gpu, clock_mem]

    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    gpu_data_path = data_path / "gpu-power.csv"

    with NVMLLib() as lib:
        device = Device.from_index(visible_devices)
        print("start recording")
        while not done.is_set():
            gpu_data.append(get_all_stats(device))
            for collector in collectors:
                collector.collect(device)

            if get_data.is_set():
                get_data.clear()
                data.put(gpu_data)
            sleep(1)

        df = pd.DataFrame(gpu_data)
        df.to_csv(gpu_data_path)
        df.mean()
        for collector in collectors:
            collector.to_csv(data_path)

def patch(data_root, enable_energy, visible_devices):
    timestamp_log_path = get_log_path(data_root)
    # timestamp_log = open(timestamp_log_path , "w", buffering=1)
    # timestamp_log.write(f"timestamp,event,data\n")

    power_process = Process(target=collect_power_data,
                            args=(data_root, stop_event, data_event, data_queue, visible_devices))
    power_process.start()

    @atexit.register
    def on_exit():
        stop_event.set()
        power_process.join()

    def get_patched_fit(original_function):
        def patched_fit(*args, **kwargs):
            num_epochs = kwargs.get("epochs")
            callback = EnergyCallback(enable_energy, num_epochs, timestamp_log_path, visible_devices)
            kwargs.setdefault("callbacks", list()).append(callback)
            if not enable_energy:
                kwargs["verbose"] = 2
            return original_function(*args, **kwargs)

        return patched_fit

    model = tensorflow.keras.models.Model
    model.fit = get_patched_fit(model.fit)
    model.fit_generator = get_patched_fit(model.fit_generator)

    # log_event("experiment_begin")
