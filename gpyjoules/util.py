from collections import namedtuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from pathlib import Path
from datetime import datetime
from enum import Enum
from scipy.interpolate import interp1d
from typing import List, NamedTuple

# import pint
# unit = pint.UnitRegistry()
mpl.rcParams["figure.dpi"] = 300

MEGA = 1_000_000
KILO = 1_000


def diff3(array):
    """Compute the three point difference."""
    diff = (array[2:] - array[:-2]) / 2
    result = np.zeros_like(array, dtype=diff.dtype)
    result[1:-1] = diff
    return result


def load_data(path):
    """Load data from csv files."""
    path = Path(path)
    assert path.exists(), f"Path does not exist! {str(path)}"
    power_data = pd.read_csv(path / "gpu-power.csv")
    timestamps = pd.read_csv(path / "timestamps.csv")
    return power_data, timestamps


# %%
def preprocess_timestamps(timestamps):
    timestamps["timestamp"] = pd.to_datetime(timestamps["timestamp"])
    # return timestamps


# %%
def units_to_si_base(power_data):
    """Converts milliwatt to watt (in place)."""
    power_data.power = power_data.power / KILO


class Epoch(NamedTuple):
    """Represent one epoch marker."""
    index: int
    begin: datetime
    end: datetime


class Batch(NamedTuple):
    """Represent one batch marker."""
    index: int
    begin: datetime
    end: datetime


# %%
class BatchIterator:
    """BatchIterator for iterating batches."""
    def __init__(self, t_info):
        self.t_info = t_info
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < self.t_info.batch_count:
            begin = self.t_info.get_batch_begin(self.index)
            end = self.t_info.get_batch_end(self.index)
            batch = Batch(self.index, begin, end)
            self.index += 1
            return batch
        else:
            raise StopIteration


# Epoch = namedtuple("Epoch", ["index", "begin", "end"])


class EpochIterator:
    """EpochIterator for iterating epochs."""
    def __init__(self, t_info, start=0, end=None, step=1):
        self.t_info = t_info
        self.start = start
        self.index = self.start
        if end is None:
            self.end = self.t_info.epoch_count
        else:
            self.end = min(self.t_info.epoch_count, end)
        self.step = step

    def __iter__(self):
        self.index = self.start
        return self

    def __next__(self):
        if self.index < self.end:
            epoch_begin = self.t_info.get_epoch_begin(self.index)
            epoch_end = self.t_info.get_epoch_end(self.index)

            epoch = Epoch(self.index, epoch_begin, epoch_end)

            self.index += self.step
            return epoch
        else:
            raise StopIteration

    def __getitem__(self, index):
        index = index + self.start
        if index >= self.end:
            raise IndexError()

        epoch_begin = self.t_info.get_epoch_begin(index)
        epoch_end = self.t_info.get_epoch_end(index)
        return Epoch(index, epoch_begin, epoch_end)


class EventType(Enum):
    """Define possible event types."""
    EPOCH_BEGIN = "epoch_begin"
    EPOCH_END = "epoch_end"

    TRAIN_BEGIN = "train_begin"
    TRAIN_END = "train_end"

    EXPERIMENT_BEGIN = "experiment_begin"
    EXPERIMENT_END = "experiment_end"


class TimestampInfo:
    def __init__(self, timestamps):
        self.timestamps = timestamps
        self.timestamps["timestamp"] = pd.to_datetime(self.timestamps["timestamp"])

    def _get_event_index(self, event, index):
        if index is None:
            selection = self.timestamps  # [self.timestamps.data.isnull()]
        else:
            selection = self.timestamps[self.timestamps.data == index]

        selected_event = selection[selection.event == event]

        if selected_event.empty:
            return None
        else:
            return selected_event.iloc[0].timestamp

    def get_epoch_begin(self, index):
        return self._get_event_index("epoch_begin", index)

    def get_epoch_end(self, index):
        return self._get_event_index("epoch_end", index)

    def get_batch_begin(self, index):
        return self._get_event_index("batch_begin", index)

    def get_batch_end(self, index):
        return self._get_event_index("batch_end", index)

    @property
    def batch_count(self):
        return len(self.timestamps[self.timestamps.event == "batch_begin"])

    @property
    def epoch_count(self):
        return len(self.timestamps[self.timestamps.event == "epoch_begin"])

    @property
    def time_per_epoch(self):
        epochs = np.array(list(self.iter_epochs()))
        e_times = np.array(
            epochs[:, 2] - epochs[:, 1], dtype=np.timedelta64
        ) / np.timedelta64(1, "s")
        return e_times

    @property
    def train_begin(self):
        return self._get_event_index("train_begin", None)

    @property
    def train_end(self):
        return self._get_event_index("train_end", None)

    @property
    def experiment_begin(self):
        return self._get_event_index("experiment_begin", None)

    @property
    def experiment_end(self):
        return self._get_event_index("experiment_end", None)

    @property
    def total_train_duration(self):
        return self.train_end() - self.train_begin()

    @property
    def total_experiment_duration(self):
        return self.experiment_end - self.experiment_begin

    def iter_epochs(self, start=0, end=None, step=1):
        return EpochIterator(self, start, end, step)

    @property
    def epochs(self):
        return self.iter_epochs()

    @property
    def batches(self):
        return BatchIterator(self)


class PowerData:
    """"""

    def __init__(self, power_gpu, timestamps):
        self.power_gpu = power_gpu
        self.timestamps = timestamps

        self.preprocess()

        self.t_info = TimestampInfo(self.timestamps)

    def preprocess(self):
        preprocess_timestamps(self.timestamps)
        preprocess_timestamps(self.power_gpu)
        units_to_si_base(self.power_gpu)
        # self.timestamps.data = self.timestamps.data.astype(int)

    def iter_epochs(self, *args, **kwargs):
        return self.t_info.iter_epochs(*args, **kwargs)

    @property
    def epochs(self):
        return self.t_info.epochs

    @staticmethod
    def load(path):
        return PowerData(*load_data(path))


def calculate_energy(power, timestamps):
    """Calculate the integral of power [W] over time [DateTime] (energy [J])
    using a 3point difference for the timestamps
    
    Power should be in Watts
    returns Energy in Joule
    """
    power = np.array(power)
    timestamps = np.array(timestamps)
    time_diff = diff3(timestamps) / np.timedelta64(1, "s")
    return power.T @ time_diff


def get_cumulative_energy(power, timestamps):
    power = np.array(power)
    timestamps = np.array(timestamps)
    time_diff = diff3(timestamps)  # / np.timedelta64(1, 's')
    energy = power * time_diff
    return np.cumsum(energy)


def calculate_total_energy(power_data, devices, start=None, end=None):
    start = power_data.timestamps.iloc[0].timestamp if start is None else start
    end = power_data.timestamps.iloc[-1].timestamp if end is None else end

    data_slice = power_data.power_gpu[
        (power_data.power_gpu.timestamp >= start)
        & (power_data.power_gpu.timestamp <= end)
    ]
    # todo get timestamp for actual experiment start without baseline
    total_energy = 0
    for dev in devices:
        dev_data = data_slice[data_slice["gpu-index"] == dev]
        total_energy += calculate_energy(dev_data.power, dev_data.timestamp)
    return total_energy


def calculate_total_energy_experiment(power_data, devices):
    start = power_data.t_info.experiment_begin()
    end = power_data.t_info.experiment_end()
    return calculate_total_energy(power_data, devices, start, end)


def interpolate(x, y, step):
    x_tick = np.arange(x[0], x[-1], step)
    f_cubic = interp1d(x, y, kind="cubic")
    return x_tick, f_cubic(x_tick)


def pad_n_mask(values):
    max_len = max([len(x) for x in values])
    result_padded = []
    for i in values:
        padded = np.zeros(max_len)
        mask = np.ones(max_len)

        padded[: len(i)] = i
        mask[: len(i)] = 0

        tmp = np.ma.array(padded, mask=mask)
        result_padded.append(tmp)
    return np.ma.array(result_padded)


def plot_mean_std(x, Y, **kwargs):
    Y = Y / KILO
    mean = Y.mean(axis=0)
    std = Y.std(axis=0)
    import scipy.stats as st

    # a = 1.0 * np.array(data)
    # n = len(a)
    confidence = 0.95
    h = std * st.t.ppf((1 + confidence) / 2.0, Y.shape[1] - 1)

    plt.plot(x, mean, **kwargs)
    kwargs["label"] = None
    # sns.lineplot(x="x", y="y", data=df)
    plt.fill_between(
        x, mean - h, mean + h, alpha=0.1, linewidth=3, antialiased=True, **kwargs
    )


def timestamp2second(ts):
    return (ts - ts[0]) / np.timedelta64(1, "s")


def calculate_total_cumulative_energy(power_data, devices, start=None, end=None):
    start = power_data.iloc[0].timestamp if start is None else start
    end = power_data.iloc[-1].timestamp if end is None else end

    data_slice = power_data.power_gpu[
        (power_data.power_gpu.timestamp >= start)
        & (power_data.power_gpu.timestamp <= end)
    ]
    # todo get timestamp for actual experiment start without baseline
    total_energy = []
    for dev in devices:
        dev_data = data_slice[data_slice["gpu-index"] == dev]
        timestamp = np.array(dev_data.timestamp)
        power = dev_data.power
        timestamp = timestamp2second(timestamp)

        timestamp, power = interpolate(timestamp, power, 0.25)
        total_energy.append((timestamp, get_cumulative_energy(power, timestamp)))
    return total_energy


def get_energy_per_epoch(power_data, devices):
    energy = []
    for index, epoch_begin, epoch_end in power_data.iter_epochs():
        tmp = calculate_total_energy(power_data, devices, epoch_begin, epoch_end)
        energy.append(tmp)
    return energy


def predict_energy(power_data, devices, num_epochs, start_epoch=1):
    """Predict energy using selected measures and data in range of
    start_epoch to start_epoch + num_epochs."""

    start = power_data.t_info.get_epoch_begin(start_epoch)
    end = power_data.t_info.get_epoch_end(start_epoch + num_epochs - 1)

    exp_begin = power_data.t_info.experiment_begin()
    exp_end = power_data.t_info.experiment_end()

    first_epoch_begin = power_data.t_info.get_epoch_begin(0)
    first_epoch_end = power_data.t_info.get_epoch_end(0)

    last_epoch_end = power_data.t_info.get_epoch_end(
        power_data.t_info.epoch_count() - 1
    )

    energy_warm = calculate_total_energy(power_data, devices, start, end)
    energy_warmup = calculate_total_energy(
        power_data, devices, exp_begin, first_epoch_begin
    ) + calculate_total_energy(power_data, devices, first_epoch_begin, first_epoch_end)
    energy_teardown = calculate_total_energy(
        power_data, devices, last_epoch_end, exp_end
    )

    energy_per_epoch = energy_warm / num_epochs
    energy_warm = (power_data.t_info.epoch_count() - 1) * energy_per_epoch

    total_energy = energy_warmup + energy_warm + energy_teardown
    return total_energy


def predict_energy_live(
    power_data: PowerData,
    devices: List[int],
    num_epochs: int,
    current_epoch: int,
    start_epoch: int = 1,
) -> float:
    """Predict energy using selected measures and data in range
    of start_epoch to start_epoch + num_epochs.
    
    :returns total_energy in Joule
    """

    epochs_since_start = current_epoch - start_epoch + 1
    warm_epochs = num_epochs - start_epoch

    exp_begin = power_data.t_info.train_begin
    first_epoch_end = power_data.epochs[0].end
    energy_warmup = calculate_total_energy(
        power_data, devices, exp_begin, first_epoch_end
    )

    start = power_data.epochs[start_epoch].begin
    end = power_data.epochs[current_epoch].end
    energy_warm = calculate_total_energy(power_data, devices, start, end)

    energy_per_epoch = energy_warm / epochs_since_start
    energy_warm = energy_per_epoch * warm_epochs

    total_energy = energy_warmup + energy_warm
    return total_energy
