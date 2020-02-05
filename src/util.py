import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl
from pathlib import Path
from datetime import datetime
#import pint
#unit = pint.UnitRegistry()
mpl.rcParams['figure.dpi'] = 300


def load_data(path):
    path = Path(path)
    assert path.exists(), f"Path does not exist! {str(path)}"
    power_data = pd.read_csv(path / "gpu-power.csv")
    timestamps = pd.read_csv(path / "timestamps.csv")
    return power_data, timestamps


# %%
def preprocess_timestamps(timestamps):
    timestamps['timestamp'] = pd.to_datetime(timestamps['timestamp'])
    #return timestamps


# %%
def units_to_si_base(power_data):
    columns = [x for x in power_data.columns if "-power" in x]
    for c in columns:
        power_data[c] = (power_data[c] * unit.milliwatt).to("watt")


# %%
class BatchIterator():
    def __init__(self, t_info):
        self.t_info = t_info
        self.index = 0
    
    def __iter__(self):
        self.index = 0
        return self
    
    def __next__(self):
        if self.index < self.t_info.get_batch_count():
            begin = self.t_info.get_batch_begin(self.index)
            #end = self.t_info.get_batch_end(self.index) 
            self.index += 1
            return begin#, end
        else:
            raise StopIteration


class EpochIterator():
    def __init__(self, t_info, start=0, end=None, step=1):
        self.t_info = t_info
        self.start= start
        self.index = self.start
        if end is None:
            self.end = self.t_info.get_epoch_count()
        else:
            self.end = min(self.t_info.get_epoch_count(), end)
        self.step = step
    
    def __iter__(self):
        self.index = self.start
        return self
    
    def __next__(self):
        if self.index < self.end:
            epoch_begin = self.t_info.get_epoch_begin(self.index)
            epoch_end = self.t_info.get_epoch_end(self.index) 
            self.index += self.step

            return self.index-self.step, epoch_begin, epoch_end
        else:
            raise StopIteration



class TimestampInfo():
    def __init__(self, timestamps):
        self.timestamps = timestamps
    
    def _get_event_index(self, event, index):
        if index is None:
            selection = self.timestamps#[self.timestamps.data.isnull()]
        else:
            selection = self.timestamps[self.timestamps.data == index]
        #print(event, index)
        #print(len(selection))
        return selection[selection.event == event].iloc[0].timestamp
    
    def get_epoch_begin(self, index):
        return self._get_event_index("epoch_begin", index)
    
    def get_epoch_end(self, index):
        return self._get_event_index("epoch_end", index)
    
    def get_batch_begin(self, index):
        return self._get_event_index("batch_begin", index)

    #def get_batch_end(self, index):
    #    return self._get_event_index("batch_end", index)
    
    def get_batch_count(self):
        return len(self.timestamps[self.timestamps.event == "batch_begin"])
    
    def get_epoch_count(self):
        return len(self.timestamps[self.timestamps.event == "epoch_begin"])
    
    def get_time_per_epoch(self):
        epochs = np.array(list(self.epochs()))
        e_times = np.array(epochs[:,2] - epochs[:,1], dtype=np.timedelta64) / np.timedelta64(1, 's')
        return e_times
    
    def get_train_begin(self):
        return self._get_event_index("train_begin", None)
    
    def get_train_end(self):
        return self._get_event_index("train_end", None)

    def get_experiment_begin(self):
        return self._get_event_index("experiment_begin", None)
    
    def get_experiment_end(self):
        return self._get_event_index("experiment_end", None)

    def get_total_train_duration(self):
        return self.get_train_end() - self.get_train_begin()
    
    def get_total_experiment_duration(self):
        return self.get_experiment_end() - self.get_experiment_begin()
    
    def epochs(self, start=0, end=None, step=1):
        return EpochIterator(self, start, end, step)
    
    def batches(self):
        return BatchIterator(self)

class PowerData():
    def __init__(self, power_gpu, timestamps):
        self.power_gpu = power_gpu
        self.timestamps = timestamps

        self.preprocess()

        self.t_info = TimestampInfo(self.timestamps)
    
    def preprocess(self):
        preprocess_timestamps(self.timestamps)
        preprocess_timestamps(self.power_gpu)
        units_to_si_base(self.power_gpu)
        #self.timestamps.data = self.timestamps.data.astype(int)
    
    def epochs(self, *args, **kwargs):
        return self.t_info.epochs(*args, **kwargs)

    @staticmethod    
    def load(path):
        return PowerData(*load_data(path))
    

def diff3(x):
    d =(x[2:]-x[:-2]) / 2
    result = np.zeros_like(x, dtype=d.dtype)
    result[1:-1] = d
    return result


def calculate_energy(power, timestamps):
    """Calculate the integral of power [W] over time [DateTime] (energy [J]) using a 3point difference for the timestamps
        Power should be in Watts
        returns Energy in Joule
    """
    power = np.array(power)
    timestamps = np.array(timestamps)
    time_diff = diff3(timestamps) / np.timedelta64(1, 's')
    return power.T @ time_diff

def get_cumulative_energy(power, timestamps):
    power = np.array(power)
    timestamps = np.array(timestamps)
    time_diff = diff3(timestamps) / np.timedelta64(1, 's')
    energy = power * time_diff
    return np.cumsum(energy)

def calculate_total_energy(power_data, devices, start=None, end=None):
    start = power_data.timestamps.iloc[0].timestamp if start is None else start
    end = power_data.timestamps.iloc[-1].timestamp if end is None else end

    data_slice = power_data.power_gpu[(power_data.power_gpu.timestamp >= start) & (power_data.power_gpu.timestamp <= end)]
    #todo get timestamp for actual experiment start without baseline
    total_energy = 0
    for dev in devices:
        dev_data = data_slice[data_slice["gpu-index"] == dev]
        #print(f"max: {np.max(dev_data.power)/1_000}")
        #print(f"min: {np.min(dev_data.power)/1_000}")
        #print(f"mean: {np.mean(dev_data.power)/1_000}")
        #print(f"sd: {np.std(dev_data.power)/1_000}")
        total_energy += calculate_energy(dev_data.power, dev_data.timestamp)
    return total_energy

def calculate_total_energy_experiment(power_data, devices):
    start = power_data.t_info.get_experiment_begin()
    end = power_data.t_info.get_experiment_end()
    return calculate_total_energy(power_data, devices, start, end)
    

def calculate_total_cumulative_energy(power_data, devices, start=None, end=None):
    start = power_data.iloc[0].timestamp if start is None else start
    end = power_data.iloc[-1].timestamp if end is None else end

    data_slice = power_data.power_gpu[(power_data.power_gpu.timestamp >= start) & (power_data.power_gpu.timestamp <= end)]
    #todo get timestamp for actual experiment start without baseline
    total_energy = []
    for dev in devices:
        dev_data = data_slice[data_slice["gpu-index"] == dev]
        total_energy.append(get_cumulative_energy(dev_data.power, dev_data.timestamp))
    return total_energy

def get_energy_per_epoch(power_data):
    energy = []
    for index, epoch_begin, epoch_end in power_data.epochs():
        tmp = calculate_total_energy(power_data, [0], epoch_begin, epoch_end)
        energy.append(tmp)
    return energy

def predict_energy(power_data,devices,num_epochs,start_epoch=1):
    """Predict energy using selected measures and data in range of start_epoch to start_epoch + num_epochs. """

    start = power_data.t_info.get_epoch_begin(start_epoch)
    end = power_data.t_info.get_epoch_end(start_epoch+num_epochs - 1 )

    exp_begin = power_data.t_info.get_experiment_begin()
    exp_end= power_data.t_info.get_experiment_end()
    
    first_epoch_begin = power_data.t_info.get_epoch_begin(0)
    first_epoch_end = power_data.t_info.get_epoch_end(0)

    last_epoch_end = power_data.t_info.get_epoch_end(power_data.t_info.get_epoch_count()-1)
    
    energy_warm = calculate_total_energy(power_data, devices, start, end)
    energy_warmup = calculate_total_energy(power_data, devices, exp_begin, first_epoch_begin) +\
                        calculate_total_energy(power_data, devices, first_epoch_begin, first_epoch_end)
    energy_teardown = calculate_total_energy(power_data, devices,last_epoch_end, exp_end)

    energy_per_epoch = energy_warm / num_epochs
    energy_warm = (power_data.t_info.get_epoch_count() - 1) * energy_per_epoch

    total_energy = energy_warmup + energy_warm + energy_teardown
    return total_energy


def predict_energy_live(power_data,devices,num_epochs,current_epoch,start_epoch=1):
    """Predict energy using selected measures and data in range of start_epoch to start_epoch + num_epochs. """

    start = power_data.t_info.get_epoch_begin(start_epoch)
    end = power_data.t_info.get_epoch_end(current_epoch)

    exp_begin = power_data.t_info.get_experiment_begin()
    
    first_epoch_begin = power_data.t_info.get_epoch_begin(0)
    first_epoch_end = power_data.t_info.get_epoch_end(0)
    
    energy_warm = calculate_total_energy(power_data, devices, start, end)
    energy_warmup = calculate_total_energy(power_data, devices, exp_begin, first_epoch_begin) +\
                        calculate_total_energy(power_data, devices, first_epoch_begin, first_epoch_end)
    
    energy_per_epoch = energy_warm / (current_epoch-start_epoch+1)
    #energy_warm = energy_per_epoch * num_epochs
    energy_warm += (num_epochs-(current_epoch-start_epoch+1)) * energy_per_epoch

    total_energy = energy_warmup + energy_warm
    return total_energy