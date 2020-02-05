# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl
from pathlib import Path
from datetime import datetime
import pint
unit = pint.UnitRegistry()
mpl.rcParams['figure.dpi'] = 300


# %%
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
            selection = self.timestamps[self.timestamps.data.isnull()]
        else:
            selection = self.timestamps[self.timestamps.data == index]
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
    
    def epochs(self, *args, **kwargs):
        return self.t_info.epochs(*args, **kwargs)

    @staticmethod    
    def load(path):
        return PowerData(*load_data(path))
    


# %%
power_data = PowerData.load("power-data/8batch128denseX1-2020-01-26 22:45:57.471455")


# %% [markdown]
# ``` python3
# #plt.plot(np.array(lastxp["timestamp"])[:-1], gpu_mem[1:]-gpu_mem[:-1])
# #plt.plot(np.array(lastxp["timestamp"]), lastxp["gpu0-util"])gpu0-pci-tx
# plt.plot(np.array(power_data["timestamp"]), power_data["gpu0-power"]/power_data["gpu0-power"].max())
# plt.plot(np.array(power_data["timestamp"]), power_data["gpu0-pci-tx"]/power_data["gpu0-pci-tx"].max())
# plt.plot(np.array(power_data["timestamp"]), power_data["gpu0-pci-rx"]/power_data["gpu0-pci-rx"].max())
# plt.plot(epoch_begin['timestamp'], np.full(len(epoch_begin),0.5),"x")
# #plt.plot(batch_begin['timestamp'], np.full(len(batch_begin),20),"^")
# #plt.plot(df["gpu0-power"][:1000]/3000)
# plt.legend(["gpu0-util", "epoch-begin"])
# plt.title("Training on one GPU -- Power vs. Time")
# plt.xlabel("Time [UTC]")
# plt.ylabel("Power Consumption [W]")
# ```

# %%
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


# %%
errors = []


# %%
def calculate_total_energy(power_data, devices, start=None, end=None):
    start = power_data.iloc[0].timestamp if start is None else start
    end = power_data.iloc[-1].timestamp if end is None else end

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


# %%
get_energy_per_epoch(power_data)
#power_data.timestamps[ (power_data.timestamps.event == "train_begin") & power_data.timestamps.data.isnull()]
#(power_data.timestamps.event == "train_begin") &
#power_data.power_gpu.head()
energys = get_energy_per_epoch(power_data)
print(np.mean(energys)/1_000_000)
print(np.std(energys)/1_000_000)



# %%
energy_epoch = get_energy_per_epoch(power_data)

# %%
x = np.arange(len(energy_epoch))
plt.bar(x,energy_epoch)
plt.xticks(x[::2])

# %%
plt.bar(np.arange(len(energy_epoch)),energy_epoch)

# %%
predictions = []
actual = calculate_total_energy_experiment(power_data,[0])
for i in range(1,power_data.t_info.get_epoch_count()):
    pred = predict_energy(power_data,[0],i)
    predictions.append(pred)

# %%
error = ((actual - np.array(predictions)) / actual)# **2)
#plt.plot((power_data.t_info.get_epoch_end(np.arange(1,power_data.t_info.get_epoch_count()-1))  - power_data.t_info.get_epoch_begin(0))  / np.timedelta64(1, 's'),error*100)
plt.plot(np.arange(2,len(error)+2),error*100)
plt.title("Predicted Energy Error")
plt.xlabel("#Epochs used for Prediction")
plt.ylabel("Prediction Error [%]")


# %%
power_single_epoch = power_data[(power_data.timestamp > e1) & (power_data.timestamp < e2)]
power_warmup = power_data[power_data.timestamp < e1]
power_warm = power_data[(power_data.timestamp > e1) & (power_data.timestamp < en)]

# %%
actual = integrate_power(power_warm["gpu0-power"]/1000, power_warm["timestamp"]) / (3600 * 1000)


# %%
def get_experiment_duration(timestamps):
    return timestamps.get


# %%
dense_count = [1, 2, 4]
dense_sizes = [128, 256]
batch_sizes = [8, 16, 32, 64, 128]

data = []
for c in dense_count:
    for d in dense_sizes:
        for b in batch_sizes:
            description = f"{b}batch{d}denseX{c}"
            print(f"loading {description}")
            data_root = Path("power-data")
            data_path = list(data_root.glob(f"{description}*"))[0]
            
            power_data = PowerData.load(data_path)
            energy = get_energy_per_epoch(power_data)
            # take average energy, exclude first epoch
            energy = np.mean(energy[1:])
            time = power_data.t_info.get_time_per_epoch()
            time = np.mean(time[1:])
            tmp = {
                "dense_count":c
                , "dense_sizes":d
                , "batch_sizes":b
                , "energy":energy
                , "time":time
            }
            data.append(tmp)

data = pd.DataFrame(data)
data

# %%
d = data[(data["dense_sizes"] == 256) &(data["dense_count"] == 4)]

# %%
x = np.arange(1,128,1)
#reg = LinearRegression().fit((np.array(1/d["batch_sizes"])).reshape(-1,1), (d["time"]/ np.timedelta64(1, 's')))
reg = np.poly1d(np.polyfit(1/np.array(d["batch_sizes"]), d["energy"], 1))
y_hat = reg(1/x) #reg.predict(1/x.reshape(-1,1))1

# %%
plt.plot(d["batch_sizes"], d["energy"]/1_000_000,"x")
plt.plot(x, y_hat/1_000_000)
plt.title("Energy vs. Batch Size")
plt.xlabel("Batch Size")
plt.ylabel("Energy [MJ]")

# %%
data.columns

# %%
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(np.array(data[['batch_sizes', 'dense_count', 'dense_sizes']]), data["energy"] )

# %%
reg.score(np.array(data[['batch_sizes', 'dense_count', 'dense_sizes']]), data["energy"])

# %%
from sklearn.metrics import mean_squared_error
mean_squared_error(np.array(data["energy"]), reg.predict(np.array(data[['batch_sizes', 'dense_count', 'dense_sizes']])))

# %%
np.array(data["energy"]) - reg.predict(np.array(data[['batch_sizes', 'dense_count', 'dense_sizes']]))

# %%
reg.predict(np.array(data[['batch_sizes', 'dense_count', 'dense_sizes']]))/1000

# %%
power_data, timestamps = load_data("power-data-dgx/power-data/powercap250-2020-01-28T23:21:35.720872")

preprocess(power_data, timestamps)
t_info = TimestampInfo(timestamps)

# %%
power_data.head()

# %%
plt.plot(power_data[(power_data["gpu-index"] == 0) ]["power"]/1000)


# %%
def get_time_per_epoch(t_info):



# %%
#power_caps = [150, 200, 250, 300] #Watts
power_caps = [150,175, 200,225, 250, 275] #Watts

epoch_count = 5#20
data = []
for p in power_caps:
    description = f"powercap{p}"
    print(f"loading {description}")
    data_root = Path("power-random/power-data-fine")
    data_path = list(data_root.glob(f"{description}*"))[0]
    
    power_data = PowerData.load(data_path)
    assert power_data.t_info.get_epoch_count() == epoch_count, "Unexpected Number of epochs!"
    energy = get_energy_per_epoch(power_data)#calculate_total_energy(power_data, ["power"],start=t_info.get_epoch_begin(2))
    energy = np.mean(energy[1:])
    time = power_data.t_info.get_time_per_epoch()
    time = np.mean(time[1:])
    total_time =power_data.t_info.get_total_experiment_duration() / np.timedelta64(1,"s")
    total_energy = calculate_total_energy(power_data, [0]
                    ,start=power_data.t_info.get_train_begin()
                    , end=power_data.t_info.get_train_end())
    edp = total_energy * total_time
    gpu0 = power_data.power_gpu[power_data.power_gpu["gpu-index"] == 0]
    voltage = np.sqrt(gpu0.power / (gpu0["util-gpu"] * gpu0["clock-sm"]))
    tmp = {
        "power_caps":p
        , "energy":energy
        , "time":time
        , "total_time":total_time
        , "total_energy":total_energy
        , "cum_energy":calculate_total_cumulative_energy(power_data, [0]
                    ,start=power_data.t_info.get_train_begin()
                    , end=power_data.t_info.get_train_end())[0]
        , "edp": edp
        ,"voltage": voltage
    }
    data.append(tmp)
power_cap_data = pd.DataFrame(data)

# %%
#plt.plot(power_data.power_gpu["clock-sm"])
#gpu0 = power_data.power_gpu[power_data.power_gpu["gpu-index"] == 0]

#plt.plot(voltage)
for c in power_cap_data.voltage:
    plt.plot(c)
plt.legend(power_cap_data.power_caps)

# %%
gpu0.head()

# %%
plt.plot(power_cap_data.power_caps, power_cap_data.total_energy/1_000_000)

# %%
plt.plot(power_cap_data.power_caps, power_cap_data.edp/1_000_000, "x")
plt.title("Energy Delay Product vs. Power Limit")
plt.xlabel("Power Limit [W]")
plt.ylabel("EDP [$M J s$]")
plt.xticks(power_caps)

# %%
for c in power_cap_data.cum_energy:
    plt.plot(c / 1_000_000)
plt.legend([f"{x}W" for x in power_cap_data.power_caps])
plt.title("Cumulative Energy w/ Power Limits")
plt.xlabel("Time [$s$]")
plt.ylabel("Energy [$MJ$]")

# %%
plt.plot(power_cap_data.power_caps, power_cap_data.total_time, "x")
plt.title("Execution Time vs. Power Limit")
plt.xlabel("Power Limit [$W$]")
plt.ylabel("Execution Time [$s$]")

# %%
gpu0 = power_data.power_gpu[power_data.power_gpu["gpu-index"] == 0]
cum_energy = get_cumulative_energy(gpu0.power, gpu0.timestamp)
plt.plot(gpu0.timestamp,cum_energy)
a,b,c = zip(*power_data.epochs())
plt.plot(b, np.full(len(b),0.5),"x")

# %%
epochs = np.array(list(power_data.epochs(1)))
e_times = np.array(epochs[:,2] - epochs[:,1], dtype=np.timedelta64) / np.timedelta64(1, 's')
plt.hist(e_times)
#e_times

# %%
plt.plot(power_cap_data.power_caps, power_cap_data.time/np.timedelta64(1, 's'))

# %%

# %% [markdown]
#
