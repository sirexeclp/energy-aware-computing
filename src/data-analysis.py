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
%load_ext autoreload

%autoreload 2

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

from util import *


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


# %%
errors = []


# %%



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


# %%
GPU1=1
devices=[GPU1]
power_caps = [150, 200, 250, 300]

#%%
def load_power_cap_data(data_root, network, epoch_count):
    
    data_root = Path(data_root)
    runs = list(data_root.glob("run*"))

    all_data = []

    for cap in power_caps:
        for run in runs:
            description = f"powercap{cap}-{network}"
            #print(description)
            data_path = list(run.glob(f"{description}*"))[0]
            power_data = PowerData.load(data_path)
            assert power_data.t_info.get_epoch_count() == epoch_count, "Unexpected Number of epochs!"
            epoch_energy = get_energy_per_epoch(power_data, devices=devices)
            energy = np.mean(epoch_energy[1:])
           
            epoch_times = power_data.t_info.get_time_per_epoch()
            mean_time = np.mean(epoch_times[1:])
            
            total_time = power_data.t_info.get_total_experiment_duration() / np.timedelta64(1,"s")
            total_energy = calculate_total_energy(power_data, devices=devices
                            ,start=power_data.t_info.get_train_begin()
                            , end=power_data.t_info.get_train_end())
            
            edp = total_energy * total_time

            cum_energy = calculate_total_cumulative_energy(power_data, devices=devices
                            ,start=power_data.t_info.get_train_begin()
                            , end=power_data.t_info.get_train_end())[0]

            tmp = {
                "power_cap":cap
                ,"energy":energy
                ,"epoch_energy":epoch_energy
                ,"mean_time":mean_time
                ,"total_time":total_time
                ,"epoch_times":epoch_times
                ,"total_energy":total_energy
                ,"cum_energy": cum_energy
                ,"edp": edp
                ,"run": str(run).split("/")[-1].replace("run","")
                ,"power_data": power_data
            }
            all_data.append(tmp)
    return all_data
# %%
mnist_data_big = load_power_cap_data("../data/data-1.2", "mnist-big", epoch_count = 10)
mnist_data_big = pd.DataFrame(mnist_data_big)

# %%plot_epoch_times
    fig, ax = plt.subplots(len(power_caps), 1, sharex=True)
    for index,(_, pl) in enumerate(df[df.run=="1"].iterrows()):
        current_ax = ax[index]
        power_data = pl["power_data"]
        device = power_data.power_gpu[power_data.power_gpu["gpu-index"] == device_idx]
        timestamps = (np.array(device.timestamp) - np.array(device.timestamp)[0]) / np.timedelta64(1, "s")
        current_ax.plot(timestamps, device.power)
        current_ax.set_ylim(0,310)

        for i, epoch_begin, _ in power_data.t_info.epochs():
            epoch_ts = (epoch_begin - np.array(device.timestamp)[0]) / np.timedelta64(1, "s")
            current_ax.axvline(x=epoch_ts,color='orange',linestyle='--')
    current_ax.set_ylabel("Power [W]")
    current_ax.set_xlabel("Time [s]")
    fig.suptitle("GPU Power vs. Time  w/ Power Limits [150, 200 ,250, 300]")
    #plt.tight_layout() 
    plt.show()


plot_power_raw(mnist_data_big, devices[0])
# %%

def plot_cum_energy(df, net):
    runs = df.groupby("run")
    for run_idx, run in runs:
        pl_list =[]
        for pl_idx, pl in run.groupby("power_cap"):
            pl = pl.iloc[0]
            pl_list.append(pl_idx)
            epoch2_begin = pl.power_data.t_info.get_epoch_begin(1)

            plt.plot(pl.cum_energy/MEGA)
            #plt.plot((epoch2_begin-pl.power_data.power_gpu.iloc[0].timestamp)/np.timedelta64(1,"s"),0,"x")
        plt.legend([f"{x}W" for x in pl_list])
        plt.title(f"[{net}]Cumulative Energy w/ Power Limits (Run {run_idx})")
        plt.xlabel("Time [$s$]")
        plt.ylabel("Energy [$MJ$]")
        plt.show()

plot_cum_energy(mnist_data_big)

# %%
def epoch_times_hist(df):
    times = {}
    for pl_idx, pl in df.groupby("power_cap"):
        times[pl_idx] = []
        for run_idx, run in pl.groupby("run"):
            run = run.iloc[0]
            times[pl_idx].extend(run.epoch_times[1:])
            
        plt.hist(times[pl_idx], bins=30)
        plt.title(f"[{net}]Distribution of Time per Epoch ({pl_idx}W)")
        plt.show()


epoch_times_hist(mnist_data_big)


# %%
def epoch_times_boxplot(df):
    times = {}
    for pl_idx, pl in df.groupby("power_cap"):
        times[pl_idx] = []
        for run_idx, run in pl.groupby("run"):
            run = run.iloc[0]
            times[pl_idx].extend(run.epoch_times[1:])
            
    plt.boxplot(times.values())
    plt.xticks(*zip(*enumerate([f"{x}W" for x in times.keys()],1)))
    plt.title(f"[{net}]Time per Epoch")
    plt.ylabel("Time [s]")
    plt.show()


epoch_times_boxplot(mnist_data_big)



# %%
def epoch_energy_hist(df):
    times = {}
    for pl_idx, pl in df.groupby("power_cap"):
        times[pl_idx] = []
        for run_idx, run in pl.groupby("run"):
            run = run.iloc[0]
            times[pl_idx].extend(run.epoch_energy[1:])
            
        plt.hist(times[pl_idx], bins=30)
        plt.title(f"[{net}]Distribution of Energy per Epoch ({pl_idx}W)")
        plt.show()


epoch_energy_hist(mnist_data_big)


# %%
def epoch_energy_boxplot(df):
    times = {}
    for pl_idx, pl in df.groupby("power_cap"):
        times[pl_idx] = []
        for run_idx, run in pl.groupby("run"):
            run = run.iloc[0]
            times[pl_idx].extend(run.epoch_energy[1:])
            
    plt.boxplot(times.values())
    plt.xticks(*zip(*enumerate([f"{x}W" for x in times.keys()],1)))
    plt.title(f"[{net}]Energy per Epoch")
    plt.ylabel("Energy [J]")
    plt.show()


epoch_energy_boxplot(mnist_data_big)

#%%
def plot_mean_total_energy(df):
    energy = {}
    for pl_idx, pl in df.groupby("power_cap"):
        energy[pl_idx] = []
        for run_idx, run in pl.groupby("run"):
            run = run.iloc[0]
            energy[pl_idx].append(run.total_energy)
            
    plt.plot([x for x in energy.keys()], np.array([np.array(x).mean() for x  in energy.values()]) / MEGA, "x")
    plt.title(f"[{net}]Mean Total Energy vs. Power Limit")
    plt.xlabel("Power Limit [W]")
    plt.ylabel("Energy [MJ]")
    plt.xticks(power_caps)
    plt.show()


plot_mean_total_energy(mnist_data_big)


#%%
def plot_mean_edp(df):
    energy = {}
    for pl_idx, pl in df.groupby("power_cap"):
        energy[pl_idx] = []
        for run_idx, run in pl.groupby("run"):
            run = run.iloc[0]
            energy[pl_idx].append(run.edp)
            
    plt.plot([x for x in energy.keys()], np.array([np.array(x).mean() for x  in energy.values()]) / MEGA, "x")
    plt.title(f"[{net}]Mean EDP vs. Power Limit")
    plt.xlabel("Power Limit [W]")
    plt.ylabel("Energy Delay Product [MJs]")
    plt.xticks(power_caps)
    plt.show()


plot_mean_edp(mnist_data_big)


#%%
def plot_mean_total_time(df):
    energy = {}
    for pl_idx, pl in df.groupby("power_cap"):
        energy[pl_idx] = []
        for run_idx, run in pl.groupby("run"):
            run = run.iloc[0]
            energy[pl_idx].append(run.total_time)
            
    plt.plot([x for x in energy.keys()], np.array([np.array(x).mean() for x  in energy.values()]), "x")
    plt.title(f"[{net}]Mean Total Time vs. Power Limit")
    plt.xlabel("Power Limit [W]")
    plt.ylabel("Time [s]")
    plt.xticks(power_caps)
    plt.show()


plot_mean_total_time(mnist_data_big)
#%%
# acc = 0
# for dev in devices:
#     dev_data = df[df.]
#     acc +=  get_cumulative_energy()
# tmp = calculate_total_cumulative_energy(df.power_data, devices=devices
#                     ,start=power_data.t_info.get_train_begin()
#                     , end=power_data.t_info.get_train_end())[0]
        
#cum_energy = get_cumulative_energy(gpu0.power, gpu0.timestamp)
#plt.plot(gpu0.timestamp,cum_energy)

# %%
