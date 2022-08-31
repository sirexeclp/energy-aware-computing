#%load_ext autoreload
#%autoreload 2

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
            #assert power_data.t_info.get_epoch_count() == epoch_count, f"Unexpected Number of epochs! Expected: {epoch_count}, but got: {power_data.t_info.get_epoch_count()} in {data_path}"
            #epoch_energy = get_energy_per_epoch(power_data, devices=devices)
            #energy = np.mean(epoch_energy[1:])
           
            #epoch_times = power_data.t_info.get_time_per_epoch()
            #mean_time = np.mean(epoch_times[1:])
            print(description)
            
            total_time = power_data.t_info.total_experiment_duration() / np.timedelta64(1, "s")
            total_energy = calculate_total_energy(power_data, devices=devices
                                                  , start=power_data.t_info.experiment_begin()
                                                  , end=power_data.t_info.experiment_end())
            
            edp = total_energy * total_time

            cum_energy = calculate_total_cumulative_energy(power_data, devices=devices
                                                           , start=power_data.t_info.experiment_begin()
                                                           , end=power_data.t_info.experiment_end())[0]

            tmp = {
                "power_cap":cap
                ,"total_time":total_time
                ,"total_energy":total_energy
                ,"cum_energy": cum_energy
                ,"edp": edp
                ,"run": str(run).split("/")[-1].replace("run","")
                ,"power_data": power_data
            }
            all_data.append(tmp)
    return all_data

# %%
def plot_all(df, net):
    plot_power_raw(df, devices[0], net)
    plt.clf()
    plot_cum_energy(df,net)
    plt.clf()
    # epoch_times_boxplot(df, net)
    # plt.clf()
    # epoch_energy_boxplot(df, net)
    # plt.clf()
    plot_mean_total_energy(df, net)
    plt.clf()
    plot_mean_edp(df, net)
    plt.clf()
    plot_mean_total_time(df, net)



# %%
def plot_power_raw(df, device_idx, net):
    fig, ax = plt.subplots(len(power_caps), 1, sharex=True)
    for index,(_, pl) in enumerate(df[df._run == "1"].iterrows()):
        current_ax = ax[index]
        power_data = pl["power_data"]
        device = power_data.power_gpu[power_data.power_gpu["gpu-index"] == device_idx]
        timestamps = (np.array(device.timestamp) - np.array(device.timestamp)[0]) / np.timedelta64(1, "s")
        current_ax.plot(timestamps, device.power)
        current_ax.set_ylim(0,310)

        # for i, epoch_begin, _ in power_data.t_info.epochs():
        #     epoch_ts = (epoch_begin - np.array(device.timestamp)[0]) / np.timedelta64(1, "s")
        #     current_ax.axvline(x=epoch_ts,color='orange',linestyle='--')
    current_ax.set_ylabel("Power [W]")
    current_ax.set_xlabel("Time [s]")
    fig.suptitle(f"[{net}]GPU Power vs. Time  w/ Power Limits [150, 200 ,250, 300]")
    #plt.tight_layout() 
    plt.show()
    plt.savefig(fig_root/f"{net}-power-raw.png")


#plot_power_raw(mnist_data_big, devices[0], "mnist-big")
# %%

def plot_cum_energy(df, net):
    runs = df.groupby("run")
    for run_idx, run in runs:
        pl_list =[]
        for pl_idx, pl in run.groupby("power_cap"):
            pl = pl.iloc[0]
            pl_list.append(pl_idx)
            #epoch2_begin = pl.power_data.t_info.get_epoch_begin(1)

            plt.plot(pl.cum_energy/MEGA)
            #plt.plot((epoch2_begin-pl.power_data.power_gpu.iloc[0].timestamp)/np.timedelta64(1,"s"),0,"x")
        plt.legend([f"{x}W" for x in pl_list])
        plt.title(f"[{net}]Cumulative Energy w/ Power Limits (Run {run_idx})")
        plt.xlabel("Time [$s$]")
        plt.ylabel("Energy [$MJ$]")
        plt.show()
        plt.savefig(fig_root/f"{net}-cum-energy-{run_idx}.png")
        plt.clf()

#plot_cum_energy(mnist_data_big)

# %%
def epoch_times_hist(df, net):
    times = {}
    for pl_idx, pl in df.groupby("power_cap"):
        times[pl_idx] = []
        for run_idx, run in pl.groupby("run"):
            run = run.iloc[0]
            times[pl_idx].extend(run.epoch_times[1:])
            
        plt.hist(times[pl_idx], bins=30)
        plt.title(f"[{net}]Distribution of Time per Epoch ({pl_idx}W)")
        plt.show()
        plt.savefig(fig_root/f"{net}-epoch-times-hist-{pl_idx}.png")


#epoch_times_hist(mnist_data_big)


# %%
def epoch_times_boxplot(df, net):
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
    plt.savefig(fig_root/f"{net}-epoch-times-boxplot.png")


#epoch_times_boxplot(mnist_data_big)



# %%
def epoch_energy_hist(df, net):
    times = {}
    for pl_idx, pl in df.groupby("power_cap"):
        times[pl_idx] = []
        for run_idx, run in pl.groupby("run"):
            run = run.iloc[0]
            times[pl_idx].extend(run.epoch_energy[1:])
            
        plt.hist(times[pl_idx], bins=30)
        plt.title(f"[{net}]Distribution of Energy per Epoch ({pl_idx}W)")
        plt.show()
        plt.savefig(fig_root/f"{net}-epoch-energy-hist-{pl_idx}.png")


#epoch_energy_hist(mnist_data_big)


# %%
def epoch_energy_boxplot(df, net):
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
    plt.savefig(fig_root/f"{net}-epoch-energy-boxplot.png")


#epoch_energy_boxplot(mnist_data_big)

# %%
def plot_mean_metric(df, metric, net, scale):
    energy = {}
    for pl_idx, pl in df.groupby("power_cap"):
        energy[pl_idx] = []
        for run_idx, run in pl.groupby("run"):
            run = run.iloc[0]
            energy[pl_idx].append(run[metric])
    
    plt.plot([x for x in energy.keys()], np.array([np.array(x).mean() for x  in energy.values()]) / scale, "x")
    plt.xlabel("Power Limit [W]")
    plt.xticks(power_caps)

# %%
def plot_mean_total_energy(df, net):
    plot_mean_metric(df, "total_energy", net, MEGA)
    plt.title(f"[{net}]Mean Total Energy vs. Power Limit")
    plt.ylabel("Energy [MJ]")
    plt.show()
    plt.savefig(fig_root/f"{net}-mean-total-energy.png")


#plot_mean_total_energy(mnist_data_big, "MNIST-Big")


#%%
def plot_mean_edp(df, net):
    plot_mean_metric(df, "edp", net, MEGA)
    plt.title(f"[{net}]Mean EDP vs. Power Limit")
    plt.ylabel("Energy Delay Product [MJs]")
    plt.show()
    plt.savefig(fig_root/f"{net}-mean-edp.png")

#plot_mean_edp(mnist_data_big)


#%%
def plot_mean_total_time(df, net):
    plot_mean_metric(df, "total_time", net, 1)        
    plt.title(f"[{net}]Mean Total Time vs. Power Limit")
    plt.ylabel("Time [s]")
    plt.show()
    plt.savefig(fig_root/f"{net}-mean-total-time.png")


import matplotlib
matplotlib.use('Agg')
fig_root = Path("../report/fig")
fig_root.mkdir(parents=True, exist_ok=True)

# mnist_data_big = load_power_cap_data("../data/data-1.2", "mnist-big", epoch_count = 10)
# mnist_data_big = pd.DataFrame(mnist_data_big)
# plot_all(mnist_data_big, "mnist-big")


# mnist_data_small = load_power_cap_data("../data/data-1.2", "mnist-2", epoch_count = 10)
# mnist_data_small = pd.DataFrame(mnist_data_small)
# plot_all(mnist_data_small, "mnist-small")

ecg_data = load_power_cap_data("../data/data-1.3.1", "ecg", epoch_count = 5)
ecg_data = pd.DataFrame(ecg_data)
plot_all(ecg_data, "ecg")