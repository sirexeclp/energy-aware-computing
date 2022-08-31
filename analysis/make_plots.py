# %%
# #%load_ext autoreload
# #%autoreload 2

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl
from pathlib import Path
from datetime import datetime
import pint

unit = pint.UnitRegistry()
mpl.rcParams["figure.dpi"] = 300
# plt.style.use('seaborn')
from util import *

# %%
GPU1 = 1
GPU0 = 0
GPU3 = 3
devices = [GPU3]
# devices=[GPU0]
power_caps = [150, 200, 250, 300]
clocks = [135, 285, 435, 585, 735, 885, 1035, 1185, 1335, 1485]


# clocks = [(877,x) for x in clocks]
# power_caps = clocks
# %%
def load_power_cap_data(data_root, network, epoch_count):
    data_root = Path(data_root)
    runs = list(data_root.glob("run*"))

    all_data = []

    for cap in power_caps:
        for run in runs:
            # description = f"clocks(*{cap}*)-{network}"
            description = f"powercap{cap}-{network}"
            # print(description)
            data_path = list(run.glob(f"{description}*"))[0]
            power_data = PowerData.load(data_path)
            assert (
                power_data.t_info.epoch_count == epoch_count
            ), f"Unexpected Number of epochs! Expected: {epoch_count}, but got: {power_data.t_info.epoch_count()} in {data_path}"
            epoch_energy = get_energy_per_epoch(power_data, devices=devices)
            energy = np.mean(epoch_energy[1:])

            epoch_times = power_data.t_info.time_per_epoch
            mean_time = np.mean(epoch_times[1:])

            total_time = power_data.t_info.total_experiment_duration / np.timedelta64(
                1, "s"
            )
            total_energy = calculate_total_energy(
                power_data,
                devices=devices,
                start=power_data.t_info.train_begin,
                end=power_data.t_info.train_end,
            )

            edp = total_energy * total_time

            cum_energy = calculate_total_cumulative_energy(
                power_data,
                devices=devices,
                start=power_data.t_info.train_begin,
                end=power_data.t_info.train_end,
            )[0]

            tmp = {
                "power_cap": cap,
                "energy": energy,
                "epoch_energy": epoch_energy,
                "epoch_edp": epoch_energy / epoch_times,
                "mean_time": mean_time,
                "total_time": total_time,
                "epoch_times": epoch_times,
                "total_energy": total_energy,
                "cum_energy": cum_energy,
                "edp": edp,
                "run": str(run).split("/")[-1].replace("run", ""),
                "power_data": power_data,
            }
            all_data.append(tmp)
    return all_data


# %%
def plot_all(df, net):
    # plt.clf()
    # plot_power_raw(df, devices[0], net)
    # plt.clf()
    # plot_cum_energy(df,net)
    # plt.clf()
    epoch_edp_boxplot(df, net)
    plt.clf()

    # epoch_times_boxplot(df, net)
    # plt.clf()
    # epoch_energy_boxplot(df, net)
    # plt.clf()

    # plot_mean_total_energy(df, net)
    # plt.clf()
    # plot_mean_edp(df, net)
    # plt.clf()
    # plot_mean_total_time(df, net)
    # plt.clf()


# %%
def plot_power_raw(df, device_idx, net):
    fig, ax = plt.subplots(len(power_caps), 1, sharex=True)
    for index, (_, pl) in enumerate(df[df._run == "0"].iterrows()):
        current_ax = ax[index]
        power_data = pl["power_data"]
        device = power_data.power_gpu[power_data.power_gpu["gpu-index"] == device_idx]
        timestamps = (
            np.array(device.timestamp) - np.array(device.timestamp)[0]
        ) / np.timedelta64(1, "s")
        current_ax.plot(timestamps, device.power)
        current_ax.set_ylim(0, 310)

        for i, epoch_begin, _ in power_data.t_info.iter_epochs():
            epoch_ts = (epoch_begin - np.array(device.timestamp)[0]) / np.timedelta64(
                1, "s"
            )
            current_ax.axvline(x=epoch_ts, color="orange", linestyle="--")
    current_ax.set_ylabel("Power [W]")
    current_ax.set_xlabel("Time [s]")
    fig.suptitle(f"[{net}]GPU Power vs. Time  w/ Power Limits [150, 200 ,250, 300]")
    # plt.tight_layout()
    plt.show()
    plt.savefig(fig_root / f"{net}-power-raw.pdf")


# plot_power_raw(mnist_data_big, devices[0], "mnist-big")
# %%


def plot_cum_energy(df, net):
    # runs = df.groupby("run")
    pl_list = []
    ax = plt.gca()
    for pl_idx, pl in df.groupby("power_cap"):
        color = next(ax._get_lines.prop_cycler)["color"]

        pl_list.append(pl_idx)
        results = list(pl.groupby("run").apply(lambda x: x.cum_energy))
        x, y = zip(*results)
        # pl[pl.run == 0].cum_energy
        results = pad_n_mask(y)
        plot_mean_std(
            np.arange(0, results.shape[1] // 4, 0.25),
            results,
            color=color,
            label=f"{pl_idx}W",
        )
        # x,y = pl[pl.run == "0"].cum_energy.iloc[0]
        # plt.plot(y)
        # print(len(pl[pl.run == "0"].cum_energy.iloc[0]))

        max_x = max([x_[-1] for x_ in x])
        max_y = max([y_[-1] for y_ in y]) / KILO
        plt.axvline(x=max_x, linestyle=":", color=color)
        # plt.axhline(y=max_y,linestyle=':',color=color)

        # if pl_idx == 200:
        #     break
        # for run_idx, run in pl.groupby("run"):
        #     run = run.iloc[0]
        #     results.append(pl.cum_energy)
        #     timestamps, cum_energy = pl.cum_energy
        #     cum_energy = cum_energy / 1000
        #     pad_n_mask

        #     plt.plot(pl.cum_energy/MEGA)
        #     #plt.plot((epoch2_begin-pl.power_data.power_gpu.iloc[0].timestamp)/np.timedelta64(1,"s"),0,"x")
    plt.legend()  # [f"{x}W" for x in pl_list])
    plt.title(f"[{net}]Cumulative Energy w/ Power Limits")
    plt.xlabel("Time [$s$]")
    plt.ylabel("Energy [$kJ$]")
    plt.show()
    plt.savefig(fig_root / f"{net}-cum-energy.pdf", bbox_inches="tight")
    plt.clf()


# plot_cum_energy(mnist_data_big)

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
        plt.savefig(fig_root / f"{net}-epoch-times-hist-{pl_idx}.pdf")


# epoch_times_hist(mnist_data_big)


# %%
def epoch_times_boxplot(df, net):
    times = {}
    for pl_idx, pl in df.groupby("power_cap"):
        times[pl_idx] = []
        for run_idx, run in pl.groupby("run"):
            run = run.iloc[0]
            times[pl_idx].extend(run.epoch_times[1:])

    plt.boxplot(times.values())
    plt.xticks(*zip(*enumerate([f"{x}W" for x in times.keys()], 1)))
    plt.title(f"[{net}]Time per Epoch")
    plt.ylabel("Time [s]")

    # regression
    x = np.array([x for x in times.keys()])
    y = np.array([np.array(x).mean() for x in times.values()])  # / scale
    from sklearn.linear_model import LinearRegression

    reg = np.poly1d(np.polyfit(x, y, 2))

    y_hat = reg(x)

    y_bar = np.mean(y)
    ssreg = np.sum((y_hat - y_bar) ** 2)
    sstot = np.sum((y - y_bar) ** 2)
    r2 = ssreg / sstot

    step = 100
    x_hat = np.linspace(min(x), max(x), step)
    y_hat = reg(x_hat)
    plt.plot([], [], " ", label=f"$R^2={r2:.3f}$")
    plt.legend()
    # label="second degree least squares polynomial fit"
    plt.plot(np.linspace(1, len(x), step), y_hat, "--", alpha=0.7)

    plt.show()
    plt.savefig(fig_root / f"{net}-epoch-times-boxplot.pdf")


# epoch_times_boxplot(mnist_data_big)


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
        plt.savefig(fig_root / f"{net}-epoch-energy-hist-{pl_idx}.pdf")


# epoch_energy_hist(mnist_data_big)


def total_relative_energy(df, net):
    times = {}
    for pl_idx, pl in df.groupby("power_cap"):
        times[pl_idx] = []
        for run_idx, run in pl.groupby("run"):
            run = run.iloc[0]
            times[pl_idx].extend(run.total_energy)

    plt.boxplot(times.values())
    plt.xticks(*zip(*enumerate([f"{x}W" for x in times.keys()], 1)))
    plt.title(f"[{net}]Energy per Epoch")
    plt.ylabel("Energy [J]")

    # regression
    x = np.array([x for x in times.keys()])
    y = np.array([np.array(x).mean() for x in times.values()])  # / scale
    from sklearn.linear_model import LinearRegression

    reg = np.poly1d(np.polyfit(x, y, 2))

    y_hat = reg(x)

    y_bar = np.mean(y)
    ssreg = np.sum((y_hat - y_bar) ** 2)
    sstot = np.sum((y - y_bar) ** 2)
    r2 = ssreg / sstot

    step = 100
    x_hat = np.linspace(min(x), max(x), step)
    y_hat = reg(x_hat)
    plt.plot([], [], " ", label=f"$R^2={r2:.3f}$")
    plt.legend()
    # label="second degree least squares polynomial fit"
    plt.plot(np.linspace(1, len(x), step), y_hat, "--", alpha=0.7)

    plt.show()
    plt.savefig(fig_root / f"{net}-epoch-energy-boxplot.pdf")


# %%
def epoch_energy_boxplot(df, net):
    times = {}
    for pl_idx, pl in df.groupby("power_cap"):
        times[pl_idx] = []
        for run_idx, run in pl.groupby("run"):
            run = run.iloc[0]
            times[pl_idx].extend(run.epoch_energy[1:])

    plt.boxplot(times.values())
    plt.xticks(*zip(*enumerate([f"{x}W" for x in times.keys()], 1)))
    plt.title(f"[{net}]Energy per Epoch")
    plt.ylabel("Energy [J]")

    # regression
    x = np.array([x for x in times.keys()])
    y = np.array([np.array(x).mean() for x in times.values()])  # / scale
    from sklearn.linear_model import LinearRegression

    reg = np.poly1d(np.polyfit(x, y, 2))

    y_hat = reg(x)

    y_bar = np.mean(y)
    ssreg = np.sum((y_hat - y_bar) ** 2)
    sstot = np.sum((y - y_bar) ** 2)
    r2 = ssreg / sstot

    step = 100
    x_hat = np.linspace(min(x), max(x), step)
    y_hat = reg(x_hat)
    plt.plot([], [], " ", label=f"$R^2={r2:.3f}$")
    plt.legend()
    # label="second degree least squares polynomial fit"
    plt.plot(np.linspace(1, len(x), step), y_hat, "--", alpha=0.7)

    plt.show()
    plt.savefig(fig_root / f"{net}-epoch-energy-boxplot.pdf")


def epoch_edp_boxplot(df, net):
    times = {}
    for pl_idx, pl in df.groupby("power_cap"):
        times[pl_idx] = []
        for run_idx, run in pl.groupby("run"):
            run = run.iloc[0]
            times[pl_idx].extend(run.epoch_edp[1:])

    plt.boxplot(times.values())
    plt.xticks(*zip(*enumerate([f"{x}W" for x in times.keys()], 1)))
    plt.title(f"[{net}]EDP per Epoch")
    plt.ylabel("EDP [Js]")

    # regression
    x = np.array([x for x in times.keys()])
    y = np.array([np.array(x).mean() for x in times.values()])  # / scale
    from sklearn.linear_model import LinearRegression

    reg = np.poly1d(np.polyfit(x, y, 2))

    y_hat = reg(x)

    y_bar = np.mean(y)
    ssreg = np.sum((y_hat - y_bar) ** 2)
    sstot = np.sum((y - y_bar) ** 2)
    r2 = ssreg / sstot

    step = 100
    x_hat = np.linspace(min(x), max(x), step)
    y_hat = reg(x_hat)
    plt.plot([], [], " ", label=f"$R^2={r2:.3f}$")
    plt.legend()
    # label="second degree least squares polynomial fit"
    plt.plot(np.linspace(1, len(x), step), y_hat, "--", alpha=0.7)

    plt.show()
    plt.savefig(fig_root / f"{net}-epoch-edp-boxplot.pdf")


# epoch_energy_boxplot(mnist_data_big)

# %%
def plot_mean_metric(df, metric, net, scale):
    energy = {}
    for pl_idx, pl in df.groupby("power_cap"):
        energy[pl_idx] = []
        for run_idx, run in pl.groupby("run"):
            run = run.iloc[0]
            energy[pl_idx].append(run[metric])

    x = np.array([x for x in energy.keys()])
    y = np.array([np.array(x).mean() for x in energy.values()]) / scale
    from sklearn.linear_model import LinearRegression

    reg = np.poly1d(np.polyfit(x, y, 2))

    y_hat = reg(x)

    y_bar = np.mean(y)
    ssreg = np.sum((y_hat - y_bar) ** 2)
    sstot = np.sum((y - y_bar) ** 2)
    r2 = ssreg / sstot

    step = 5
    x_hat = np.arange(min(x), max(x) + step, step)
    y_hat = reg(x_hat)

    # label="second degree least squares polynomial fit"
    plt.plot(x_hat, y_hat, "--", alpha=0.7)
    plt.plot(x, y, "x")
    plt.xlabel("Power Limit [W]")
    plt.plot([], [], " ", label=f"$R^2={r2:.3f}$")
    plt.legend()
    plt.xticks(power_caps)


# %%
def plot_mean_total_energy(df, net):
    plot_mean_metric(df, "total_energy", net, MEGA)
    plt.title(f"[{net}]Mean Total Energy vs. Power Limit")
    plt.ylabel("Energy [MJ]")
    plt.show()
    plt.savefig(fig_root / f"{net}-mean-total-energy.pdf")


# plot_mean_total_energy(mnist_data_big, "MNIST-Big")


# %%
def plot_mean_edp(df, net):
    plot_mean_metric(df, "edp", net, MEGA)
    plt.title(f"[{net}]Mean EDP vs. Power Limit")
    plt.ylabel("Energy Delay Product [MJs]")
    plt.show()
    plt.savefig(fig_root / f"{net}-mean-edp.pdf")


# plot_mean_edp(mnist_data_big)


# %%
def plot_mean_total_time(df, net):
    plot_mean_metric(df, "total_time", net, 1)
    plt.title(f"[{net}]Mean Total Time vs. Power Limit")
    plt.ylabel("Time [s]")
    plt.show()
    plt.savefig(fig_root / f"{net}-mean-total-time.pdf")


def build_empty_plots(count, subplots):
    return [plt.subplots(subplots, 1, sharex=True) for x in range(count)]


# # def make_raw_plot(df, device_idx, net):
# #     plots =
# #     for idx, run in df.groupby("run"):
# #         fig, ax =
# #         for index, (_,pl) in enumerate(run.groupby("power_cap")):
# #             current_ax = ax[index]
# #             power_data = pl.iloc[0]["power_data"]
# #             device = power_data.power_gpu[power_data.power_gpu["gpu-index"] == device_idx]
# #             timestamps = np.array(device.timestamp)
# #             timestamps = (timestamps - timestamps[0]) / np.timedelta64(1, "s")
# #             epochs = power_data.t_info.epochs()

# #             for i, epoch_begin, _ in epochs:
# #                 epoch_ts = epoch_begin / np.timedelta64(1, "s") - timestamps[0]
# #                 current_ax.axvline(x=epoch_ts,color='green',linestyle='--')

# #             a, = current_ax.plot(timestamps, device["pci-tx"])
# #             b, = current_ax.plot(timestamps, device["pci-rx"])

# #             #current_ax.plot(timestamps, device["clock-sm"])
# #             #current_ax.plot(timestamps, device.power/250)
# #             #current_ax.set_ylim(35,60)
# #             #current_ax.set_ylim(1000,1550)
# #             current_ax.legend([a,b],["tx","rx"])

# #         current_ax.set_ylabel("Power [W]")
# #         current_ax.set_xlabel("Time [s]")
# #         fig.suptitle(f"[{net}]GPU Power vs. Time  w/ Power Limits [150, 200 ,250, 300]")
# #         #plt.tight_layout()
# #         plots.append((fig, ax))
# #         #plt.show()
# #     #plt.savefig(fig_root/f"{net}-power-raw.pdf")
# #     return plots

# def plot_epoch_markers(epochs, ax, timestamps):


def plot_pci_raw(df, device_idx, net):
    plots = []
    for idx, run in df.groupby("run"):
        fig, ax = plt.subplots(len(power_caps), 1, sharex=True)
        for index, (_, pl) in enumerate(run.groupby("power_cap")):
            current_ax = ax[index]
            power_data = pl.iloc[0]["power_data"]
            device = power_data.power_gpu[
                power_data.power_gpu["gpu-index"] == device_idx
            ]
            timestamps = (
                np.array(device.timestamp) - np.array(device.timestamp)[0]
            ) / np.timedelta64(1, "s")

            for i, epoch_begin, _ in power_data.t_info.iter_epochs():
                epoch_ts = (
                    epoch_begin - np.array(device.timestamp)[0]
                ) / np.timedelta64(1, "s")
                current_ax.axvline(x=epoch_ts, color="green", linestyle="--")

            (a,) = current_ax.plot(timestamps, device["pci-tx"])
            (b,) = current_ax.plot(timestamps, device["pci-rx"])

            # current_ax.plot(timestamps, device["clock-sm"])
            # current_ax.plot(timestamps, device.power/250)
            # current_ax.set_ylim(35,60)
            # current_ax.set_ylim(1000,1550)
            current_ax.legend([a, b], ["tx", "rx"])

        current_ax.set_ylabel("Power [W]")
        current_ax.set_xlabel("Time [s]")
        fig.suptitle(f"[{net}]GPU Power vs. Time  w/ Power Limits [150, 200 ,250, 300]")
        # plt.tight_layout()
        plots.append((fig, ax))
        # plt.show()
    # plt.savefig(fig_root/f"{net}-power-raw.pdf")
    return plots


def plot_clock_raw(df, device_idx, net):
    plots = []
    for idx, run in df.groupby("run"):
        fig, ax = plt.subplots(len(power_caps), 1, sharex=True)
        for index, (_, pl) in enumerate(run.groupby("power_cap")):
            current_ax = ax[index]
            power_data = pl.iloc[0]["power_data"]
            device = power_data.power_gpu[
                power_data.power_gpu["gpu-index"] == device_idx
            ]
            timestamps = (
                np.array(device.timestamp) - np.array(device.timestamp)[0]
            ) / np.timedelta64(1, "s")

            for i, epoch_begin, _ in power_data.t_info.iter_epochs():
                epoch_ts = (
                    epoch_begin - np.array(device.timestamp)[0]
                ) / np.timedelta64(1, "s")
                current_ax.axvline(x=epoch_ts, color="green", linestyle="--")

            current_ax.plot(timestamps, device["clock-sm"])
            # current_ax.plot(timestamps, device.power/250)
            # current_ax.set_ylim(35,60)
            current_ax.set_ylim(1000, 1550)

        current_ax.set_ylabel("Clock Frequency [Hz]")
        current_ax.set_xlabel("Time [s]")
        fig.suptitle(f"[{net}]GPU Clock vs. Time  w/ Power Limits [150, 200 ,250, 300]")
        # plt.tight_layout()
        plots.append((fig, ax))
        # plt.show()
    # plt.savefig(fig_root/f"{net}-power-raw.pdf")
    return plots


def plot_temp_raw(df, device_idx, net):
    plots = []
    for idx, run in df.groupby("run"):
        fig, ax = plt.subplots(len(power_caps), 1, sharex=True)
        for index, (_, pl) in enumerate(run.groupby("power_cap")):
            current_ax = ax[index]
            power_data = pl.iloc[0]["power_data"]
            device = power_data.power_gpu[
                power_data.power_gpu["gpu-index"] == device_idx
            ]
            timestamps = (
                np.array(device.timestamp) - np.array(device.timestamp)[0]
            ) / np.timedelta64(1, "s")

            for i, epoch_begin, _ in power_data.t_info.iter_epochs():
                epoch_ts = (
                    epoch_begin - np.array(device.timestamp)[0]
                ) / np.timedelta64(1, "s")
                current_ax.axvline(x=epoch_ts, color="green", linestyle="--")

            (a,) = current_ax.plot(timestamps, device["tmp"])
            current_ax.set_ylim(35, 60)

        current_ax.set_ylabel("Temp [°C]")
        current_ax.set_xlabel("Time [s]")
        fig.suptitle(f"[{net}]GPU Temp vs. Time  w/ Power Limits [150, 200 ,250, 300]")
        # plt.tight_layout()
        plots.append((fig, ax))
        # plt.show()
    # plt.savefig(fig_root/f"{net}-power-raw.pdf")
    return plots


def plot_power_state_raw(df, device_idx, net):
    plots = []
    for idx, run in df.groupby("run"):
        fig, ax = plt.subplots(len(power_caps), 1, sharex=True)
        for index, (_, pl) in enumerate(run.groupby("power_cap")):
            current_ax = ax[index]
            power_data = pl.iloc[0]["power_data"]
            device = power_data.power_gpu[
                power_data.power_gpu["gpu-index"] == device_idx
            ]
            timestamps = (
                np.array(device.timestamp) - np.array(device.timestamp)[0]
            ) / np.timedelta64(1, "s")

            for i, epoch_begin, _ in power_data.t_info.iter_epochs():
                epoch_ts = (
                    epoch_begin - np.array(device.timestamp)[0]
                ) / np.timedelta64(1, "s")
                current_ax.axvline(x=epoch_ts, color="green", linestyle="--")

            (a,) = current_ax.plot(timestamps, device["power-state"])
            # current_ax.set_ylim(35,60)

        current_ax.set_ylabel("Power-State")
        current_ax.set_xlabel("Time [s]")
        fig.suptitle(
            f"[{net}]GPU Power-State vs. Time  w/ Power Limits [150, 200 ,250, 300]"
        )
        # plt.tight_layout()
        plots.append((fig, ax))
        # plt.show()
    # plt.savefig(fig_root/f"{net}-power-raw.pdf")
    return plots


if __name__ == "__main__":
    import matplotlib

    mpl.rcParams["figure.dpi"] = 150
    mpl.rcParams["figure.figsize"] = 16, 9
    mpl.rcParams["lines.linewidth"] = 3.0
    mpl.rcParams["lines.markeredgewidth"] = 2
    mpl.rcParams["font.size"] = 16
    mpl.rcParams["lines.markersize"] = 10

    plt.style.use("seaborn-colorblind")
    matplotlib.use("Agg")
    fig_root = Path("../report/fig3")
    fig_root.mkdir(parents=True, exist_ok=True)

    # mnist_data_big = load_power_cap_data("../data/data-1.3.1", "mnist-big", epoch_count = 10)
    # # mnist_data_big = load_power_cap_data("../data/8951db562b4334491c8f5b7a35f2f06d72751a8d", "mnist-big", epoch_count = 10)
    # mnist_data_big = pd.DataFrame(mnist_data_big)
    # plot_all(mnist_data_big, "mnist-big")
    # # # plots = plot_temp_raw(mnist_data_big, devices[0], "mnist_big")
    # # # #mpl.rcParams['figure.dpi'] = 300
    # # # for fig, ax in plots:
    # # #     fig.show()

    # # # plt.show()
    # # # plots = plot_pci_raw(mnist_data_big, devices[0], "mnist_big")
    # # # #mpl.rcParams['figure.dpi'] = 300
    # # # for fig, ax in plots:
    # # #     fig.show()
    # # # plt.show()
    # # # plots = plot_clock_raw(mnist_data_big, devices[0], "mnist_big")
    # # # #mpl.rcParams['figure.dpi'] = 300
    # # # for fig, ax in plots:
    # # #     fig.show()
    # # # plt.show()

    # # # plots = plot_power_state_raw(mnist_data_big, devices[0], "mnist_big")
    # # # #mpl.rcParams['figure.dpi'] = 300
    # # # for fig, ax in plots:
    # # #     fig.show()
    # # # plt.show()
    # # # input()
    # # # plt.show()
    # # # plt.show()

    # 45d4f713fd015f50e0b6e5afdd75a4bdde066eb2
    # mnist_data_small = load_power_cap_data("../data/8951db562b4334491c8f5b7a35f2f06d72751a8d", "mnist-2", epoch_count = 10)
    # mnist_data_small = load_power_cap_data("../data/45d4f713fd015f50e0b6e5afdd75a4bdde066eb2", "mnist-2", epoch_count = 10)
    # mnist_data_small = pd.DataFrame(mnist_data_small)
    # plot_all(mnist_data_small, "mnist_data_small")

    # # ecg_data = load_power_cap_data("../data/8951db562b4334491c8f5b7a35f2f06d72751a8d", "ecg", epoch_count = 10)
    # # ecg_data = pd.DataFrame(ecg_data)
    # # plot_all(ecg_data, "ecg_data")

    # # # ecg_data = load_power_cap_data("../data/8951db562b4334491c8f5b7a35f2f06d72751a8d", "ecg", epoch_count = 10)
    # # # ecg_data = pd.DataFrame(ecg_data)
    # # # plot_all(ecg_data, "ecg")

    # new data

    # mnist_data_small = load_power_cap_data("../data/45d4f713fd015f50e0b6e5afdd75a4bdde066eb2", "mnist-2"
    #                                        , epoch_count=10)
    # mnist_data_small = pd.DataFrame(mnist_data_small)
    # plot_all(mnist_data_small, "mnist-small")

    mnist_data_big = load_power_cap_data(
        "../data/45d4f713fd015f50e0b6e5afdd75a4bdde066eb2", "mnist-big", epoch_count=10
    )
    mnist_data_big = pd.DataFrame(mnist_data_big)
    plot_all(mnist_data_big, "mnist-big")

    # ecg_data = load_power_cap_data("../data/413c573c6168b68ffa991f0c64d65d4ec7d01871", "ecg", epoch_count=10)
    # ecg_data = pd.DataFrame(ecg_data)
    # plot_all(ecg_data, "ecg")
