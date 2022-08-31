# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 300

path = "sensorlog.txt.bck"
with open(path) as f:
    file_content = f.readlines()


# +
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


generator = chunks(file_content, 105)
# -

time_steps = list(generator)

df = pd.DataFrame([x.split("|")[0:2] for l in time_steps for x in l])


def process_timestep(data):
    sensor_readings = {}
    for raw_reading in data:
        name, value = [x.strip() for x in raw_reading.split("|")][0:2]
        if "." in value:
            value = float(value)
        elif "0x" in value:
            value = int(value, 16)
        sensor_readings[name] = value
    return sensor_readings


# +
processed_timesteps = [process_timestep(x) for x in time_steps]

df = pd.DataFrame(processed_timesteps)
# -

psus = [f"PSU{x} Input" for x in range(1, 5)]
psus

df[psus].hist()

plt.plot(np.arange(len(df[psus])) / 60, df[psus])

df[psus].describe()

all_data = np.array(df[psus])

np.sum(all_data) / 60 / 12 * 24 / 1000 * 365

207.196721 + 171.024590 + 171.07377 + 171.0

# !sudo pip3 install nvidia-ml-py3

import nvidia_smi as smi
import subprocess
import time
from datetime import datetime

smi.nvmlInit()
smi.nvmlShutdown()

handles = []
num_devices = smi.nvmlDeviceGetCount()
for id in range(num_devices):
    handles.append(smi.nvmlDeviceGetHandleByIndex(id))
print(f"found {num_devices} GPUs")


def read_data():
    tmp = {"timestamp": str(datetime.now())}
    for id, handle in enumerate(handles):
        res = smi.nvmlDeviceGetUtilizationRates(handle)
        # print(smi.nvmlDeviceGetClockInfo(handle,smi.NVML_CLOCK_MEM))
        # print(smi.nvmlDeviceGetApplicationsClock(handle,smi.NVML_CLOCK_MEM))
        # print(smi.nvmlDeviceGetPciInfo(handle))
        # print(smi.nvmlDeviceGetPowerState(handle))
        # print(smi.nvmlDeviceGetComputeRunningProcesses(handle))
        # print(smi.nvmlDeviceGetCurrPcieLinkWidth(handle))

        # print(smi.nvmlDeviceGetPowerUsage(handle))
        # print(smi.nvmlDeviceGetTemperature(handle,smi.NVML_TEMPERATURE_GPU))
        # print(smi.nvmlDeviceGetPcieThroughput(handle,smi.NVML_PCIE_UTIL_TX_BYTES))
        # print(smi.nvmlDeviceGetPcieThroughput(handle,smi.NVML_PCIE_UTIL_RX_BYTES))
        tmp[f"gpu{id}-util"] = res.gpu
        tmp[f"gpu{id}-mem"] = res.memory

        tmp[f"gpu{id}-clock-mem"] = smi.nvmlDeviceGetClockInfo(
            handle, smi.NVML_CLOCK_MEM
        )
        tmp[f"gpu{id}-clock-sm"] = smi.nvmlDeviceGetClockInfo(handle, smi.NVML_CLOCK_SM)

        tmp[f"gpu{id}-power-state"] = smi.nvmlDeviceGetPowerState(handle)

        tmp[f"gpu{id}-power"] = smi.nvmlDeviceGetPowerUsage(handle)
        tmp[f"gpu{id}-tmp"] = smi.nvmlDeviceGetTemperature(
            handle, smi.NVML_TEMPERATURE_GPU
        )
        tmp[f"gpu{id}-pci-tx"] = smi.nvmlDeviceGetPcieThroughput(
            handle, smi.NVML_PCIE_UTIL_TX_BYTES
        )
        tmp[f"gpu{id}-pci-rx"] = smi.nvmlDeviceGetPcieThroughput(
            handle, smi.NVML_PCIE_UTIL_RX_BYTES
        )
        # print(tmp)
        # print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
    return tmp


raise smi.NVMLError_NotSupported()

smi.nvmlDeviceGetPowerManagementLimitConstraints(hdl)

hdl = handles[0]
try:
    smi.nvmlDeviceSetPowerManagementLimit(hdl, 50)
except smi.NVMLError_NoPermission as perm:
    pass
except smi.NVMLError_NotSupported as nosup:
    pass
smi.NVML_ERROR

smi.nvmlDeviceGetPersistenceMode(hdl)

smi.nvmlDeviceSetPersistenceMode(hdl, 1)

# #%timeit
data = []
# data.append(read_data())

# +
def execute_with_output(self, command):
    def sub_routine(cmd):
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
        for stdout_line in iter(popen.stdout.readline, ""):
            yield stdout_line
        popen.stdout.close()
        return_code = popen.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, cmd)

    # for line in sub_routine(command):
    #    print(line, end="")


# -

for i in range(1 * 5):
    data.append(read_data())
    time.sleep(0.5 - 0.0752)
print("warmup-done")
p = subprocess.Popen(["/bin/bash", "./train_cinc17.sh"])  # , stdout=subprocess.PIPE)
while p.poll() is None:
    # for line in iter(p.stdout.readline, ''):  # replace '' with b'' for Python 3
    #    print(line)
    data.append(read_data())
    time.sleep(0.5 - 0.0752)
import pickle
from datetime import datetime

with open(f"data_1_{str(datetime.now())}.pkl", "wb") as f:
    pickle.dump(data, f)

df = pd.read_csv("data_1_2020-01-07 19:25:25.639810.csv")

df = pd.DataFrame(data)

import pandas as pd

df = pd.read_csv("idle_test.csv")

import matplotlib.pyplot as plt

df[df["util-gpu"] > 9]

import os

os.getcwd()

timestamps = pd.read_csv("timestamp_log2020-01-07 19:14:04.076241.csv")

epoch_begin = timestamps[timestamps.event == "epoch_begin"]
epoch_end = timestamps[timestamps.event == "epoch_end"]
batch_begin = timestamps[timestamps.event == "batch_begin"]

timestamps["timestamp"] = pd.to_datetime(timestamps["timestamp"])

df["timestamp"] = pd.to_datetime(df["timestamp"])
df.dtypes

lastxp = df[df.timestamp > pd.to_datetime("2020-01-07 19:00:00")]

lastxp.to_csv("data_1_2020-01-07 19:25:25.639810.csv")

gpu_mem = np.array(lastxp["gpu0-util"])

np.corrcoef([lastxp["gpu0-power"], lastxp["gpu0-util"]], rowvar=True)

np.array(lastxp[["gpu0-util", "gpu0-mem"]])

from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(
    np.array(lastxp[["gpu0-util", "gpu0-mem"]]), lastxp["gpu0-power"]
)

reg.score(np.array(lastxp[["gpu0-util", "gpu0-mem"]]), lastxp["gpu0-power"])

from sklearn.metrics import mean_squared_error

mean_squared_error(
    np.array(lastxp["gpu0-power"]),
    reg.predict(np.array(lastxp[["gpu0-util", "gpu0-mem"]])),
)

plt.plot(
    reg.predict(np.array(lastxp[["gpu0-util", "gpu0-mem"]])), lastxp["gpu0-power"], "x"
)

# ?sig.butter

# +
import scipy.signal as sig

coefs = sig.butter(4, [0.00001, 0.2], "bandpass")

plt.plot(sig.filtfilt(*coefs, np.array(lastxp["gpu0-power"])) / 1000)
plt.plot(
    sig.filtfilt(*coefs, reg.predict(np.array(lastxp[["gpu0-util", "gpu0-mem"]])))
    / 1000
)
plt.title("Training on one GPU -- Power [Predicted|Actual] vs. Time Smoothed")
plt.xlabel("Time [UTC]")
plt.ylabel("Power Consumption [W]")
plt.legend(["actual", "predicted"])
# -

plt.plot(np.array(lastxp["gpu0-power"]) / 1000)
plt.plot(reg.predict(np.array(lastxp[["gpu0-util", "gpu0-mem"]])) / 1000)
plt.title("Training on one GPU -- Power [Predicted|Actual] vs. Time")
plt.xlabel("Time [UTC]")
plt.ylabel("Power Consumption [W]")
plt.legend(["actual", "predicted"])

reg.predict(np.array(lastxp[["gpu0-util", "gpu0-mem"]])[500:502]) / 1000


np.array(lastxp["gpu0-power"])[500:502] / 1000

# plt.plot(np.array(lastxp["timestamp"])[:-1], gpu_mem[1:]-gpu_mem[:-1])
# plt.plot(np.array(lastxp["timestamp"]), lastxp["gpu0-util"])
plt.plot(np.array(lastxp["timestamp"]), lastxp["gpu0-power"] / 1000)
plt.plot(epoch_begin["timestamp"], np.full(len(epoch_begin), 80), "x")
# plt.plot(batch_begin['timestamp'], np.full(len(batch_begin),20),"^")
# plt.plot(df["gpu0-power"][:1000]/3000)
plt.legend(["gpu0-util", "epoch-begin"])
plt.title("Training on one GPU -- Power vs. Time")
plt.xlabel("Time [UTC]")
plt.ylabel("Power Consumption [W]")

# plt.plot(np.array(lastxp["timestamp"])[:-1], gpu_mem[1:]-gpu_mem[:-1])
plt.plot(np.array(lastxp["timestamp"]), lastxp["gpu0-util"])
# plt.plot(np.array(lastxp["timestamp"]), lastxp["gpu0-power"]/1000)
plt.plot(epoch_begin["timestamp"], np.full(len(epoch_begin), 20), "x")
# plt.plot(batch_begin['timestamp'], np.full(len(batch_begin),20),"^")
# plt.plot(df["gpu0-power"][:1000]/3000)
plt.legend(["gpu0-util", "epoch-begin"])
plt.title("Training on one GPU -- Utilization vs. Time")
plt.xlabel("Time [UTC]")
plt.ylabel("GPU Utilization [%]")

epoch_begin["timestamp"][2]

import pickle
from datetime import datetime

with open(f"data_1_{str(datetime.now())}.pkl", "wb") as f:
    pickle.dump(data, f)

# +
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Parameters
x_len = 200  # Number of points to display
y_range = [10, 40]  # Range of possible Y values to display
temp_c = 0

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = list(range(0, 200))
ys = [0] * x_len
ax.set_ylim([0, 10000000])


# Create a blank line. We will update the line in animate
(line,) = ax.plot(xs, ys)

# Add labels
plt.title("TMP102 Temperature over Time")
plt.xlabel("Samples")
plt.ylabel("Temperature (deg C)")


def get_data():
    i = 15
    while True:
        i += 10
        yield i


# This function is called periodically from FuncAnimation
def animate(i, ys):
    # global temp_c
    # Read temperature (Celsius) from TMP102
    temp_c = nvidia_smi.nvmlDeviceGetPowerUsage(handle) / 1000

    # Add y to list
    ys.append(temp_c + ys[-1])

    # Limit y list to set number of items
    ys = ys[-x_len:]

    # Update line with new Y values
    line.set_ydata(ys)

    return (line,)


# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(
    fig, animate, fargs=(ys,), interval=100, blit=True  # frames=100,
)
plt.show()

# -


with open(f"data_1_2020-01-07 19:25:25.639810.pkl", "rb") as f:
    unpack_test = pickle.load(f)

unpack_test

# ?pd.DataFrame.to_csv
