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
import subprocess

# %%
#subprocess.run(["nvidia-smi", "-q","-x"],capture_output=True).stdout.decode("utf-8")


# %%

# %%
import nvidia_smi

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Parameters
x_len = 200         # Number of points to display
y_range = [10, 40]  # Range of possible Y values to display
temp_c = 0

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = list(range(0, 200))
ys = [0] * x_len
ax.set_ylim([0,10000000])


# Create a blank line. We will update the line in animate
line, = ax.plot(xs, ys)

# Add labels
plt.title('TMP102 Temperature over Time')
plt.xlabel('Samples')
plt.ylabel('Temperature (deg C)')

def get_data():
    i=15
    while True:
        i+=10
        yield i
        
# This function is called periodically from FuncAnimation
def animate(i, ys):
    #global temp_c
    # Read temperature (Celsius) from TMP102
    temp_c = nvidia_smi.nvmlDeviceGetPowerUsage(handle)/1000

    # Add y to list
    ys.append(temp_c+ys[-1])

    # Limit y list to set number of items
    ys = ys[-x_len:]

    # Update line with new Y values
    line.set_ydata(ys)

    return line,

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig,
    animate, #frames=100,
    fargs=(ys,),
    interval=100,
    blit=True)
plt.show()
