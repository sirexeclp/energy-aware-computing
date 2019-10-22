#! /usr/bin/python3 main.py
import time
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt
print(sys.argv[1:])
assert len(sys.argv)>1, "missing arguments"

timings = []
reppetitions = 100
warmup = 3

for i in range(reppetitions+warmup):
    start = time.time()
    subprocess.run(sys.argv[1:])
    end = time.time()
    timings.append(end-start)

warm_timings = timings[warmup:]

print(f"Mean: {np.mean(warm_timings)} SD:{np.std(warm_timings)}")
plt.plot(warm_timings)
plt.title("Warm Timings")
plt.show()

plt.plot(timings)
plt.title("ALL Timings")
plt.show()