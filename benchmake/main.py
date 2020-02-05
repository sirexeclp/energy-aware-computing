#! /usr/bin/python3 main.py
import time
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
# print(sys.argv[1:])
# assert len(sys.argv)>1, "missing arguments"

def set_boxplot_legend(legend):
    plt.xticks(*zip(*enumerate(legend,1)))

config = {
    "name": "ffts_bench_small_n"
    ,"runs":[
        # {"args":["../fft/dft", "dft" ,"10","none","none"]},
        {"args":["../fft/dft", "fft" ,"17","file","file"]}
        ,{"args":["../fft/dft", "fftw" ,"17","file","file"]}
    ]
}

reppetitions = 20
warmup = 2

runs = []
for run in config["runs"]:
    timings = []
    print(run["args"])
    for i in tqdm(range(reppetitions+warmup)):
        start = time.time()
        tmp = subprocess.run(run["args"], stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,cwd="../fft")
        tqdm.write(tmp.stderr.decode("UTF-8"))
        end = time.time()
        timings.append(end-start)
    runs.append(timings)

algorithms = [run["args"][1] for run in config["runs"]]
df = pd.DataFrame(runs, algorithms).transpose()
df.to_csv(config["name"]+".csv")

for timings,run in zip(runs,config["runs"]):
    warm_timings = timings[warmup:]
    print(run["args"])
    print(f"Mean: {np.mean(warm_timings)} SD:{np.std(warm_timings)}")
    plt.plot(warm_timings)
plt.title("Warm Timings")
plt.legend(algorithms)
plt.show()
plt.boxplot(runs)
set_boxplot_legend(algorithms)
plt.show()

# plt.plot(timings)
# plt.title("ALL Timings")
# plt.show()