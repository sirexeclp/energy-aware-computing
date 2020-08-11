# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from typing import List, Tuple
import subprocess
import time
from datetime import datetime
from collections import namedtuple
from pathlib import Path
import pandas as pd
from pynvml3 import NVMLLib, Device, PowerLimit, ApplicationClockLimit
from random import shuffle
import copy
import yaml

from system_info import SystemInfo

WARMUP = 5 #warmup in seconds
VISIBLE_DEVICES="3"

def run_experiment(data_path, working_directory, module, args ,baseline = 0, power_limit=None, clocks=None):
    data_path.mkdir(parents=True)

    SMIWrapper.set_power_limit(power_limit)
    SMIWrapper.set_clocks(clocks)
        
    args = ["python3", "-m","gPyJoules", "-d", str(data_path.absolute()),"-v",str(VISIBLE_DEVICES), "-w", str(working_directory), module, "--"] + args
    print(args)
    p = subprocess.Popen(args)
    while p.poll() is None:
        time.sleep(1)


# %%

def run_power_cap_experiment(module, args, working_directory, power_caps, data_root ,description_template):
    data_root = Path(data_root)
    power_caps_shuffled = copy.deepcopy(power_caps)
    shuffle(power_caps_shuffled)
    for p in power_caps_shuffled:
        description = description_template.format(p)
        print(f"[Running] {description}")
        data_path = data_root / Path(f"{description}-{datetime.now().isoformat()}")
        run_experiment(data_path, str(working_directory), module, args, power_limit=p)
        time.sleep(10)

def run_clock_experiment(module, args, working_directory, clocks, data_root ,description_template):
    data_root = Path(data_root)
    clocks_shuffled = copy.deepcopy(clocks)
    shuffle(clocks_shuffled)
    for c in clocks_shuffled:
        description = description_template.format(c)
        print(f"[Running] {description}")
        data_path = data_root / Path(f"{description}-{datetime.now().isoformat()}")
        run_experiment(data_path, str(working_directory), module, args, power_limit=None, clocks=c)
        time.sleep(10)


def run_power_cap_experiment_ecg(data_root, power_caps):
    epochs = 10
    description = "powercap{}-ecg"
    args = ["examples/cinc17/config.json" ,"-e" ,"cinc17", "-n", str(epochs)]
    run_power_cap_experiment("ecg.train", args, "../../ecg/", power_caps, data_root, description)

def run_power_cap_experiment_mnist(data_root, power_caps):
    description = "powercap{}-mnist"
    run_power_cap_experiment("mnist_cnn", ["-c", "2", "-n", "512", "-b", "128"], "./", power_caps, data_root, description)

def run_power_cap_experiment_mnist_big(data_root, power_caps):
    description = "powercap{}-mnist-big"
    run_power_cap_experiment("mnist_cnn", ["-c", "2", "-n", "16384", "-b", "32"], "./", power_caps, data_root, description)


def run_clock_experiment_ecg(data_root, clocks):
    epochs = 10
    description = "clocks{}-ecg"
    args = ["examples/cinc17/config.json" ,"-e" ,"cinc17", "-n", str(epochs)]
    run_clock_experiment("ecg.train", args, "../../ecg/", clocks, data_root, description)

def run_clock_experiment_mnist(data_root, clocks):
    description = "clocks{}-mnist"
    run_clock_experiment("mnist_cnn", ["-c", "2", "-n", "512", "-b", "128"], "./", clocks, data_root, description)

def run_clock_experiment_mnist_big(data_root, clocks):
    description = "clocks{}-mnist-big"
    run_clock_experiment("mnist_cnn", ["-c", "2", "-n", "16384", "-b", "32"], "./", clocks, data_root, description)


# def run_power_cap_experiment_cifar(data_root, power_caps):
#     description = "powercap{}-cifar"
#     run_power_cap_experiment("cifar10_resnet", [], "./", power_caps, data_root, description)

def run_all_power_cap_corse(data_root, repititions):
    power_caps = [150, 200, 250, 300] #Watts
    data_root = Path(data_root)
    
    for i in range(repititions):
        data_path = data_root / f"run{i}"
        # run_power_cap_experiment_mnist(data_path, power_caps)
        # time.sleep(5)
        # run_power_cap_experiment_mnist_big(data_path, power_caps)
        # time.sleep(5)
        run_power_cap_experiment_ecg(data_path, power_caps)
        time.sleep(5)

def run_all_clocks(data_root, repititions):
    clocks = [135, 285, 435, 585, 735, 885, 1035, 1185, 1335, 1485]
    clocks = [(877,x) for x in clocks]
    data_root = Path(data_root)

    for i in range(repititions):
        data_path = data_root / f"run{i}"
        run_clock_experiment_mnist(data_path, clocks)
        time.sleep(5)
        run_clock_experiment_mnist_big(data_path, clocks)
        time.sleep(5)
        run_clock_experiment_ecg(data_path, clocks)
        time.sleep(5)
    
        #run_power_cap_experiment_cifar(data_path, power_caps)
# %%
# def run_batch_experiment():
#     #takes 1h to run on gtx 1060
#     dense_count = [1, 2, 4]
#     dense_sizes = [128, 256]
#     batch_sizes = [8, 16, 32, 64, 128]
#     for c in dense_count:
#         for d in dense_sizes:
#             for b in batch_sizes:
#                 description = f"{b}batch{d}denseX{c}"
#                 print(f"running {description}")
#                 data_root = Path("power-data")
#                 data_path = data_root / Path(f"{description}-{datetime.now().isoformat()}")
#                 args = ["python3","mnist_cnn.py",str(c), str(d), str(b) , str(data_path)]
#                 run_experiment(data_path, args)

# %%
#run_batch_experiment()

# def run_power_cap_experiment_fine():
#     import random
#     power_caps = list(range(150,300,25))#[150, 200, 250, 300] #Watts
#     random.shuffle(power_caps)
#     epochs = 5
#     for p in power_caps:
#         description = f"powercap{p}"
#         print(f"running {description}")
#         data_root = Path("power-data-fine")
#         data_path = data_root / Path(f"{description}-{datetime.now().isoformat()}")
#         args = ["/bin/bash","./train_cinc17.sh", "-n", str(epochs), "-p", str(data_path)]
        
#         run_experiment(data_path, args,baseline=5,power_limit=p)
#         time.sleep(10)

# def run_batch_experiment2(data_root):
#     dense_count = [8, 16, 32]
#     dense_sizes = [128, 256, 512]
#     batch_sizes = [64, 128, 256, 512]
#     for c in dense_count:
#         for d in dense_sizes:
#             for b in batch_sizes:
#                 description = f"{b}batch{d}denseX{c}"
#                 print(f"running {description}")
#                 data_root = Path(data_root)
#                 data_path = data_root / Path(f"{description}-{datetime.now().isoformat()}")
#                 args = ["-c", str(c), "-n", str(d), "-b", str(b)]
#                 asyncio.run(run_experiment(data_path, "./", "mnist_cnn", args))

def load_experiment_definition():

    with open("../experiment.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return

    return config


def load_benchmark_definition(path):
    with open(path) as f:
        try:
            benchmark = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
            return
    benchmark["benchmark_name"] = Path(path).stem
    print(benchmark)        
    return benchmark



def run_experiment(device_index: int, data_path: str, working_directory: str, module: str,
                   args: List[str], power_limit: int, clocks: Tuple[int, int],
                   experiment_name: str = None, benchmark_name: str = None):
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    with NVMLLib() as lib:
        # get device
        device = lib.device.from_index(device_index)

        SystemInfo.gather(device).save(data_path / "system_info.json")

        # set constraints
        limit = PowerLimit(device, power_limit)
        clocks = ApplicationClockLimit(device, *clocks)
        #with limit, clocks:
        args = ["python3", "-m", "g_py_joules", "-d", str(data_path.absolute()),# "-e",
                "-v", str(device_index), "-w", str(working_directory), module, "--"] + args
        print(args)
        p = subprocess.Popen(args)
        while p.poll() is None:
            time.sleep(1)


if __name__ == "__main__":
    benchmarks_dir = "../benchmarks"
    experiment = load_experiment_definition()

    benchmarks = []
    for bench in experiment["benchmarks"]:
        path = Path(benchmarks_dir) / bench
        path = path.with_suffix(".yaml")
        benchmarks.append(load_benchmark_definition(path))

    for bench in benchmarks:
        for power_limit in experiment["power_limits"]:
            config = {**experiment, **bench}
            del config["power_limits"]
            del config["clock_limits"]
            del config["benchmarks"]
            config["power_limit"] = power_limit
            config["device_index"] = int(config.pop("devices"))
            config["clocks"] = (None, None)
            run_experiment(**config)

    # from git import Repo
    #
    # repo = Repo("../")
    # current_tag = next((tag for tag in repo.tags if tag.commit == repo.head.commit), None)
    #
    # #assert current_tag is not None, "Error: Need a tagged commit to be checked out!"
    # if current_tag is not None:
    #     data_path = Path("../data") / str(current_tag.name)
    # else:
    #     data_path = Path("../data") / str(repo.head.commit.hexsha)
    # data_path.mkdir(parents=True, exist_ok=False)
    #
    # run_all_power_cap_corse(data_path, 10)
    #
    # #run_all_clocks(data_path, 5)
    #
    # # reset clocks, when done
    # SMIWrapper.set_clocks(None)
    # SMIWrapper.set_power_limit(None)