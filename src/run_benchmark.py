# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import nvidia_smi as smi
import subprocess
import time
from datetime import datetime
from collections import namedtuple
from pathlib import Path
import pandas as pd
from smi_wrapper import GPU, SMIWrapper, Utilization
from random import shuffle
import copy
# %%
WARMUP = 5 #warmup in seconds

def run_experiment(data_path, working_directory, module, args ,baseline = 0, power_limit=None):
    data_path.mkdir(parents=True)

    SMIWrapper.set_power_limit(power_limit)
        
    args = ["python3", "-m","gPyJoules", "-d", str(data_path.absolute()), "-w", str(working_directory), module, "--"] + args
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


def run_power_cap_experiment_ecg(data_root, power_caps):
    epochs = 20
    description = "powercap{}-ecg"
    args = ["examples/cinc17/config.json" ,"-e" ,"cinc17", "-n", str(epochs)]
    run_power_cap_experiment("ecg.train", args, "../../ecg/", power_caps, data_root, description)

def run_power_cap_experiment_mnist(data_root, power_caps):
    description = "powercap{}-mnist"
    run_power_cap_experiment("mnist_cnn", ["-c", "2", "-n", "512", "-b", "128"], "./", power_caps, data_root, description)

def run_power_cap_experiment_mnist_big(data_root, power_caps):
    description = "powercap{}-mnist-big"
    run_power_cap_experiment("mnist_cnn", ["-c", "2", "-n", "16384", "-b", "32"], "./", power_caps, data_root, description)

# def run_power_cap_experiment_cifar(data_root, power_caps):
#     description = "powercap{}-cifar"
#     run_power_cap_experiment("cifar10_resnet", [], "./", power_caps, data_root, description)

def run_all_power_cap_corse(data_root, repititions):
    power_caps = [150, 200, 250, 300] #Watts
    data_root = Path(data_root)
    
    for i in range(repititions):
        data_path = data_root / f"run{i}"
        run_power_cap_experiment_mnist(data_path, power_caps)
        time.sleep(5)
    
    for i in range(repititions):
        data_path = data_root / f"run{i}"
        run_power_cap_experiment_mnist_big(data_path, power_caps)
        time.sleep(5)
    
    for i in range(repititions):
        data_path = data_root / f"run{i}"
        run_power_cap_experiment_ecg(data_path, power_caps)
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


if __name__ == "__main__":
    
    run_all_power_cap_corse("../data-1.3", 5)

    with SMIWrapper() as sw:
        success = sw.reset_all_powerlimit()
        print(f"[{'Success' if success else 'Failed'}] reset power-limit to default (max)")