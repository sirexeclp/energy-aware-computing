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

# %%
WARMUP = 5 #warmup in seconds

def run_experiment(data_path, args ,baseline = 0, power_limit=None):
    gpu_data = []
    
    data_path.mkdir(parents=True)
    gpu_data_path = data_path / "gpu-power.csv"

    with SMIWrapper() as sw:
        if power_limit is not None:
            success = sw.set_all_powerlimit(power_limit*1000)#convert to mW
            print(f"[{'Success' if success else 'Failed'}] set power-limit to {power_limit}")
        else:
            success = sw.reset_all_powerlimit()
            print(f"[{'Success' if success else 'Failed'}] reset power-limit to default (max)")
        
        assert success, "Failed setting/resetting power-limit, abborting experiment!"
        
        if baseline:
            sw.get_baseline(baseline, gpu_data)
       
        p = subprocess.Popen(args)
        while p.poll() is None:
            sw.get_all_stats(gpu_data)
            #time.sleep(0.5-0.0752)
        
        df = pd.DataFrame(gpu_data)
        df.to_csv(gpu_data_path)


# %%


def run_batch_experiment():
    #takes 1h to run on gtx 1060
    dense_count = [1, 2, 4]
    dense_sizes = [128, 256]
    batch_sizes = [8, 16, 32, 64, 128]
    for c in dense_count:
        for d in dense_sizes:
            for b in batch_sizes:
                description = f"{b}batch{d}denseX{c}"
                print(f"running {description}")
                data_root = Path("power-data")
                data_path = data_root / Path(f"{description}-{datetime.now().isoformat()}")
                args = ["python3","mnist_cnn.py",str(c), str(d), str(b) , str(data_path)]
                run_experiment(data_path, args)

# %%
#run_batch_experiment()

def run_power_cap_experiment():
    import random
    power_caps = [150, 200, 250, 300] #Watts
    random.shuffle(power_caps)
    epochs = 20#5
    for p in power_caps:
        description = f"powercap{p}"
        print(f"running {description}")
        data_root = Path("power-data")
        data_path = data_root / Path(f"{description}-{datetime.now().isoformat()}")
        args = ["/bin/bash","./train_cinc17.sh", "-n", str(epochs), "-p", str(data_path)]
        
        run_experiment(data_path, args,baseline=5,power_limit=p)
        time.sleep(10)

def run_power_cap_experiment_fine():
    import random
    power_caps = list(range(150,300,25))#[150, 200, 250, 300] #Watts
    random.shuffle(power_caps)
    epochs = 5
    for p in power_caps:
        description = f"powercap{p}"
        print(f"running {description}")
        data_root = Path("power-data-fine")
        data_path = data_root / Path(f"{description}-{datetime.now().isoformat()}")
        args = ["/bin/bash","./train_cinc17.sh", "-n", str(epochs), "-p", str(data_path)]
        
        run_experiment(data_path, args,baseline=5,power_limit=p)
        time.sleep(10)

def run_batch_experiment2(data_root):
    dense_count = [8, 16, 32]
    dense_sizes = [128, 256, 512]
    batch_sizes = [64, 128, 256, 512]
    for c in dense_count:
        for d in dense_sizes:
            for b in batch_sizes:
                description = f"{b}batch{d}denseX{c}"
                print(f"running {description}")
                data_root = Path(data_root)
                data_path = data_root / Path(f"{description}-{datetime.now().isoformat()}")
                args = ["python3","mnist_cnn.py",str(c), str(d), str(b) , str(data_path)]
                run_experiment(data_path, args)


if __name__ == "__main__":
    #run_power_cap_experiment()
    #time.sleep(120)
    #run_power_cap_experiment_fine()
    run_batch_experiment2("../testing/data")

    with SMIWrapper() as sw:
        success = sw.reset_all_powerlimit()
        print(f"[{'Success' if success else 'Failed'}] reset power-limit to default (max)")