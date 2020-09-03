# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import os
import random
import subprocess
import time
from pathlib import Path
from typing import List, Tuple

import yaml
from pynvml3 import NVMLLib, PowerLimit, ApplicationClockLimit

from system_info import SystemInfo


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


def watt2milliwatt(value):
    return value * 1000


def run_experiment(device_index: int, data_path: str, working_directory: str, module: str,
                   args: List[str], power_limit: int, clocks: Tuple[int, int], repetition: int,
                   experiment_name: str = None, benchmark_name: str = None):
    data_path = Path(data_path) / experiment_name
    data_path = data_path / f"run{repetition}"
    data_path = data_path / benchmark_name
    data_path = data_path / f"{power_limit}W"
    data_path.mkdir(parents=True, exist_ok=False)

    with NVMLLib() as lib:
        # get device
        device = lib.device.from_index(device_index)

        # set constraints
        # reset power-limit to default value, when we are done and check if it was set successfully
        # convert watts to milliwatts
        limit = PowerLimit(device, watt2milliwatt(power_limit), set_default=True, check=True)
        clocks = ApplicationClockLimit(device, *clocks, set_default=True, check=True)
        print(power_limit)
        with limit, clocks:

            SystemInfo.gather(device).save(data_path / "system_info.json")

            args = ["python3", "-m", "g_py_joules", "-d", str(data_path.absolute()),  # "-e",
                    "-v", str(device_index), "-w", str(working_directory), module, "--"] + args
            print(args)
            try:
                p = subprocess.Popen(args)
                while p.poll() is None:
                    time.sleep(1)
            except Exception as e:
                p.kill()
                raise e


def randomly(seq):
    shuffled = list(seq)
    random.shuffle(shuffled)
    return shuffled


if __name__ == "__main__":
    benchmarks_dir = "../benchmarks"
    experiment = load_experiment_definition()

    benchmarks = []
    for bench in experiment["benchmarks"]:
        path = Path(benchmarks_dir) / bench
        path = path.with_suffix(".yaml")
        benchmarks.append(load_benchmark_definition(path))

    reps = experiment.get("repeat", 1)
    for repetition in range(reps):
        print(f"Repetition {repetition}/{reps}")
        for benchmark_name, bench in zip(experiment["benchmarks"], benchmarks):
            # iterate randomly over power limits
            for power_limit in randomly(experiment["power_limits"]):
                config = {**experiment, **bench}
                del config["power_limits"]
                del config["clock_limits"]
                del config["benchmarks"]
                del config["repeat"]
                config["repetition"] = repetition
                config["benchmark_name"] = benchmark_name
                config["power_limit"] = power_limit
                config["device_index"] = int(os.environ["NVIDIA_VISIBLE_DEVICES"])  # int(config.pop("devices"))
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
