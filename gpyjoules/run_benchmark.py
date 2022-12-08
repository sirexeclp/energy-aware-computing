"""This module is the main entry point to run the benchmarks."""
import argparse
import json
import logging as log
import os
import random
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import List, Tuple, Dict, Union, Any, Hashable, Sequence, Optional
from gpyjoules.Experiment import Experiment

import yaml
from pynvml3 import NVMLLib, PowerLimit, ApplicationClockLimit

from .system_info import SystemInfo

EXPERIMENTS_PATH = Path("experiments")

def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    # path_parser = parser.add_mutually_exclusive_group()

    group.add_argument(
        "-host",
        "--hostname",
        action="store_true",
        dest="hostname",
        help="If invoked, automatically collect and execute all available benchmarks based on the current hostname."
    )

    group.add_argument(
        "-p",
        "--path",
        dest="path",
        action="store",
        nargs="+",
        help="Provide a path to a specific experiment configuration."
    )

    return parser.parse_args()
    

def yaml_from_path(path: Path) -> dict:
    """Load a yaml file from the given path."""
    with open(path, "r", encoding="UTF-8") as f:
        return yaml.safe_load(f)


def watt2milliwatt(value: Union[int, float, None]) -> Union[int, float, None]:
    """Converts Watts to milliWatts.

    Args:
        value: a power value measured in watt

    Returns:
        a power value measured in milliwatt

    """
    return value * 1000 if value is not None else None


def randomly(seq: Sequence) -> List:
    """Helper function to iterate randomly over a sequence.

    Args:
        seq: a sequence to randomize

    Returns:
        a shuffled list

    """
    shuffled = list(seq)
    random.shuffle(shuffled)
    return shuffled


def prepare_configs(exp_config: Dict, bench_config: Dict, repetition: int) -> Dict:
    """Merge experiment and benchmark config and prepare for
    passing to `run_benchmark` function.

    Args:
        exp_config: the global experiment configuration
        bench_config: the configuration for a benchmark

    Returns:
        a dict with a configuration that can be unpacked
        and passed to the `run_benchmark` function.

    """
    config = {**exp_config, **bench_config}
    config.pop("power_limits", None)
    config.pop("clock_limits", None)
    del config["benchmarks"]
    del config["repeat"]
    config["repetition"] = repetition
    config["device_index"] = int(
        os.environ["NVIDIA_VISIBLE_DEVICES"]
    )  # int(config.pop("devices"))
    return config


def collect_experiments(args):
    hostname = socket.gethostname()
    experiments = []
    if(args.hostname):
        paths = EXPERIMENTS_PATH.rglob("*.yml")
        for path in paths:
            config = yaml_from_path(path)
            if config.get("host", None) == hostname:
                print(config)
                e = Experiment(**config)
                experiments.append(e)
    else:
        paths = args.path
        for path in paths:
            if not os.path.isfile(path):
                log.error(f"File {path} does not exist")
            else:
                config = yaml_from_path(path)
                print(config)
                e = Experiment(**config)
                experiments.append(e)
    return experiments


def run_experiment(e: Experiment, device_index):
    # print("Getting baseline measurements ...")
    # baseline_args = e.get_baseline_args(device_index)
    # run_measurement(baseline_args)
    # print("Baseline measurements Done.")
    for rep in range(e.repetitions):
        # TODO: iterate randomly through power/clock limits and pass on
        for bench in e.benchmarks:
            benchmark_args = e.get_benchmark_args(bench, rep, device_index)
            with NVMLLib() as lib:
                # get device
                device = lib.device.from_index(device_index)

                # set constraints
                # reset power-limit to default value, when we are done and check if it was set successfully
                # convert watts to milliwatts
                limit = PowerLimit(
                    device, watt2milliwatt(e.power_limits[0]), set_default=True, check=True
                )

                # print(e.power_limits[0])
                # print(*tuple(e.clock_limits[0]))
                clocks = ApplicationClockLimit(device, *tuple(e.clock_limits[0]), set_default=True, check=True)
                with limit, clocks:
                    run_measurement(benchmark_args)


def run_measurement(args):
    command = [
        "python3",
        "-m",
        "gpyjoules.gpyjoules",
    ]
    command += args
    try:
        p = subprocess.Popen(command)
        while p.poll() is None:
            time.sleep(1)
    except Exception as e:
        p.kill()
        raise e

BASELINE_LENGTH = 60


def main():
    """The main function."""
    # if(os.path.exists(Path.home() / "tmp/data-test")):
    #     shutil.rmtree(Path.home() / "tmp/data-test")
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "0"
    args = parse_args()
    
    print("Collecting Experiments...")
    experiments = collect_experiments(args)

    print("Collected following experiments:")
    for experiment in experiments:
        print(f" - {experiment.name}")

    # experiment
    for i in range(len(experiments)):
        print(80*"-")
        print(f"Starting Experiment {i+1}/{len(experiments)}: {experiments[i].name}")
        # print("Details:")
        # print(json.dumps(vars(experiment), indent=2))

        print(80*"-")
        print("Collecting Benchmarks...")
        experiments[i].collect_benchmarks()
        print("Benchmarks collected")
        
        print(80*"-")
        run_experiment(experiments[i], int(os.environ["NVIDIA_VISIBLE_DEVICES"]))
        return
        reps = experiments[i].repetitions
        # benchmark repetition
        for repetition in range(reps):
            print("---------------------------------------------------------------------------------------")
            print(f"Starting Repetition {repetition+1}/{reps} of Experiment {experiments[i].name}")
            # benchmark
            for bench in experiments[i].benchmarks:
                print("---------------------------------------------------------------------------------------")
                # print("Benchmark Details:")
                # print(json.dumps(bench, indent=2))

#                if 'power_limits' in experiments[i].keys():
                if experiments[i].power_limits:
                    print("Iterating through power limits")
                    for power_limit in randomly(experiments[i].power_limits):
                        config = prepare_configs(experiment, bench, repetition)
                        config["power_limit"] = power_limit
                        config["clocks"] = (None, None)
                        run_benchmark(**config)
                elif experiments[i].clock_limits:
#                 elif 'clock_limits' in experiments[i].keys():
                    print("Iterating through clock limits")
                    for clock_limits in randomly(experiments[i].clock_limits):
                        config = prepare_configs(experiment, bench, repetition)
                        config["power_limit"] = None
                        config["clocks"] = tuple(clock_limits)
                        run_benchmark(**config)
                else:
                    print("No clock or power limits specified")
                    config = prepare_configs(experiment, bench, repetition)
                    config["power_limit"] = None
                    config["clocks"] = (None, None)
                    json.dumps(config)
                    run_benchmark(**config)
                

if __name__ == "__main__":
    main()
