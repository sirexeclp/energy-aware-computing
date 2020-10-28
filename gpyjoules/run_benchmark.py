import os
import random
import subprocess
import time
from pathlib import Path
from typing import List, Tuple, Dict, Union, Any, Hashable, Sequence, Optional

import yaml
from pynvml3 import NVMLLib, PowerLimit, ApplicationClockLimit

from .system_info import SystemInfo


def load_experiment_definition(path: str) -> Union[Dict[Hashable, Any], list, None]:
    """Load the experiment definition, which is stored in
    `experiment.yaml`.

    Args:
        path: a path to an experiment definition yaml-file
    
    Returns:
        the experiment definion as a dict

    """
    with open(path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return

    return config


def load_benchmark_definition(path: Union[Path, str]) -> Dict[Hashable, Any]:
    """Load the benchmark definition yaml-file from the given path.
    
    Args:
        path: a path to a benchmark definition yaml-file

    Returns:
        the benchmark definition as a dict

    """
    with open(path) as f:
        try:
            benchmark = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
            return
    benchmark["benchmark_name"] = Path(path).stem
    print(benchmark)
    return benchmark


def watt2milliwatt(value: Union[int, float, None]) -> Union[int, float, None]:
    """Converts Watts to milliWatts.

    Args:
        value: a power value measured in watt

    Returns:
        a power value measured in milliwatt

    """
    if value is None:
        return None
    else:
        return value * 1000


def run_benchmark(device_index: int, data_path: str, working_directory: str, module: str,
                  args: List[str], power_limit: Optional[int], clocks: Optional[Tuple[int, int]], repetition: int,
                  experiment_name: str = None, benchmark_name: str = None) -> None:
    """Run a benchmark once on a given device, with the given constraints, while collecting
    power-data.

    Args:
        device_index: the index of the gpu device to run the benchmark on
        data_path: the root data path (defined in experiment.yaml)
        working_directory: the working directory to run the benchmark in
        module: the python module to run as a benchmark
        args: a list of commandline arguments passed to the python module
        power_limit: a powerlimit in Watt or None
        clocks: a tuple of gpu memory and graphics clock limit or None
        repetition: the index of this benchmark repetition
        experiment_name: the name of the experiment
        benchmark_name: the name of the benchmark

    """

    data_path = Path(data_path) / experiment_name
    data_path = data_path / benchmark_name
    if power_limit is not None:
        data_path = data_path / f"{power_limit}W"
    else:
        data_path = data_path / f"{clocks[0]}MHz,{clocks[1]}MHz"

    data_path = data_path / f"{repetition}"
    if data_path.exists():
        max_id = max([int(x.name) for x in data_path.parent.glob("*")])
        next_id = max_id + 1
        data_path = data_path.parent / f"{next_id}"
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

            args = ["python3", "-m", "gpyjoules.g_py_joules", "-d", str(data_path.absolute()),  # "-e",
                    "-v", str(device_index), "-w", str(working_directory), module, "--"] + args
            print(args)
            try:
                p = subprocess.Popen(args)
                while p.poll() is None:
                    time.sleep(1)
            except Exception as e:
                p.kill()
                raise e


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


def prepare_configs(exp_config: Dict, bench_config: Dict) -> Dict:
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
    config["benchmark_name"] = benchmark_name
    config["device_index"] = int(os.environ["NVIDIA_VISIBLE_DEVICES"])  # int(config.pop("devices"))
    return config


def get_baseline(data_path: Union[Path, str], device_index: int, baseline_length: int):
    data_path = Path(data_path) / "_baseline"
    data_path.mkdir(exist_ok=True, parents=True)
    args = ["python3", "-m", "gpyjoules.g_py_joules", "-d", str(data_path.absolute()),
            "-v", str(device_index), "None", "-b", "-bl", str(baseline_length)]
    print(args)
    try:
        p = subprocess.Popen(args)
        while p.poll() is None:
            time.sleep(1)
    except Exception as e:
        p.kill()
        raise e


BASELINE_LENGTH = 60

if __name__ == "__main__":
    benchmarks_dir = "benchmarks"
    experiments = load_experiment_definition("experiment.yaml")
    experiments = experiments.get("experiments", [experiments])

    benchmarks = []
    # experiment
    for experiment in experiments:
        print(f"Experiment: {experiment['experiment_name']}")

        print("Getting baseline measurements ...")
        get_baseline(Path(experiment["data_path"]) / experiment["experiment_name"],
                     int(os.environ["NVIDIA_VISIBLE_DEVICES"]), BASELINE_LENGTH)
        print("Done.")

        for bench in experiment["benchmarks"]:
            path = Path(benchmarks_dir) / bench
            path = path.with_suffix(".yaml")
            benchmarks.append(load_benchmark_definition(path))

        reps = experiment.get("repeat", 1)
        # benchmark repetition
        for repetition in range(reps):
            print(f"Repetition {repetition}/{reps}")
            # benchmark
            for benchmark_name, bench in zip(experiment["benchmarks"], benchmarks):
                # iterate randomly over power limits
                # benchmark configuration
                for power_limit in randomly(experiment.get("power_limits", [])):
                    config = prepare_configs(experiment, bench)
                    config["power_limit"] = power_limit
                    config["clocks"] = (None, None)
                    run_benchmark(**config)

                # itrate randomly over clock_limits
                # benchmark configuration
                for clock_limits in randomly(experiment.get("clock_limits", [])):
                    config = prepare_configs(experiment, bench)
                    config["power_limit"] = None
                    config["clocks"] = tuple(clock_limits)
                    run_benchmark(**config)

