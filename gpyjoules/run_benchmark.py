"""This module is the main entry point to run the benchmarks."""
import os
import random
import subprocess
import time
from pathlib import Path
from typing import List, Tuple, Dict, Union, Any, Hashable, Sequence, Optional

import yaml
from pynvml3 import NVMLLib, PowerLimit, ApplicationClockLimit

from .system_info import SystemInfo

EXPERIMENTS_PATH = Path("experiments")


def get_hostname() -> str:
    """Return the hostname."""
    hostname = os.environ.get("HOST")
    if hostname is None:
        hostname = os.environ.get("HOSTNAME")
    return hostname


def yaml_from_path(path: Path) -> dict:
    """Load a yaml file from the given path."""
    with open(path, "r", encoding="UTF-8") as f:
        return yaml.safe_load(f)


def load_experiment_definition(
    host_name: str,
) -> Union[Dict[Hashable, Any], list, None]:
    """Load the experiment definition file which path is specified in `platform.txt`.

    Args:
        path: a path to an experiment definition yaml-file

    Returns:
        the experiment definion as a dict

    """
    experiments = []
    for path in EXPERIMENTS_PATH.rglob("*.yml"):
        config = yaml_from_path(path)
        for experiment in config:
            if experiment.get("host", None) == host_name:
                experiments.append(experiment)
    return experiments


def load_benchmark_definition(path: Union[Path, str]) -> Dict[Hashable, Any]:
    """Load the benchmark definition yaml-file from the given path.

    Args:
        path: a path to a benchmark definition yaml-file

    Returns:
        the benchmark definition as a dict

    """
    benchmark = yaml_from_path(path)
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
    return value * 1000 if value is not None else None


def run_benchmark(
    device_index: int,
    data_path: str,
    working_directory: str,
    module: str,
    args: List[str],
    power_limit: Optional[int],
    clocks: Optional[Tuple[int, int]],
    repetition: int,
    experiment_name: str = None,
    benchmark_name: str = None,
) -> None:
    """Run a benchmark on a given device, with the given constraints, while collecting power-data.

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
        limit = PowerLimit(
            device, watt2milliwatt(power_limit), set_default=True, check=True
        )

        clocks = ApplicationClockLimit(device, *clocks, set_default=True, check=True)
        print(power_limit)
        with limit, clocks:

            SystemInfo.gather(device).save(data_path / "system_info.json")

            args = [
                "python3",
                "-m",
                "gpyjoules.gpyjoules",
                "-d",
                str(data_path.absolute()),  # "-e",
                "-v",
                str(device_index),
                "-w",
                str(working_directory),
                module,
                "--",
            ] + args
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


def prepare_configs(exp_config: Dict, bench_config: Dict, repetition: int, benchmark_name: str) -> Dict:
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


def get_baseline(data_path: Union[Path, str], device_index: int, baseline_length: int):
    data_path = Path(data_path) / "_baseline"
    data_path.mkdir(exist_ok=True, parents=True)
    args = [
        "python3",
        "-m",
        "gpyjoules.gpyjoules",
        "-d",
        str(data_path.absolute()),
        "-v",
        str(device_index),
        "None",
        "-b",
        "-bl",
        str(baseline_length),
    ]
    print(args)
    try:
        p = subprocess.Popen(args)
        while p.poll() is None:
            time.sleep(1)
    except Exception as e:
        p.kill()
        raise e


BASELINE_LENGTH = 60


def main():
    """The main function."""
    benchmarks_dir = "benchmarks"
    experiments = load_experiment_definition(get_hostname())
    
    # experiment
    for experiment in experiments:
        print(f"Experiment: {experiment['experiment_name']}")

        print("Getting baseline measurements ...")
        get_baseline(
            Path(experiment["data_path"]) / experiment["experiment_name"],
            os.environ["NVIDIA_VISIBLE_DEVICES"],
            BASELINE_LENGTH,
        )
        print("Done.")
        benchmarks = []

        for bench in experiment["benchmarks"]:
            path = Path(benchmarks_dir) / bench
            path = path.with_suffix(".yaml")
            benchmarks.append(load_benchmark_definition(path))

        reps = experiment.get("repeat", 1)
        # benchmark repetition
        for repetition in range(reps):
            print(f"Repetition {repetition}/{reps}")
            # benchmark
            for bench in benchmarks:
                # iterate randomly over power limits
                # benchmark configuration
                for power_limit in randomly(experiment.get("power_limits", [])):
                    config = prepare_configs(experiment, bench, repetition)
                    config["power_limit"] = power_limit
                    config["clocks"] = (None, None)
                    run_benchmark(**config)

                # itrate randomly over clock_limits
                # benchmark configuration
                for clock_limits in randomly(experiment.get("clock_limits", [])):
                    config = prepare_configs(experiment, bench, repetition)
                    config["power_limit"] = None
                    config["clocks"] = tuple(clock_limits)
                    run_benchmark(**config)


if __name__ == "__main__":
    main()
