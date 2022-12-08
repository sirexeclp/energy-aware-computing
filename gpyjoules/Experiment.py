import yaml
from pathlib import Path


class Benchmark():
    def __init__(self, name, module, args, working_directory):
        self.name = name
        self.module = module
        self.args = args
        self.working_directory = working_directory


class Experiment():
    def __init__(self, name, host, data_path, baseline_length=1, repetitions=1, power_limits=None, clock_limits= None, benchmarks=[]):
        self.name = name
        self.host = host
        self.data_path = data_path + "/" + name
        self.baseline_length = baseline_length
        self.repetitions = repetitions
        self.power_limits = power_limits
        self.clock_limits = clock_limits
        self.benchmark_names = benchmarks
        self.benchmarks = []
        self.check_data_path()


    def check_data_path(self):
        path = Path(self.data_path)
        # if path.exists():
        #     max_id = max([int(x.split("_")[-1]) for x in path.parent.glob("*")])
        #     next_id = max_id + 1
        #     path = path.parent + f"{next_id}"
        path.mkdir(parents=True, exist_ok=True)
        self.data_path = str(path)


    def collect_benchmarks(self):
        for b in self.benchmark_names:
            path = Path("benchmarks", b)
            path = path.with_suffix(".yaml")
            with open(path, "r", encoding="UTF-8") as f:
                benchmark = yaml.safe_load(f)
                bench = Benchmark(b, **benchmark)
            self.benchmarks.append(bench)


    def get_baseline_args(self, device_index):
        baseline_path = Path(Path.home(), self.data_path, "_baseline")
        baseline_path.mkdir(exist_ok=True, parents=True)
        args = [
            "-d",
            str(baseline_path.absolute()),
            "-v",
            str(device_index),
            "None",
            "-b",
            "-bl",
            str(self.baseline_length),
        ]
        return args


    def get_benchmark_args(self, benchmark: Benchmark, repetition, device_index):
        data_path = Path(Path.home(), self.data_path, benchmark.name, str(repetition))
        
        # if self.power_limit is not None:
        #     data_path = data_path / f"{power_limit}W"
        
        # if clocks != (None, None):
        #     data_path = data_path / f"{clocks[0]}MHz,{clocks[1]}MHz"

        # data_path = data_path / f"{repetition}"
        # if data_path.exists():
        #     max_id = max([int(x.name) for x in data_path.parent.glob("*")])
        #     next_id = max_id + 1
        #     data_path = data_path.parent / f"{next_id}"
        # data_path.mkdir(parents=True, exist_ok=False)

        # with NVMLLib() as lib:
        #     # get device
        #     device = lib.device.from_index(device_index)

        #     # set constraints
        #     # reset power-limit to default value, when we are done and check if it was set successfully
        #     # convert watts to milliwatts
        #     limit = PowerLimit(
        #         device, watt2milliwatt(power_limit), set_default=True, check=True
        #     )

        #     clocks = ApplicationClockLimit(device, *clocks, set_default=True, check=True)
        #     # print(power_limit)
        #     with limit, clocks:

        args = [
            "-d",
            str(data_path),  # "-e",
            "-v",
            str(device_index),
            "-w",
            str(benchmark.working_directory),
            benchmark.module,
            "--",
        ] + benchmark.args
        return args


