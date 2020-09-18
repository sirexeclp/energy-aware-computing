import json
import re
from functools import reduce
from pathlib import Path
from typing import Union, List, Dict, Optional

import numpy as np
import pandas as pd
from scipy import integrate
from scipy.interpolate import interp1d

from src.util import TimestampInfo


def overlay_plots(plots: List):
    return reduce(lambda x, y: x * y, plots)


def atoi(text):
    return int(text) if text.isdigit() else text


# Source: https://stackoverflow.com/a/5967539/7997186
def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def interpolate(x, y, freq, start=0):
    x = np.array(x)
    y = np.array(y)
    x_tick = np.arange(start, x[-1], 1 / freq)
    f_cubic = interp1d(x, y, kind="cubic")
    return x_tick, f_cubic(x_tick)


def interpolate_df(df: pd.DataFrame, freq: float, start=0):
    x = df.index
    new_df = {}
    for col in df:
        y = df[col]
        x_hat, y_hat = interpolate(x, y, freq, start)
        new_df[col] = y_hat

    new_df[x.name] = x_hat
    new_df = pd.DataFrame(new_df)
    return new_df

def normalize_timestamp(x, start_time):
    return (x - start_time) / np.timedelta64(1, 's')


class Measurement:

    def __init__(self, path: Union[Path, str], name: str, df=None):
        self.path = Path(path)
        self.name = name
        if df is None:
            self.data = self.load()
        else:
            self.data = df

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def load(self):
        df = pd.read_csv(self.path, index_col=0)
        return df

    def normalize_timestamps(self, start_time):
        self["timestamp"] = normalize_timestamp(self["timestamp"], start_time)
        # df = df[df.timestamp >= 0]

    def interpolate(self, freq: float, start=0):
        self.data = interpolate_df(self.data.set_index("timestamp"), freq, start)

    def calculate_energy(self):
        self["energy"] = get_energy(self["power"], self["timestamp"])

    def calculate_edp(self):
        self["edp"] = self["energy"] * self["timestamp"]

    def plot(self, metric: str, label: str):
        return self.data.set_index("timestamp")[metric].hvplot(label=label)

    def __add__(self, other):
        result = self.data + other.data
        return Measurement(self.path.parent, self.name, result)

    def __sub__(self, other):
        result = self.data - other.data
        return Measurement(self.path.parent, self.name, result)

    def __truediv__(self, other):
        result = self.data / other.data
        return Measurement(self.path.parent, self.name, result)

    def __mul__(self, other):
        result = self.data * other.data
        return Measurement(self.path.parent, self.name, result)


class BenchmarkRun:
    def __init__(self, path: Union[Path, str], benchmark: "Benchmark"):
        self.path = Path(path)
        self.name = self.path.name
        self.benchmark = benchmark
        self.experiment = self.benchmark.experiment
        self.repetitions = self._load_repetitions()
        self.exclude_outliers(0.95)

    def __str__(self):
        return f"BenchmarkRun({self.name})"

    def __repr__(self):
        return self.__str__()

    def _load_repetitions(self) -> List["BenchmarkRepetition"]:
        return [BenchmarkRepetition(x, self, self.benchmark) for x in sorted(self.path.glob("*"))]

    def __getitem__(self, index):
        return self.repetitions[index]

    def aggregate(self, data_source: str, func: str):
        dfs = [x.measurements[data_source].data for x in self.repetitions]  # [:-1]
        new_df = {}
        for col in dfs[0]:
            new_df[col] = pd.concat([x[col] for x in dfs], axis=1).aggregate(func, axis=1)#.mean(axis=1)

        new_df = pd.DataFrame(new_df)
        return Measurement(self.path, f"{data_source}-{func}", df=new_df)

    def get_total_values(self, aggregate=None, data_source: str = "hd"):
        values = [r.get_total_values(data_source) for r in self.repetitions]
        df = pd.DataFrame(values)
        if aggregate is not None:
            df = df.apply(aggregate)
        df["run"] = self.name
        return df

    def exclude_outliers(self, percentile=0.95):
        total_times = [x.get_total_values().timestamp for x in self.repetitions]
        outlier_indices = np.argwhere(np.quantile(total_times, percentile) < total_times)
        for [x] in outlier_indices:
            self.repetitions.pop(x)


class BenchmarkRepetition:
    def __init__(self, path: Union[Path, str], run: "BenchmarkRun", benchmark: "Benchmark"):
        self.path = Path(path)
        self.name = self.path.name
        self.benchmark = benchmark
        self.run = run
        self.experiment = self.benchmark.experiment

        self.timestamps = self._load_timestamps()
        self.files = self._list_files()
        self.measurements = self.load_measurements()

        self.system_info = self._load_system_info()
        self.start_index = self._get_start_index()

        self.prepare_data()

    def __str__(self):
        return f"BenchmarkRepetition({self.name})"

    def __repr__(self):
        return self.__str__()

    def prepare_data(self):
        for key, m in self.measurements.items():
            m.normalize_timestamps(self.start_index)
            m.interpolate(50)
            m.data["power"] = m.data["power"] / 1_000
            m.calculate_energy()
            m.calculate_edp()

        # self.power_hd = self.interpolate_df(self.power_hd.set_index("timestamp"), 50, start=0)
        # self.data_sd = self.interpolate_df(self.data_sd.set_index("timestamp"), 50, start=0)

        # self.data_sd["power"] = self.data_sd["power"] / 1_000
        # self.power_hd["power"] = self.power_hd["power"] / 1_000

        # self.calculate_energy(self.data_sd)
        # self.calculate_energy(self.power_hd)
        #
        # self.calculate_edp(self.data_sd)
        # self.calculate_edp(self.power_hd)

        # self.power_hd["power"] = self.interpolate(self.power_hd.timestamp, self.power_hd.power, 1/50)

    def _list_files(self) -> List[Path]:
        return list(self.path.glob("*"))

    def load_measurements(self) -> Dict[str, "Measurement"]:
        sd = Measurement(self.path / "gpu-power.csv", "sd")
        sd.data["timestamp"] = pd.to_datetime(sd.data["timestamp"])

        hd = Measurement(self.path / "total_power_samples.csv", "hd")
        hd.data = hd.data.rename(columns={"value": "power"})
        hd.data["timestamp"] = pd.to_datetime(hd.data["timestamp"], unit="us")

        measurements = [sd, hd]
        return name_dict_from_list(measurements)

    def _load_timestamps(self) -> Optional[TimestampInfo]:
        path = self.path / "timestamps.csv"
        if path.exists():
            return TimestampInfo(pd.read_csv(path))
        else:
            return None

    # def _load_sd_data(self) -> pd.DataFrame:
    #     df = pd.read_csv(self.path / "gpu-power.csv", index_col=0)
    #     df["timestamp"] = pd.to_datetime(df["timestamp"])
    #     return df

    def _load_system_info(self) -> dict:
        with open(self.path / "system_info.json", "r") as f:
            return json.load(f)

    # def _load_hd_power(self) -> pd.DataFrame:
    #     # df = pd.read_csv(self.path / "total_power_samples.csv", index_col=0)
    #     # df = df.rename(columns={"value": "power"})
    #     df["timestamp"] = pd.to_datetime(df["timestamp"], unit="us")
    #     return df

    def _get_start_index(self) -> np.datetime64:
        if self.timestamps is not None and self.timestamps.train_begin is not None:
            return self.timestamps.train_begin
        else:
            start_times = [x.data.timestamp.min() for x in self.measurements.values()]
            return max(start_times)

    def plot(self, metric, data_source="hd"):
        return self.measurements[data_source].plot(metric=metric, label=self.run.name)

    def get_total_values(self, data_source: str = "hd") -> pd.Series:
        return self.measurements[data_source].data.iloc[-1].rename(self.name)


class Benchmark:
    def __init__(self, path: Union[Path, str], experiment: "Experiment"):
        self.path = Path(path)
        self.name = self.path.name
        self.experiment = experiment
        self.runs = self._load_runs()

    def __str__(self):
        return f"Benchmark({self.name})"

    def __repr__(self):
        return self.__str__()

    def _load_runs(self) -> Dict[str, BenchmarkRun]:
        paths = [str(p) for p in self.path.glob("*")]
        paths.sort(key=natural_keys)
        return dict(map(lambda x: (x.name, x), [BenchmarkRun(x, self) for x in paths]))

    def plot(self, metric, data_slice, data_source="hd"):
        plots = []
        if isinstance(data_slice, str):
            for key, run in self.runs.items():
                aggregate: Measurement = run.aggregate(data_source, "mean")
                plot = aggregate.plot(metric=metric, label=key)
                plots.append(plot)
        else:
            for key, run in self.runs.items():
                bench_rep: BenchmarkRepetition = run[data_slice]
                plot = bench_rep.plot(metric=metric, data_source=data_source)
                plots.append(plot)
            pass

        return overlay_plots(plots)

    def plot_raw_power(self, data_slice, data_source="hd"):
        return self.plot(metric="power",
                         data_slice=data_slice, data_source=data_source)

    def plot_energy(self, data_slice, data_source="hd"):
        return self.plot(metric="energy",
                         data_slice=data_slice, data_source=data_source)

    def get_total_values(self, aggregate=None):
        dfs = []
        for key, run in self.runs.items():
            dfs.append(run.get_total_values(aggregate))

        if aggregate is None:
            return pd.concat(dfs)
        else:
            df = pd.DataFrame(dfs)
            return df.set_index("run")

    def boxplot(self, metric):
        df = self.get_total_values()
        boxplot = df.hvplot.box(y=metric, by='run'
                                , height=450, width=800
                                , legend=True)
        return boxplot


class Experiment:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.name = self.path.name
        self.benchmarks = self.load_benchmarks()

    def load_benchmarks(self) -> Dict[str, Benchmark]:
        benchmarks = [Benchmark(x, self) for x in self.path.glob("*")]
        return name_dict_from_list(benchmarks)

    def __str__(self):
        return f"Experiment({self.name})"

    def __repr__(self):
        return self.__str__()


class DataLoader:
    def __init__(self, data_root: Union[str, Path]):
        self.data_root = Path(data_root)
        self.experiments = self._load_experiments()

    def _load_experiments(self):
        experiment_paths = list(self.data_root.glob("*"))
        experiments = [Experiment(path) for path in experiment_paths]
        return name_dict_from_list(experiments)

    def __str__(self):
        return f"DataLoader({list(self.experiments.keys())})"

    def __repr__(self):
        return self.__str__()


def name_dict_from_list(data):
    return {x.name: x for x in data}


def get_energy(power, time):
    return integrate.cumtrapz(power, time, initial=0)

