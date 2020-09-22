import json
import re
from functools import reduce
from pathlib import Path
from typing import Union, List, Dict, Optional

import numpy as np
import pandas as pd
from scipy import integrate
from scipy.interpolate import interp1d
import holoviews as hv

from src.util import TimestampInfo

label_replacements = {
    "power": "Power [W]",
    "clock-gpu": "GPU Clock [MHz]",
    "clock": "GPU Clock [MHz]",
    "edp": "Energy Delay Product (EDP) [kJS]",
    "energy": "Energy [kJ]",
    "run": "Limit",
    "timestamp": "Time [s]"
}


def overlay_plots(plots: List):
    return reduce(lambda x, y: x * y, plots)


def replace_dimension_labels(plot):
    for dim in plot.dimensions():
        # if dim.label == "run":
        #     dim_values = plot.dimension_values(dim.label)

        dim.label = label_replacements.get(dim.label, dim.label)


def apply_plot_default_settings(plot: "hv.core.overlay.Overlay"):
    replace_dimension_labels(plot)
    return plot.opts(width=800, height=450)


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


class LinearFit:
    def __init__(self, x, y, degree=2):
        self.x = x
        self.y = y
        self.degree = degree
        self.coefficients = np.polyfit(x, y, degree)
        self.poly = np.poly1d(self.coefficients)

    def __call__(self, arg):
        self.predict(arg)

    def predict(self, x_hat):
        return self.poly(x_hat)

    def r2(self, test_data=None):
        if test_data is None:
            x = self.x
            y = self.y
        else:
            x, y = test_data

        y_hat = self.predict(x)
        residuals = y_hat - y

        sst = np.sum((y - np.mean(y)) ** 2)
        ssr = np.sum(residuals ** 2)
        return 1 - (ssr / sst)

    def mse(self, test_data=None):
        if test_data is None:
            x = self.x
            y = self.y
        else:
            x, y = test_data

        y_hat = self.predict(x)
        residuals = y_hat - y
        return np.mean(residuals ** 2)

    def plot(self, res=100):
        x_hat = np.linspace(min(self.x), max(self.x), res)
        y_hat = self.predict(x_hat)
        return hv.Curve(data=(x_hat, y_hat))

    def __str__(self):
        return f"LinearFit(degree: {self.degree}, R2: {self.r2():.3f}, MSE: {self.mse():.3f})"

    def __repr__(self):
        return self.__str__()


class Measurement:

    def __init__(self, path: Union[Path, str], name: str, run: "BenchmarkRun", df=None):
        self.path = Path(path)
        self.name = name
        self.run = run
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
        self["energy"] = get_energy(self["power"], self["timestamp"]) / 1_000

    def calculate_edp(self):
        self["edp"] = self["energy"] * self["timestamp"]

    def plot(self, metric: str, label: str):
        return apply_plot_default_settings(self.data.hvplot(x="timestamp", y=metric, label=label))

    def __add__(self, other):
        result = self.data + other.data
        return Measurement(self.path.parent, self.name, self.run, result)

    def __sub__(self, other):
        result = self.data - other.data
        return Measurement(self.path.parent, self.name, self.run, result)

    def __truediv__(self, other):
        result = self.data / other.data
        return Measurement(self.path.parent, self.name, self.run, result)

    def __mul__(self, other):
        result = self.data * other.data
        return Measurement(self.path.parent, self.name, self.run, result)

    def r_join(self, other: "Measurement") -> "Measurement":
        joined = self.data.join(other.data, lsuffix="_sd")
        joined = joined.drop(joined.filter(regex='_sd$').columns.tolist(), axis=1)
        joined = joined.dropna()
        joined = Measurement(self.path.parent, "joined", self.run, joined)
        return joined


class BenchmarkRun:
    def __init__(self, path: Union[Path, str], benchmark: "Benchmark"):
        self.path = Path(path)
        self.name = self.path.name

        self.name = self.name.split(",")[-1]

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
            new_df[col] = pd.concat([x[col] for x in dfs], axis=1).aggregate(func, axis=1)  # .mean(axis=1)

        new_df = pd.DataFrame(new_df)
        return Measurement(self.path, f"{data_source}-{func}", self, df=new_df)

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
        self.add_joined_measurements()

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
        sd = Measurement(self.path / "gpu-power.csv", "sd", self.run)
        sd.data["timestamp"] = pd.to_datetime(sd.data["timestamp"])

        hd = Measurement(self.path / "total_power_samples.csv", "hd", self.run)
        hd.data = hd.data.rename(columns={"value": "power"})
        hd.data["timestamp"] = pd.to_datetime(hd.data["timestamp"], unit="us")

        measurements = [sd, hd]
        return name_dict_from_list(measurements)

    def add_joined_measurements(self):
        sd = self.measurements["sd"]
        hd = self.measurements["hd"]
        joined = sd.r_join(hd)
        self.measurements["joined"] = joined

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

    def plot(self, metric, data_source="joined"):
        return self.measurements[data_source].plot(metric=metric, label=self.run.name)

    def get_total_values(self, data_source: str = "joined") -> pd.Series:
        return self.measurements[data_source].data.iloc[-1].rename(self.name)

    def aggregate(self, data_source: str = "joined", func: str = "mean"):
        return self.measurements[data_source].data.aggregate(func)


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

    def plot(self, metric, data_slice="mean", data_source="joined"):
        plots = []
        if isinstance(data_slice, str):
            for key, run in self.runs.items():
                aggregate: Measurement = run.aggregate(data_source, data_slice)
                plot = aggregate.plot(metric=metric, label=key)
                plots.append(plot)
        else:
            for key, run in self.runs.items():
                bench_rep: BenchmarkRepetition = run[data_slice]
                plot = bench_rep.plot(metric=metric, data_source=data_source)
                plots.append(plot)
            pass

        return overlay_plots(plots).opts(legend_position='bottom_right')

    def plot_raw_power(self, data_slice, data_source="hd"):
        return self.plot(metric="power",
                         data_slice=data_slice, data_source=data_source)

    def plot_energy(self, data_slice, data_source="hd"):
        return self.plot(metric="energy",
                         data_slice=data_slice, data_source=data_source)

    def get_total_values(self, aggregate=None, data_source="joined"):
        dfs = []
        for key, run in self.runs.items():
            dfs.append(run.get_total_values(aggregate=aggregate,
                                            data_source=data_source))

        if aggregate is None:
            return pd.concat(dfs)
        else:
            df = pd.DataFrame(dfs)
            return df.set_index("run")

    def boxplot(self, metric):
        df = self.get_total_values()
        boxplot = df.hvplot.box(y=metric, by='run'
                                , legend=True)
        return apply_plot_default_settings(boxplot)

    def aggregate(self, data_source: str = "joined", func: str = "mean", func2: str = "mean"):
        data = []
        for name, run in self.runs.items():
            tmp = run.aggregate(data_source, func2).data.aggregate(func)
            tmp = dict(tmp)
            tmp["run"] = name
            data.append(tmp)

        data = pd.DataFrame(data)
        data = data.set_index("run")

        # use totals instead of aggregate for integrated values
        totals = self.get_total_values(aggregate=func, data_source=data_source)
        data["energy"] = totals["energy"]
        data["edp"] = totals["edp"]
        data["timestamp"] = totals["timestamp"]
        return data

    def plot_totals(self, x, y, aggregate="mean", data_source="joined"):
        totals = self.get_total_values(aggregate, data_source)
        return apply_plot_default_settings(totals.hvplot(x=x, y=y, hover_cols=["run"]))


    # def aggregate(self, data_source: str, func: str):
    #     dfs = [x.aggregate(data_source=data_source, func=func) for x in self.runs.values()]
    #     new_df = {}
    #     for col in dfs[0]:
    #         new_df[col] = pd.concat([x[col] for x in dfs], axis=1).aggregate(func, axis=1)  # .mean(axis=1)
    #
    #     new_df = pd.DataFrame(new_df)
    #     return Measurement(self.path, f"{data_source}-{func}", self, df=new_df)


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

    def plot(self, benchmark: str, metric, data_slice="mean", data_source="joined"):
        benchmarks = [x.benchmarks[benchmark] for x in self.experiments.values()]
        plots = [b.plot(metric, data_slice, data_source) for b in benchmarks]
        return apply_plot_default_settings(overlay_plots(plots))

    def boxplot(self, benchmark: str, metric, data_slice="mean", data_source="joined"):
        benchmarks = [x.benchmarks[benchmark] for x in self.experiments.values()]
        plots = [b.boxplot(metric) for b in benchmarks]
        return apply_plot_default_settings(overlay_plots(plots))

    def aggregate(self, benchmark: str, data_source: str = "joined", func: str = "mean", func2: str = "mean"):
        benchmarks = [x.benchmarks[benchmark] for x in self.experiments.values()]
        aggregates = [x.aggregate(data_source=data_source, func=func, func2=func2) for x in benchmarks]
        return pd.concat(aggregates)

    def get_total_values(self, benchmark: str, aggregate=None, data_source="joined"):
        benchmarks = [x.benchmarks[benchmark] for x in self.experiments.values()]
        total_values = [x.get_total_values(aggregate=aggregate, data_source=data_source) for x in benchmarks]
        return pd.concat(total_values)

    def __str__(self):
        return f"DataLoader({list(self.experiments.keys())})"

    def __repr__(self):
        return self.__str__()


def name_dict_from_list(data):
    return {x.name: x for x in data}


def get_energy(power, time):
    return integrate.cumtrapz(power, time, initial=0)
