import itertools
import json
import re
from functools import reduce
from pathlib import Path
from typing import Union, List, Dict, Optional, Sequence, Tuple, Iterable, AnyStr, Any

import holoviews as hv
import numpy as np
import pandas as pd
from bokeh.core.property.container import Seq
from holoviews import Dimension
from scipy import integrate
from scipy.interpolate import interp1d

from gpyjoules.util import TimestampInfo
import matplotlib.pyplot as plt
import seaborn as sns

def get_energy(power: Union[np.ndarray, Iterable], time: Union[np.ndarray, Iterable]) -> np.ndarray:
    """Use the trapezoidal method to integrate power over time.

    Args:
        power: a sequence of power measurements
        time: a sequence of timestamps, at which the measurements took place

    Returns: the cumulative integral of the power, aka. the cumulative energy

    """
    return integrate.cumtrapz(power, time, initial=0)


# class Dimension(NamedTuple):
#     label: str
#     unit: Optional[str]

def name_dict_from_list(data: List):
    """Create a dictionary from a list,
    where each key is the `name` attribute
    of the list item and the value is the
    list item itself.

    Args:
        data: a list, which entries have a name attribute

    Returns: a dictionary mapping names to list items

    Examples:

        class Item(NamedTuple):
            name: str
            x: int


       my_list = [Item("A", 0), Item("B", 3)]
       my_dict = name_dict_from_list(my_list)
       my_dict == {"A": Item("A", 0), "B": Item("B", 3)}

    """

    return {x.name: x for x in data}


dimensions = [
    Dimension("power", label="Power", unit="W"),
    Dimension("clock-gpu", label="GPU Clock", unit="MHz"),
    Dimension("clock", label="GPU Clock", unit="MHz"),
    Dimension("edp", label="Energy Delay Product (EDP)", unit="kJS"),
    Dimension("energy", label="Energy", unit="kJ"),
    Dimension("run", label="Limit"),
    Dimension("timestamp", label="Time", unit="s"),
]

label_replacements = name_dict_from_list(dimensions)

# {
#     "power": Dimension("Power", "W"),
#     "clock-gpu": Dimension("GPU Clock", "MHz"),
#     "clock": Dimension("GPU Clock", "MHz"),
#     "edp": Dimension("Energy Delay Product (EDP)", "kJS"),
#     "energy": Dimension("Energy", "kJ"),
#     "run": Dimension("Limit", None),
#     "timestamp": Dimension("Time", "s")
# }
benchmark_description = {
    "bert": "BERT Finetuning"
}


def overlay_plots(plots: List):
    """Overlay (combine using * operator) multiple
    holoviews plots.

    Args:
        plots: a list of hv-plots

    Returns: a singe hv-plot which contains all input plots

    """
    return reduce(lambda x, y: x * y, plots)
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # fig.axes.extend(plots)
    # return fig


def make_title(plot, benchmark: Union["Benchmark", str]) -> None:
    """Generate a title for a hv-plot, based on dimension labels and benchmark name.
    Args:
        plot: The plot to generate and set the title on.
        benchmark: a ``Benchmark`` object or name of the benchmark

    Examples:
        y vs. x (benchmark_name)

    """
    if isinstance(benchmark, Benchmark):
        benchmark = benchmark.name
    x, y, *_ = plot.dimensions()
    title = f"{y.label} vs. {x.label} ({benchmark_description.get(benchmark, benchmark)})"
    plot.opts(title=title)
    return None


def replace_dimension_labels(plot) -> None:
    """Replace dimension labels in the given plot to be more meaningful.
    Replacements are defined in the global `label_replacements` dict.

    Args:
        plot: a holoviews plot to replace dimension labels on

    """
    for dim in plot.dimensions():
        # if dim.label == "run":
        #     dim_values = plot.dimension_values(dim.label)
        replacement = label_replacements.get(dim.name, None)
        if replacement is not None:
            dim.label = replacement.label
            dim.unit = replacement.unit


def apply_plot_default_settings(plot):
    """Applys some default settings (width, height) to the given hv plot.
    Also replaces dimension labels.

    Args:
        plot: a holoviews plot

    Returns: the plot

    """
    replace_dimension_labels(plot)
    # make_title(plot)
    return plot.opts(aspect=16/9, fig_bounds=(0, 0, 2, 2))#.opts(width=800, height=450)


def atoi(text: str) -> Union[int, str]:
    """Converts the given text to an int if possible.

    Args:
        text: a string, which should be converted to an int

    Returns: the int representation if possible, the given string otherwise

    """
    return int(text) if text.isdigit() else text


# Source: https://stackoverflow.com/a/5967539/7997186
def natural_keys(text: str) -> List[Union[int, str]]:
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def interpolate(x: Union[np.ndarray, List, pd.Series], y: Union[np.ndarray, List, pd.Series], freq: float,
                start: float = 0):
    """Interpolate the given signal y, which is sampled at points x using cubic interpolation
    and sample a new signal at regular intervals defined by 1/freq, starting at the value start.

    Args:
        x: the timestamps of the given samples
        y: the signal samples to be interpolated (amplitude)
        freq: the frequency of the resulting sample of the interpolation
        start: the timestamp or x value to start interpolating from until the end of the original signal is reached

    Returns: an interpolated signal with equally spaced samples

    """
    x = np.array(x)
    y = np.array(y)
    x_tick = np.arange(start, x[-1], 1 / freq)
    f_cubic = interp1d(x, y, kind="cubic")
    return x_tick, f_cubic(x_tick)


def interpolate_df(df: pd.DataFrame, freq: float, start=0):
    """Interpolate signals in a dataframe.
    All rows in the dataframe are interpolated.
    The index is treated as sample spacing/timestamp (x).

    Args:
        df: a dataframe to interpolate
        freq: the frequency of the resulting sample of the interpolation
        start: the timestamp or x value to start interpolating from until the end of the original signal is reached

    Returns: a new dataframe with interpolated rows

    """
    x = df.index
    new_df = {}
    for col in df:
        y = df[col]
        x_hat, y_hat = interpolate(x, y, freq, start)
        new_df[col] = y_hat

    new_df[x.name] = x_hat
    new_df = pd.DataFrame(new_df)
    return new_df


def normalize_timestamp(x: Sequence[pd.Timestamp], start_time: pd.Timestamp):
    """Normalizes a timestamp with a fixed start time.
    Turns a collection of absolute timestamps (datetime) into
    relative time (in seconds) since `start_time`.

    Args:
        x: a collection of timestamps
        start_time: a timestamp

    Returns: a collection of relative timestamps in seconds since `start_time`

    """
    return (x - start_time) / np.timedelta64(1, 's')


def identity_function(x: Any) -> Any:
    """A simple identity function.
    """
    return x


class LinearFit:
    """A wrapper around np.polyfit and np.poly1d
    to provide convenient linear regression.
    """

    def __init__(self, x, y, degree=2, kernel=None):
        """Fit a new regression.

        Args:
            x: array of x values
            y: array of y values
            degree: degree of the polynomial to fit
            kernel: a kernel function which is applied element wise before calculating the regression
        """
        self.x = x
        self.y = y
        self.degree = degree
        if kernel is None:
            self.kernel = identity_function
        else:
            self.kernel = kernel

        self.coefficients = np.polyfit(self.kernel(x), y, degree)
        self.poly = np.poly1d(self.coefficients)

    def __call__(self, arg: Sequence) -> np.ndarray:
        self.predict(arg)

    def predict(self, x_hat: Sequence) -> np.ndarray:
        """Predict values for x_hat.

        Args:
            x_hat: a collection of values to predict

        Returns:

        """
        return self.poly(self.kernel(x_hat))

    def r2(self, test_data: Sequence = None) -> float:
        """Calculate the coefficient of determination on the training data or test_data, if provided.

        Args:
            test_data: optional test_data to calculate r2 on

        Returns:
            The coefficient of determination

        """
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

    def mse(self, test_data: Sequence = None) -> float:
        """Calculate the mean squared error on the training data or test_data, if provided.

        Args:
            test_data: optional test_data to calculate mse on

        Returns: the mean squared error (mse)

        """
        if test_data is None:
            x = self.x
            y = self.y
        else:
            x, y = test_data

        y_hat = self.predict(x)
        residuals = y_hat - y
        return np.mean(residuals ** 2)

    def plot(self, res: int = 100, x_range: Optional[Tuple[Union[float, int], Union[float, int]]] = None) -> hv.Curve:
        """Returns a holoviews curve, to visualize the fitted polynomial.

        Args:
            res: the resolution or data points to draw between min and max of the dataset.
            x_range: optional specify the x range to draw

        Returns:

        """
        if x_range is None:
            x_min = min(self.x)
            x_max = max(self.x)
        else:
            x_min, x_max = x_range

        x_hat = np.linspace(x_min, x_max, res)
        y_hat = self.predict(x_hat)
        return hv.Curve(data=(x_hat, y_hat))

    def __str__(self):
        return f"LinearFit(degree: {self.degree}, R2: {self.r2():.3f}, MSE: {self.mse():.3f})"

    def __repr__(self):
        return self.__str__()


class Measurement:
    """A Measurement is the smallest organizational unit in this data structure.
    It wraps around a dataframe and provides methods to calculate basic metrics, such as
    energy, and edp.
    Also methods for normalization and interpolation are provided to facilitate
    computations and combinations with multiple different measurements.

    """

    def __init__(self, path: Union[Path, str], name: str, run: "BenchmarkRun", df: Optional[pd.DataFrame] = None):
        self.path = Path(path)
        self.name = name
        self.run = run
        self.original_timestamp = None
        if df is None:
            self.data = self.load()
        else:
            self.data = df

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def load(self) -> pd.DataFrame:
        """Load the pandas dataframe from the path.

        Returns: a pandas dataframe

        """
        df = pd.read_csv(self.path, index_col=0)
        return df

    def normalize_timestamps(self, start_time: pd.Timestamp) -> None:
        """Normalize the timestamps of the internal dataframe using the `normalize_timestamp` function.

        Args:
            start_time: the reference for normalization
        """
        self["timestamp"] = normalize_timestamp(self["timestamp"], start_time)
        self.original_timestamp = self.data.timestamp[self.data.timestamp > 0]
        # df = df[df.timestamp >= 0]

    def interpolate(self, freq: float, start: float = 0) -> None:
        """Interpolate all columns in the internal dataframe using the `interpolate_df` function.

        Args:
            freq:
            start:
        """
        self.data = interpolate_df(self.data.set_index("timestamp"), freq, start)

    def calculate_energy(self) -> None:
        """Calculate the cumulative energy (returned in kJ) using the columns power (W) and timestamp (s).
        The result is stored in the column `energy`.
        """
        self["energy"] = get_energy(self["power"], self["timestamp"]) / 1_000

    def calculate_edp(self):
        """Calculate the energy delay product (kJs) using the columns energy(kj) and timestamp(s).
        The cumulative energy delay product is stored in the column edp.
        """
        self["edp"] = self["energy"] * self["timestamp"]

    def plot(self, metric: str, label: str):
        """Use holoviews to generate a plot of the given metric, with time on the x axis.

        Args:
            metric: a metric contained in this measurement (eg. energy, power, edp)
            label: a label for this plot/curve

        Returns: a holoviews plot of the metric with default settings applied

        """
        return apply_plot_default_settings(hv.Curve(self.data.set_index("timestamp")[metric], label=label))

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
        """Right-Join two Measurements.
        This can be used to combine hd and sd measurements.

        Args:
            other: an other measurement object

        Returns: a new measurement object with the joined data-frame

        """
        joined = self.data.join(other.data, lsuffix="_sd")
        joined = joined.drop(joined.filter(regex='_sd$').columns.tolist(), axis=1)
        joined = joined.dropna()
        joined = Measurement(self.path.parent, "joined", self.run, joined)
        return joined


class BenchmarkRun:
    """A BenchmarkRun is a concrete configuration of benchmark, ie. a benchmark (such as 'bert')
    with a power or clock limit (such as 150W) configured.

    """

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
        """Aggregate data from multiple benchmark repetitions.

        Args:
            data_source: selects the measurement of each repetition to be used
            func: an aggregation function to be applied across all repetitions

        Returns:

        """
        dfs = [x.measurements[data_source].data for x in self.repetitions]  # [:-1]
        new_df = {}
        for col in dfs[0]:
            new_df[col] = pd.concat([x[col] for x in dfs], axis=1).aggregate(func, axis=1)  # .mean(axis=1)

        new_df = pd.DataFrame(new_df)
        return Measurement(self.path, f"{data_source}-{func}", self, df=new_df)

    def get_total_values(self, aggregate: Union[str, callable] = None, data_source: str = "joined") -> pd.DataFrame:
        """Gets the total (or last values of cumulative series) for all repetitions of this BenchmarkRun.

        Args:
            aggregate: if provided, the totals of the repetitions will be aggregated using this function
            data_source: selects, which data_source or measurement should be used

        Returns: a dataframe with the total values

        """
        values = [r.get_total_values(data_source) for r in self.repetitions]
        df = pd.DataFrame(values)
        if aggregate is not None:
            df = df.apply(aggregate)
        df["run"] = self.name
        return df

    def get_original_timestamps(self, data_source: str) -> List[float]:
        ts = [x.get_original_timestamps(data_source) for x in self.repetitions]
        return list(itertools.chain.from_iterable(ts))

    def exclude_outliers(self, percentile: float = 0.95) -> None:
        """Exclude repetitions from this run, which took longer than the .95 percentile.

        Args:
            percentile: the cutoff percentile for exclusion

        """
        total_times = [x.get_total_values().timestamp for x in self.repetitions]
        outlier_indices = np.argwhere(np.quantile(total_times, percentile) < total_times)
        for [x] in reversed(outlier_indices):
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

    def prepare_data(self) -> None:
        """Prepare data of this repetition by:
            - normalizing timestamps
            - interpolating
            - converting power from milliwatt to watt
            - calculating energy
            - calculating edp

        """
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
        """List all files in this repetition.
        Returns: a list of paths

        """
        return list(self.path.glob("*"))

    def get_original_timestamps(self, data_source: str) -> Sequence[float]:
        return np.diff(self.measurements[data_source].original_timestamp)


    def load_measurements(self) -> Dict[str, "Measurement"]:
        """Load all measurements (sd and hd).
        Returns: a dict mapping names to measurement objects

        """
        sd = Measurement(self.path / "gpu-power.csv", "sd", self.run)
        sd.data["timestamp"] = pd.to_datetime(sd.data["timestamp"])

        hd = Measurement(self.path / "total_power_samples.csv", "hd", self.run)
        hd.data = hd.data.rename(columns={"value": "power"})
        hd.data["timestamp"] = pd.to_datetime(hd.data["timestamp"], unit="us")

        measurements = [sd, hd]

        external_path = self.path / "power-external.csv"
        if external_path.exists():
            external = Measurement(external_path, "external", self.run)
            external.data["timestamp"] = pd.to_datetime(external.data["timestamp"])
            external.data["power"] = external.data.filter(regex=r"d\dc\d").sum(axis=1)
            measurements.append(external)


        return name_dict_from_list(measurements)

    def add_joined_measurements(self) -> None:
        """Join sd and hd measurements and add them to the list of measurements.
        """
        sd = self.measurements["sd"]
        hd = self.measurements["hd"]
        joined = sd.r_join(hd)
        self.measurements["joined"] = joined

    def _load_timestamps(self) -> Optional[TimestampInfo]:
        """Load the event timestamps.
        Returns: the timestamps

        """
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
        """Load the system-info json.
        Returns: the system info (as a dict)

        """
        with open(self.path / "system_info.json", "r") as f:
            return json.load(f)

    # def _load_hd_power(self) -> pd.DataFrame:
    #     # df = pd.read_csv(self.path / "total_power_samples.csv", index_col=0)
    #     # df = df.rename(columns={"value": "power"})
    #     df["timestamp"] = pd.to_datetime(df["timestamp"], unit="us")
    #     return df

    def _get_start_index(self) -> np.datetime64:
        """Find the start index to sync up all measurements.
        If timestamps are present and the train_begin event was recorded,
        it will be used as start marker.
        Alternatively the first timestamp of the measurement, which started last will be chosen,
        so all measurements are available for all points in time.
        Returns: the start timestamp

        """
        if self.timestamps is not None and self.timestamps.train_begin is not None:
            return self.timestamps.get_epoch_begin(1)
        # if self.timestamps is not None and self.timestamps.train_begin is not None:
        #     return self.timestamps.train_begin
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

        plot = overlay_plots(plots).opts(legend_position='bottom_right')
        make_title(plot, self)
        return apply_plot_default_settings(plot)

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
        totals = self.get_total_values(aggregate, data_source).sort_values(x)
        return apply_plot_default_settings(hv.Curve(totals.set_index(x)[y], hover_cols=["run"]))

    def get_original_timestamps(self, data_source: str) -> List[float]:
        ts = [x.get_original_timestamps(data_source) for x in self.runs.values()]
        return list(itertools.chain.from_iterable(ts))

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

    def get_original_timestamps(self, data_source: str) -> List[float]:
        ts = [x.get_original_timestamps(data_source) for x in self.benchmarks.values()]
        return list(itertools.chain.from_iterable(ts))


class DataLoader:
    """This is the highest class in the data hierarchy.
    The DataLoader loads data in the given data_root directory
    from all experiments.
    It provides convenience functions to calculate and plot aggregated results.

    """

    def __init__(self, data_root: Union[str, Path]):
        self.data_root = Path(data_root)
        self.experiments = self._load_experiments()

    def _load_experiments(self):
        experiment_paths = list(self.data_root.glob("*"))
        experiments = [Experiment(path) for path in experiment_paths]
        return name_dict_from_list(experiments)

    def plot(self, benchmark: str, metric, data_slice="mean", data_source="joined"):
        benchmarks: List[Benchmark] = [x.benchmarks[benchmark] for x in self.experiments.values()]
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

    def plot_totals(self, benchmark: str, x: str, y: str, aggregate="mean", data_source="joined"):
        totals = self.get_total_values(benchmark, aggregate, data_source).sort_values(x)
        scatter = hv.Scatter(totals.set_index(x)[y], hover_cols=["run"])
        scatter = apply_plot_default_settings(scatter)
        make_title(scatter, benchmark)
        return scatter  # * hv.Curve(scatter))

    def get_original_timestamps(self, data_source: str) -> List[float]:
        ts = [x.get_original_timestamps(data_source) for x in self.experiments.values()]
        return list(itertools.chain.from_iterable(ts))

    def plot_sample_frequency_distribution(self, data_source: str):
        # this way of calculating includes some erroneous points between the different runs
        ts = self.get_original_timestamps(data_source)
        freq = 1/np.array(ts)
        sns.distplot(freq)
        plt.title(f"Distribution of Sample Frequency ({data_source})")
        plt.xlabel("Sample Frequency (Hz)")
        plt.ylabel("Density")
        return np.mean(freq), np.std(freq), np.quantile(freq, 0.05), np.quantile(freq, 0.25),np.quantile(freq, 0.5),np.quantile(freq, 0.75)

    def __str__(self):
        return f"DataLoader({list(self.experiments.keys())})"

    def __repr__(self):
        return self.__str__()
