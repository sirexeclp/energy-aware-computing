import copy
from pathlib import Path
import pandas as pd
import json
from typing import Union, List, Dict, Optional
import holoviews as hv
import hvplot.pandas
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d

from src.util import TimestampInfo


class BenchmarkRun:
    def __init__(self, path: Union[Path, str], benchmark: "Benchmark"):
        self.path = Path(path)
        self.name = self.path.name
        self.benchmark = benchmark
        self.experiment = self.benchmark.experiment
        self.timestamps = self._load_timestamps()
        self.files = self._list_files()
        self.data_sd = self._load_sd_data()
        self.system_info = self._load_system_info()
        self.power_hd = self._load_hd_power()
        self.start_index = self._get_start_index()

        self.prepare_data()

    def __str__(self):
        return f"BenchmarkRun({self.name})"

    def __repr__(self):
        return self.__str__()

    def prepare_data(self):
        self.data_sd = self.normalize_timestamps(self.data_sd)
        self.power_hd = self.normalize_timestamps(self.power_hd)

        self.data_sd["power"] = self.data_sd["power"] / 1_000
        self.power_hd["power"] = self.power_hd["power"] / 1_000

        self.calculate_energy(self.data_sd)
        self.calculate_energy(self.power_hd)

        # self.power_hd["power"] = self.interpolate(self.power_hd.timestamp, self.power_hd.power, 1/50)

    @staticmethod
    def calculate_energy(df):
        df["energy"] = get_energy(df["power"], df["timestamp"])

    def _list_files(self) -> List[Path]:
        return list(self.path.glob("*"))

    def _load_timestamps(self) -> Optional[TimestampInfo]:
        path = self.path / "timestamps.csv"
        if path.exists():
            return TimestampInfo(pd.read_csv(path))
        else:
            return None

    def _load_sd_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.path / "gpu-power.csv", index_col=0)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def _load_system_info(self) -> dict:
        with open(self.path / "system_info.json", "r") as f:
            return json.load(f)

    def _load_hd_power(self) -> pd.DataFrame:
        df = pd.read_csv(self.path / "total_power_samples.csv", index_col=0)
        df = df.rename(columns={"value": "power"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="us")
        return df

    def _get_start_index(self) -> np.datetime64:
        if self.timestamps is not None and self.timestamps.experiment_begin is not None:
            return self.timestamps.experiment_begin
        else:
            return max(self.data_sd.timestamp.min(), self.power_hd.timestamp.min())

    def normalize_timestamps(self, df):
        df = df.copy(deep=True)
        df["timestamp"] = (df["timestamp"] - self.start_index) / np.timedelta64(1, 's')
        df = df[df.timestamp >= 0]
        return df

    @staticmethod
    def interpolate(x, y, step):
        x = np.array(x)
        y = np.array(y)
        x_tick = np.arange(x[0], x[-1], step)
        f_cubic = interp1d(x, y, kind="cubic")
        return x_tick, f_cubic(x_tick)

    def remove_setup_data(self) -> "BenchmarkRun":
        result = copy.deepcopy(self)
        result.power_hd = result.power_hd[result.power_hd.timestamp >= result.get_start_index()]
        result.data_sd = result.data_sd[result.data_sd.timestamp >= result.get_start_index()]
        return result
    # @property
    # def energy(self, resolution="hd"):
    #     if resolution == "hd":
    #         power_hd = self.power_hd
    #         timestamps = power_hd["timestamp"]
    #         power = power_hd["value"]
    #
    #     elif resolution == "sd":
    #
    #     else:
    #         raise ValueError(f"resolution must be either hd or sd")


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
        return dict(map(lambda x: (x.name, x), [BenchmarkRun(x, self) for x in self.path.glob("*")]))


class Experiment:
    def __init__(self, data_root: Union[str, Path]):
        self.data_root = Path(data_root)
        self.experiment_path = list(self.data_root.glob("*"))[0]
        self.name = self.experiment_path.name
        self.benchmarks = self.load_benchmarks()

    def load_benchmarks(self) -> Dict[str, Benchmark]:
        return dict(map(lambda x: (x.name, x), [Benchmark(x, self) for x in self.experiment_path.glob("*")]))

    def __str__(self):
        return f"Experiment({self.name})"

    def __repr__(self):
        return self.__str__()


def get_energy(power, time):
    return integrate.cumtrapz(power, time, initial=0)


# class CumEnergyComparison:
#
#     @staticmethod
#     def plot(benchmark: Benchmark):
#         for name, run in benchmark.runs.items():


if __name__ == '__main__':
    e = Experiment("../data/dgx")
    print(e)
    print(e.benchmarks)
    w150 = e.benchmarks["resnet"].runs["150W"]  # ["value"].hvplot()
    print(w150.power_hd.head())
    print(w150.data_sd.timestamp.dtype)
    # print(w150.timestamps)
    print(type(w150.get_start_index()))
    print(w150.data_sd[w150.data_sd.timestamp >= w150.get_start_index()])
    w150 = w150.remove_setup_data()

    # energy1 = get_energy(w150, w150["timestamp"])
    # energy2 = get_cumulative_energy(w150["value"], w150["timestamp"])
    # # np.testing.assert_allclose(energy1[0:-1], energy2[1:-1])
    # print(len(energy1))
    # print(len(w150))
    # print()
    # print(energy2)
    # plot = pd.DataFrame({"a":energy2, "b":energy1}).hvplot()
    # #print(plot.head())
    # hvplot.show(plot)
