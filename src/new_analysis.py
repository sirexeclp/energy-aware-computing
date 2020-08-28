from pathlib import Path
import pandas as pd
import json
from typing import Union, List, Dict, Optional
import holoviews as hv
import hvplot.pandas
import numpy as np
from scipy import integrate


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

    def __str__(self):
        return f"BenchmarkRun({self.name})"

    def __repr__(self):
        return self.__str__()

    def _list_files(self) -> List[Path]:
        return list(self.path.glob("*"))

    def _load_timestamps(self) -> Optional[Path]:
        path = self.path / "timestamps.csv"
        if path.exists():
            return pd.read_csv(path)
        else:
            return None

    def _load_sd_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.path / "gpu-power.csv", index_col=0)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["energy"] = get_energy(df["power"], df["timestamp"])
        return df

    def _load_system_info(self) -> dict:
        with open(self.path / "system_info.json", "r") as f:
            return json.load(f)

    def _load_hd_power(self) -> pd.DataFrame:
        df = pd.read_csv(self.path / "total_power_samples.csv", index_col=0)
        df = df.rename(columns={"value": "power"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="us")
        df["energy"] = get_energy(df["power"], df["timestamp"])
        return df

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
    print(w150.data_sd.head())
    print(w150.timestamps)
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
