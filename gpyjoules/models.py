import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt


def calculate_derived_metrics(df,p_baseline,p_max,p_max_nominal, util):
    df["total_power"] = df["power"] + p_baseline
    df["p_util"] = df.groupby("benchmark")["power"].transform(lambda power: get_activity_from_power(power=power, p_max=p_max, p_max_nominal=p_max_nominal))
    df["util_gpu_norm"] = df["util-gpu"] / 100

    df["time_reference"] = df.groupby("benchmark")["timestamp"].transform(lambda time: time.loc[f"{p_max_nominal}W"].mean())
    df["delay"] = df["timestamp"] / df["time_reference"]  #all_agg.groupby("benchmark")["timestamp"].transform(lambda time: time/time.loc["300W"].mean())

    # all_agg["energy_reference"] = all_agg.groupby("benchmark")["energy"].transform(lambda energy: energy.loc[f"{p_max}W"].mean())
    # all_agg["relative_energy"] = all_agg["energy"] / all_agg["energy_reference"]#all_agg.groupby("benchmark")["energy"].transform(lambda energy: energy/energy.loc["300W"].mean())

    df["relative_clock_gpu"] = df["clock_gpu"] / f_max

    df["relative_power"] = np.clip(df["enforced_power_limit"] / df[util], a_max=p_max, a_min=None)
    df["p_util2"] = df["power"] / (df["enforced_power_limit"] - p_baseline)
    df["util_combined"] = df[["p_util", "util-gpu"]].mean(axis=1)

class ClockModel:
    def __init__(self, p_max, f_max):
        self.p_max = p_max
        self.f_max = f_max
        self._model = None

    def _prepare(self, data):
        # data["relative_clock_gpu"] = data["clock_gpu"] / f_max
        data["relative_power"] = np.clip(data["power_limit"] / data["p_util"], a_max=self.p_max, a_min=None)

    def fit(self, power_limit, p_util, clock_gpu):
        data = pd.DataFrame({"power_limit": power_limit,
                             "clock_gpu": clock_gpu,
                             "p_util": p_util})
        self._prepare(data)  # power sm.ols(formula="relative_clock_gpu ~ relative_power", data=data).fit()
        # self._model = sm.ols(formula="power ~ relative_power", data=data).fit()
        self._model = sm.ols(formula="clock_gpu ~ relative_power", data=data).fit()

    def predict(self, power_limit, p_util):
        data = pd.DataFrame({"power_limit": power_limit,
                             "p_util": p_util})
        self._prepare(data)
        return np.clip(self._model.predict(data), a_max=self.f_max, a_min=None)  # * f_max

    def plot(self, training_data, util, x="enforced_power_limit"):
        training_data = training_data.copy()
        sns.scatterplot(x=x, y="clock_gpu", hue="benchmark", data=training_data)
        training_data["clock_gpu_hat"] = self.predict(training_data["enforced_power_limit"], training_data[util])
        sns.lineplot(x=x, y="clock_gpu_hat", hue="benchmark", data=training_data)

    def plot_raw(self, training_data, util):
        training_data = training_data.copy()
        training_data["p_util"] = training_data[util]
        training_data = training_data.rename(columns={"enforced_power_limit": "power_limit"})
        self._prepare(training_data)
        training_data["clock_gpu_hat"] = self.predict(training_data["power_limit"], training_data[util])

        sns.scatterplot(x="relative_power", y="clock_gpu", hue="benchmark", data=training_data)
        sns.lineplot(x="relative_power", y="clock_gpu_hat", data=training_data)

    @property
    def rsquared(self):
        return self._model.rsquared


class PowerModel:
    def __init__(self, baseline):
        self._model = None
        self.baseline = baseline

    @staticmethod
    def _prepare(data):
        data["p1"] = data["p_util"] * data["clock_gpu"] ** 3
        data["p2"] = data["p_util"] * data["clock_gpu"] ** 2
        data["p3"] = data["p_util"] * data["clock_gpu"]

    def fit(self, p_util, clock_gpu, power):
        data = pd.DataFrame({
            "p_util": p_util,
            "clock_gpu": clock_gpu,
            "power": power
        })
        self._prepare(data)
        self._model = sm.ols(formula="power ~ p1 + p2 + p3", data=data).fit()

    def _df4pred(self, p_util, clock_gpu):
        data = pd.DataFrame({
            "p_util": p_util,
            "clock_gpu": clock_gpu,
        })
        return data

    def predict(self, p_util, clock_gpu):
        data = self._df4pred(p_util, clock_gpu)
        self._prepare(data)
        return np.clip(self._model.predict(data), a_min=0, a_max=None) + self.baseline

    @property
    def rsquared(self):
        return self._model.rsquared

    def plot(self, training_data, util,p_max_nominal, x="clock_gpu", hue: str = "benchmark"):
        training_data = training_data.copy()
        all_predictions = []
        for b in training_data["benchmark"].unique():
            util_selection = training_data[training_data["benchmark"] == b]
            x_pred = np.linspace(util_selection["clock_gpu"].min(), util_selection["clock_gpu"].max(), 100)
            y_pred = self.predict(util_selection.loc[f"{p_max_nominal}W"][util].mean(), x_pred)
            df = self._df4pred(util_selection.loc[f"{p_max_nominal}W"][util].mean(), x_pred)
            df["benchmark"] = b#util_selection["benchmark"].unique()[0]
            df["power_hat"] = y_pred
            all_predictions.append(df)

        all_predictions = pd.concat(all_predictions)

        sns.scatterplot(x=x, y="total_power", hue=hue, data=training_data)
        sns.lineplot(x=x, y="power_hat", data=all_predictions,
                     hue=hue)  # , style=list("xxxxxxxxxxxxxxx"), markers={"x":"x"}) ,hue="benchmark"


class DelayModel:
    def __init__(self):
        self._model = None

    @staticmethod
    def _prepare(data):
        data["d1"] = 1 / data["clock_gpu"]

    def fit(self, clock_gpu, delay):
        data = pd.DataFrame({
            "clock_gpu": clock_gpu,
            "delay": delay
        })
        self._prepare(data)
        self._model = sm.ols(formula="delay ~ d1", data=data).fit()

    def predict(self, clock_gpu):
        data = pd.DataFrame({
            "clock_gpu": clock_gpu
        })
        self._prepare(data)
        return self._model.predict(data)

    @property
    def rsquared(self):
        return self._model.rsquared

    def plot(self, training_data, x="clock_gpu", hue: str = "benchmark"):
        training_data = training_data.copy()

        x_pred = np.arange(training_data[x].min(), training_data[x].max(), 1)
        predictions = self.predict(x_pred)

        sns.scatterplot(x=x, y="delay", hue="benchmark", data=training_data)
        sns.lineplot(x=x_pred,
                     y=predictions)  # , style=list("xxxxxxxxxxxxxxx"), markers={"x":"x"}) ,hue="benchmark"


class EnergyModel:
    def __init__(self, delay_model, power_model, clock_model, p_min, p_max):
        self.delay_model = delay_model
        self.power_model = power_model
        self.clock_model = clock_model
        self.p_max = p_max
        self.p_min = p_min

    def predict(self, power_limit, p_util, time_reference):
        data = pd.DataFrame({
            "power_limit": power_limit,
            "p_util": p_util,
            "time_reference": time_reference
        })
        data["clock_gpu_hat"] = self.clock_model.predict(power_limit, p_util)
        data["power_hat"] = self.power_model.predict(p_util, data["clock_gpu_hat"])
        data["delay_hat"] = self.delay_model.predict(data["clock_gpu_hat"])

        data["time_hat"] = data["time_reference"] * data["delay_hat"]
        return (data["power_hat"] * data["time_hat"]) / 1_000

    def plot(self, training_data, x="enforced_power_limit", hue: str = "benchmark"):
        training_data = training_data.copy()

        training_data["energy"] = training_data["energy"] + (
                    self.power_model.baseline * training_data["timestamp"]) / 1000

        training_data["energy_hat"] = self.predict(p_util=training_data["p_util"],
                                                   power_limit=training_data["enforced_power_limit"],
                                                   time_reference=training_data["time_reference"])

        sns.scatterplot(x=x, y="energy", hue="benchmark", data=training_data)
        sns.lineplot(x=x,
                     y="energy_hat", hue="benchmark",
                     data=training_data)  # , style=list("xxxxxxxxxxxxxxx"), markers={"x":"x"}) ,hue="benchmark"
        plt.legend(training_data["benchmark"].unique())
        plt.xlabel("Power Limit [W]")
        plt.ylabel("Energy [kJ]")

    def visualize(self, util_min=30, util_max=100, util_step=10):
        util_range = np.arange(util_min, util_max + util_step, util_step) / 100
        x = np.arange(self.power_model.baseline, self.p_max)
        for i in reversed(util_range):
            y = self.predict(x, i, 1) * 1000
            plt.plot(x, y, label=f"{int(i * 100)}%")
            min_e = np.min(y)
            min_x = np.argmin(y)
            plt.plot(x[min_x], min_e, "x")

        # sns.scatterplot(all_agg["enforced_power_limit"],all_agg["energy"]/all_agg["time_reference"], style=all_agg["benchmark"])
        plt.vlines([self.p_min],ymin=self.p_min,ymax=self.p_max)
        plt.legend(title="Utilisation")
        plt.xlabel("Power Limit [W]")
        plt.ylabel("Energy [J]")


class EnergyDelayProductModel:
    def __init__(self, delay_model, clock_model, energy_model):
        self.delay_model = delay_model
        self.clock_model = clock_model
        self.energy_model = energy_model
        self.p_max = energy_model.p_max
        self.p_min = energy_model.p_min

    def predict(self, power_limit, p_util, time_reference):
        data = pd.DataFrame({
            "power_limit": power_limit,
            "p_util": p_util,
            "time_reference": time_reference
        })
        data["clock_gpu_hat"] = self.clock_model.predict(power_limit, p_util)
        data["delay_hat"] = self.delay_model.predict(data["clock_gpu_hat"])
        data["energy"] = self.energy_model.predict(p_util=data["p_util"],
                                                   power_limit=data["power_limit"],
                                                   time_reference=data["time_reference"])

        data["time_hat"] = data["time_reference"] * data["delay_hat"]
        return data["energy"] * data["time_hat"] 

    def plot(self, training_data, x="enforced_power_limit", hue: str = "benchmark"):
        training_data = training_data.copy()

        training_data["energy"] = training_data["energy"] + (
                    self.energy_model.power_model.baseline * training_data["timestamp"]) / 1000
        training_data["edp"] = training_data["energy"] * training_data["timestamp"]

        training_data["edp_hat"] = self.predict(p_util=training_data["p_util"],
                                                power_limit=training_data["enforced_power_limit"],
                                                time_reference=training_data["time_reference"])

        sns.scatterplot(x=x, y="edp", hue="benchmark", data=training_data)
        sns.lineplot(x=x,
                     y="edp_hat", hue="benchmark",
                     data=training_data)  # , style=list("xxxxxxxxxxxxxxx"), markers={"x":"x"}) ,hue="benchmark"
        plt.xlabel("Power Limit [W]")
        plt.ylabel("EDP [kJs]")

    def visualize(self, util_min=30, util_max=100, util_step=10):
        util_range = np.arange(util_min, util_max + util_step, util_step) / 100
        x = np.arange(self.p_min/2, self.p_max)
        for i in reversed(util_range):
            y = self.predict(x, i, 1) * 1000
            plt.plot(x, y, label=f"{int(i * 100)}%")
            min_edp = np.min(y)
            min_x = np.argmin(y)
            plt.plot(x[min_x], min_edp, "x")

        plt.vlines([self.p_min], ymin=100, ymax=250)
        # sns.scatterplot(all_agg["enforced_power_limit"],all_agg["energy"]/all_agg["time_reference"], style=all_agg["benchmark"])
        plt.legend(title="Utilisation")
        plt.xlabel("Power Limit [W]")
        plt.ylabel("EDP [Js]")
