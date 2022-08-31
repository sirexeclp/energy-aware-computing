import hvplot
import hvplot.pandas
import holoviews as hv
from bokeh.models import HoverTool
from gpyjoules.util import TimestampInfo
from analysis.new_analysis import *
import bokeh.themes
from pathlib import Path
import pandas as pd
import json
import numpy as np
import holoviews as hv
from holoviews import opts
import seaborn as sns


def proof4theorem1(dgx_e):
    # "proof" for theorem 1

    e = "power-limit"

    hv.extension("matplotlib")
    myplot = (
        dgx_e[e]
        .benchmarks["mnist-cnn"]
        .plot("power", data_slice=0, data_source="hd")
        .opts(ylim=(0, 160))
    )
    myplot = myplot * hv.HLine(150).opts(alpha=0.7, color="black")
    myplot = apply_plot_default_settings(myplot)
    myplot = myplot.opts(title="Power vs. Time (mnist-cnn) - V100").opts(
        aspect=16 / 9, show_grid=True
    )
    hv.save(
        myplot,
        backend="matplotlib",
        filename="../master-thesis/images/power-mnist-cnn-v100.pdf",
        fmt="pdf",
    )
    # hv.render(myplot, backend="matplotlib")


def proof4theorem2(dgx_e):
    # "proof" for theorem 1

    e = "power-limit"

    hv.extension("matplotlib")
    power_plot = (
        dgx_e[e]
        .benchmarks["bert"]
        .plot("power", data_slice=0, data_source="sd")
        .opts(aspect=16 / 9, show_grid=True)
        .opts(title="")
    )
    # hv.save(power_plot, backend="matplotlib", filename="../master-thesis/images/power-bert-v100.pdf", fmt="pdf")

    clock_plot = (
        dgx_e[e]
        .benchmarks["bert"]
        .plot("clock-gpu", data_slice=0, data_source="sd")
        .opts(aspect=16 / 9, show_grid=True)
        .opts(title="")
    )
    # hv.save(clock_plot, backend="matplotlib", filename="../master-thesis/images/clock-gpu-bert-v100.pdf", fmt="pdf")

    hv.save(
        ((power_plot + clock_plot).cols(1).opts(title="BERT Finetuning - V100")),
        backend="matplotlib",
        filename="../master-thesis/images/power-clock-gpu-bert-v100.pdf",
        fmt="pdf",
    )
    # hv.render(myplot, backend="matplotlib")


def mink80(k80_e):
    e = "power-limit"
    for b, bench in k80_e[e].benchmarks.items():
        myplot = bench.plot("power", data_slice=0, data_source="hd").opts(
            aspect=16 / 9, show_grid=True
        )
        hv.save(
            myplot,
            backend="matplotlib",
            filename=f"../master-thesis/images/power-{b}-k80.pdf",
            fmt="pdf",
        )
        hv.render(myplot, backend="matplotlib")


def mink80freq(k80_e):
    e = "power-limit"
    for b, bench in k80_e[e].benchmarks.items():
        myplot = bench.plot("clock-gpu", data_slice=0, data_source="sd").opts(
            aspect=16 / 9, show_grid=True
        )
        hv.save(
            myplot,
            backend="matplotlib",
            filename=f"../master-thesis/images/clock-gpu-{b}-k80.pdf",
            fmt="pdf",
        )
        hv.render(myplot, backend="matplotlib")


def mink80freq_mem(k80_e):
    e = "power-limit"
    for b, bench in k80_e[e].benchmarks.items():
        myplot = bench.plot("clock-mem", data_slice=0, data_source="sd").opts(
            aspect=16 / 9, show_grid=True
        )
        hv.save(
            myplot,
            backend="matplotlib",
            filename=f"../master-thesis/images/clock-mem-{b}-k80.pdf",
            fmt="pdf",
        )
        hv.render(myplot, backend="matplotlib")


def min_v100_mnist_cnn_box(v100_e):
    e = "power-limit"

    agg = v100_e[e].benchmarks["mnist-cnn"].aggregate("hd", None)
    myplot = hv.BoxWhisker(agg, kdims="run", vdims="energy").opts(
        aspect=16 / 9,
        show_grid=True,
        title="Energy with different power limits (mnist-cnn) - V100",
    )
    myplot = apply_plot_default_settings(myplot)
    hv.save(
        myplot,
        backend="matplotlib",
        filename=f"../master-thesis/images/energy-min-box-v100.pdf",
        fmt="pdf",
    )
    hv.render(myplot, backend="matplotlib")


def energy_freq_model(loader: "DataLoader"):

    all_pots = []
    for benchmark_name, benchmark in loader.experiments[
        "power-limit"
    ].benchmarks.items():
        # if benchmark_name == "nbody":
        #      continue
        aggregates = benchmark.aggregate("sd").sort_values("clock-gpu")
        aggregates = aggregates[aggregates["clock-gpu"] >= 600]

        aggregates["energy"] = aggregates["energy"] / aggregates["energy"].max()
        # aggregates["energy"] = aggregates["energy"] / (aggregates["timestamp"] *70)
        clock_power_model = LinearFit(aggregates["clock-gpu"], aggregates["energy"], 2)
        print(clock_power_model, clock_power_model)
        plot = apply_plot_default_settings(
            hv.Scatter(
                aggregates.set_index("clock-gpu")["energy"],
                label=benchmark_description.get(benchmark_name, benchmark_name),
            ).opts(
                title=f"Clock vs. Energy ({benchmark_description.get(benchmark_name, benchmark_name)}) - {loader.clean_name}"
            )
            * clock_power_model.plot()
        )
        all_pots.append(plot)

    all_pots = apply_plot_default_settings(overlay_plots(all_pots))
    hv.save(
        all_pots,
        backend="matplotlib",
        filename=f"../master-thesis/images/energy_freq_model-{loader.clean_name}.pdf",
        fmt="pdf",
    )
    # hv.save(all_pots, backend="matplotlib", filename=f"../master-thesis/images/energy_freq_model-{benchmark_name}-{loader.clean_name}.pdf", fmt="pdf")


def power_freq_model(loader: "DataLoader"):

    all_pots = []
    for benchmark_name, benchmark in loader.experiments[
        "power-limit"
    ].benchmarks.items():
        # if benchmark_name == "nbody":
        #      continue
        aggregates = benchmark.aggregate("sd").sort_values("clock-gpu")
        aggregates = aggregates[aggregates["clock-gpu"] >= 600]

        # aggregates["energy"] = aggregates["energy"] / aggregates["energy"].max()
        # aggregates["energy"] = aggregates["energy"] / (aggregates["timestamp"] *70)
        clock_power_model = LinearFit(
            aggregates["clock-gpu"], aggregates["enforced-power-limit"], 1
        )
        print(clock_power_model, clock_power_model)
        plot = apply_plot_default_settings(
            hv.Scatter(
                aggregates.set_index("clock-gpu")["enforced-power-limit"],
                label=benchmark_description.get(benchmark_name, benchmark_name),
            ).opts(
                title=f"Clock vs. Power Limit ({benchmark_description.get(benchmark_name, benchmark_name)}) - {loader.clean_name}"
            )
            * clock_power_model.plot()
        )
        all_pots.append(plot)

    all_pots = apply_plot_default_settings(overlay_plots(all_pots))
    hv.save(
        all_pots,
        backend="matplotlib",
        filename=f"../master-thesis/images/power_freq_model-{loader.clean_name}.pdf",
        fmt="pdf",
    )
    # hv.save(all_pots, backend="matplotlib", filename=f"../master-thesis/images/energy_freq_model-{benchmark_name}-{loader.clean_name}.pdf", fmt="pdf")


# def test_corr(dgx_e):
#     e = "power-limit"
#
#     hv.extension('matplotlib')
#     for run in dgx_e[e].benchmarks["bert"].runs.values():
#         df = run.repetitions[4].measurements["sd"].data
#         print(df[["power", "clock-gpu"]].corr())
#
#         # myplot = .plot("power", data_slice=0, data_source="hd").opts(ylim=(0, 160))


if __name__ == "__main__":
    hv.extension("matplotlib")
    #
    dgx = DataLoader("data/dgx11")
    dgx_e = dgx.experiments

    min_v100_mnist_cnn_box(dgx_e)
    # #
    # # proof4theorem2(dgx_e)
    #
    # # #
    # # # # proof4theorem1(dgx_e)
    # # #
    # k80 = DataLoader("data/k80-6/")
    # k80_e = k80.experiments
    #
    # # t4 = DataLoader("data/t4-3")
    #
    # mink80(k80_e)
    # mink80freq(k80_e)
    # mink80freq_mem(k80_e)
    # #
    # # energy_freq_model(k80)
    # # energy_freq_model(dgx)
    # # energy_freq_model(t4)
    #
    # # power_freq_model(k80)
    # # power_freq_model(dgx)
    # # power_freq_model(t4)
