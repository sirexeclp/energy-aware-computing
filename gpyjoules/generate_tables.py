import copy

import hvplot
import hvplot.pandas
import holoviews as hv
from bokeh.models import HoverTool
from gpyjoules.util import TimestampInfo
from gpyjoules.new_analysis import *
import bokeh.themes
from pathlib import Path
import pandas as pd
import json
import numpy as np
from holoviews import opts
import seaborn as sns
from pylatex import Document
from pylatex.utils import italic, NoEscape


def df_argmin(df, col):
    return df[df[col] == df[col].min()]


def aggregate_all(
    loaders: List[DataLoader], agg_rep: Optional[str] = "mean", data_source="hd"
):
    data = []
    for loader in loaders:
        for experiment in loader.experiments.values():
            for benchmark in experiment.benchmarks.values():
                tmp = benchmark.aggregate(
                    data_source=data_source, agg_time="mean", agg_rep=agg_rep
                )
                tmp["platform"] = loader.name
                tmp["exp"] = experiment.name
                tmp["benchmark"] = benchmark.name
                data.append(tmp)
    df = pd.concat(data, sort=True)
    df = df.reset_index()
    return df


def make_table_great_again(table, target, signs, units):
    signs = copy.deepcopy(signs)
    units = copy.deepcopy(units)
    if target:
        signs[target] = signs[target] + "_{min}"
        signs["run"] = signs["run"].format(signs[target])

    def make_header(sign, unit):
        result = f"${sign}$"
        if unit is not None:
            result += f" $\\left[\\SI{{}}{{{unit}}}\\right]$"
        return result

    columns = {s: make_header(signs[s], units[s]) for s in signs}

    if "run" in table.columns:
        table["run"] = r"\SI{" + table["run"] + "}{W}"

    if table.index.name == "run":
        table.index = r"\SI{" + table.index + "}{W}"

    table.index.name = signs.get(table.index.name, table.index.name)

    table = table.rename(columns=columns)

    table = table.sort_index()

    return table


def generate_column_format(table: pd.DataFrame, highlight_index: int):
    hightlight = r">{\columncolor{lightblue}}"
    result = ["l"]
    for index, _ in enumerate(table.columns):
        if index == highlight_index:
            result.append(hightlight)
        result.append("r")
    return "".join(result)


def get_long_platform_name(name):
    long_platform = {"dgx": "V100", "k80": "K80", "t4": "T4"}
    for key, value in long_platform.items():
        if key in name:
            return value


def get_doc_with_preamble():
    doc = Document("basic")
    doc.packages.append(NoEscape(r"\usepackage{siunitx}"))
    doc.packages.append(NoEscape(r"\usepackage{colortbl}"))
    doc.packages.append(NoEscape(r"\usepackage[dvipsnames,usenames]{xcolor}"))
    doc.packages.append(NoEscape(r"\usepackage{booktabs}"))
    doc.append(NoEscape(r"\definecolor{lightblue}{rgb}{0.68, 0.85, 0.9}"))
    return doc


def table_here(text):
    first_line, rest = text.split("\n", 1)
    return f"{first_line}[h]\n{rest}"


def highlight_baseline(text: str, baseline: str):
    highlight = r"\rowcolor{lightblue}"
    search_str = f"\\SI{{{baseline}}}{{W}}"
    result = []
    for line in text.split("\n"):
        if line.startswith(search_str):
            line = highlight + line
        result.append(line)
    return "\n".join(result)


def generate_min_table(data, optimization_targets):
    # generate min tables

    long_targets = {"energy": "energy", "timestamp": "time", "edp": "EDP"}

    signs = {
        "timestamp": "t",
        "run": "\\left \\lceil{{P}}\\right \\rceil _{{{}}}",
        "edp": "EDP",
        "ed2p": "ED^2P",
        "energy": "E",
    }

    units = {
        "timestamp": "s",
        "run": None,
        "edp": "kJs",
        "ed2p": "kJs^2",
        "energy": "kJ",
    }

    doc = get_doc_with_preamble()

    result = []
    for t_index, target in enumerate(optimization_targets):
        projected_columns = ["run"] + optimization_targets

        for platform, platform_data in data.groupby("platform"):
            result.append("")
            for experiment, experiment_data in platform_data.groupby("exp"):
                result.append("")

                min_runs = (
                    experiment_data.groupby(["benchmark", "run"])
                    .mean()
                    .reset_index(level="benchmark")
                    .groupby("benchmark")
                    .apply(lambda x: df_argmin(x, target))
                )

                means = min_runs.reset_index(level="run")

                stds = (
                    experiment_data.groupby(["benchmark", "run"])
                    .std()
                    .loc[min_runs.index]
                    .reset_index(level="run")
                )

                for column in optimization_targets:
                    means[column] = (
                        "$"
                        + means[column].map(lambda x: "{:.3f}".format(x))
                        + " \\pm "
                        + stds[column].map(lambda x: "{:.3f}".format(x))
                        + "$"
                    )

                means = means[projected_columns]

                formatted_table = make_table_great_again(means, target, signs, units)

                table_name = f"min-{long_targets[target]}-{get_long_platform_name(platform)}-{experiment}"
                latex_table_str = formatted_table.to_latex(
                    float_format="%.3f",
                    escape=False,
                    column_format=generate_column_format(formatted_table, t_index + 1),
                    label=f"tab:{table_name}",
                    caption=f"Optimal (minimizing) power limit with respect to {long_targets[target]}. Measured on {get_long_platform_name(platform)}.",
                )
                doc.append(NoEscape(latex_table_str))
                result.append(latex_table_str)
                table_path = (Path("../master-thesis/tables") / table_name).with_suffix(
                    ".tex"
                )
                table_path.write_text(table_here(latex_table_str))

    doc.generate_pdf("min_tables", clean_tex=False)
    return "\n".join(result)


def generate_dif_table(data, projected_columns):

    baselines = {"V100": "300", "K80": "150", "T4": "70"}

    signs = {
        "timestamp": "\\Delta t",
        "run": "\\left \\lceil{{P}}\\right \\rceil _{{{}}}",
        "edp": "\\Delta EDP",
        "ed2p": "\\Delta ED^2P",
        "energy": "\\Delta E",
    }

    units = {
        "timestamp": "\\%",
        "run": None,
        "edp": "\\%",
        "ed2p": "\\%",
        "energy": "\\%",
    }

    result = []

    doc = get_doc_with_preamble()

    for platform, platform_data in all_totals.groupby("platform"):
        result.append("")
        for experiment, experiment_data in platform_data.groupby("exp"):
            result.append("")
            for benchmark, benchmark_data in experiment_data.groupby("benchmark"):
                result.append("")

                benchmark_data = benchmark_data.set_index("run")

                means = benchmark_data.groupby("run").mean()

                baseline_pl = baselines[get_long_platform_name(platform)]

                baseline = means[means.index == baseline_pl]

                # divide by baseline and convert to percental change
                normalized_data = (
                    benchmark_data.divide(baseline.iloc[0], axis=1) - 1
                ) * 100

                means_norm = normalized_data.groupby("run").mean()
                stds_norm = normalized_data.groupby("run").std()

                for column in optimization_targets:
                    means_norm[column] = (
                        "$"
                        + means_norm[column].map(lambda x: "{:.3f}".format(x))
                        + " \\pm "
                        + stds_norm[column].map(lambda x: "{:.3f}".format(x))
                        + "$"
                    )

                table = means_norm[projected_columns]

                formatted_table = make_table_great_again(table, None, signs, units)

                table_name = (
                    f"diff-{benchmark}-{get_long_platform_name(platform)}-{experiment}"
                )
                latex_table_str = formatted_table.to_latex(
                    float_format="%.3f",
                    escape=False,
                    column_format="l" + "r" * len(formatted_table.columns),
                    label=f"tab:{table_name}",
                    caption=f"Relative differences (in \\%) of different power limits for the {benchmark} benchmark compared to the default of \\SI{{{baseline_pl}}}{{W}}.  Measured on {get_long_platform_name(platform)}.",
                )
                latex_table_str = highlight_baseline(latex_table_str, baseline_pl)
                doc.append(NoEscape(latex_table_str))
                print(latex_table_str)
                table_path = (Path("../master-thesis/tables") / table_name).with_suffix(
                    ".tex"
                )
                table_path.write_text(table_here(latex_table_str))

    doc.generate_pdf("dif_tables", clean_tex=False)
    return "\n".join(result)


def generate_corr_table(data):
    # generate min tables

    long_targets = {"energy": "energy", "timestamp": "time", "edp": "EDP"}

    doc = get_doc_with_preamble()

    result = []

    tables = []
    for platform, platform_data in data.groupby("platform"):
        result.append("")
        for experiment, experiment_data in platform_data.groupby("exp"):

            correlations = (
                experiment_data.groupby(["benchmark", "run"])
                .mean()
                .reset_index(level="benchmark")
                .groupby("benchmark")[["power", "clock-gpu"]]
                .corr()["power"]
                .iloc[1::2]
                .reset_index(level=1)
                .drop(columns="level_1")
                .rename(columns={"power": "r"})
            )

            correlations["r"] = (
                "$" + correlations["r"].map(lambda x: "{:.3f}".format(x)) + "$"
            )

            correlations = correlations.rename(
                columns={
                    "r": f"$corr(\\bar P, \\bar f)_{{ \\mathrm{{ {get_long_platform_name(platform)} }} }}$"
                }
            )
            tables.append(correlations)

    correlations = pd.concat(tables, axis=1)

    table_name = f"corr-power-clock-gpu"
    latex_table_str = correlations.to_latex(
        float_format="%.3f",
        escape=False,
        column_format=generate_column_format(correlations, None),
        label=f"tab:{table_name}",
        caption=f"Correlation coefficients of power and gpu clock (averaged over time and runs).",
    )

    doc.append(NoEscape(latex_table_str))
    result.append(latex_table_str)
    table_path = (Path("../master-thesis/tables") / table_name).with_suffix(".tex")
    table_path.write_text(table_here(latex_table_str))

    doc.generate_pdf("corr_tables", clean_tex=False)
    return "\n".join(result)


if __name__ == "__main__":
    hv.extension("bokeh", "matplotlib")

    data_root = Path("data")
    data_sets = ["k80-6", "dgx11", "t4-3"]

    data_loaders = [DataLoader(data_root / x) for x in data_sets]

    all_agg = aggregate_all(data_loaders, None)
    # select only power limit experiments
    all_totals = all_agg[all_agg.exp == "power-limit"]
    # remove W sign from run column
    all_totals["run"] = all_totals["run"].str[:-1]

    optimization_targets = ["energy", "timestamp", "edp"]
    generate_min_table(all_totals, optimization_targets)
    generate_dif_table(all_totals, optimization_targets)

    # all_agg = aggregate_all(data_loaders, None, "sd")
    # # select only power limit experiments
    # all_totals = all_agg[all_agg.exp == "power-limit"]
    # # remove W sign from run column
    # all_totals["run"] = all_totals["run"].str[:-1]
    #
    # res = generate_corr_table(all_totals)
    # print(res)
