# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: bluebird-energy-aware
#     language: python
#     name: bluebird-energy-aware
# ---

# %%
# change to correct working directory
# # %pwd
# %cd ..
# %pwd

# %%
# load the auto-reload extension
# %load_ext autoreload
# %autoreload 2

# %%
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
import holoviews as hv
from holoviews import opts
import seaborn as sns

# %%
hv.extension("bokeh", "matplotlib")

# %%
data_root = Path("data")
data_sets = ["k80-2", "dgx9", "t4-1"]

loaders = [DataLoader(data_root/x) for x in data_sets]


# %% [markdown]
# plots = [l.experiments["power-limit"].benchmarks["bert"].plot("energy", label_pre=str(l.name)+".") for l in loaders]
# overlay_plots(plots)

# %% [markdown]
# df = [l.get_total_values("bert", "mean") for l in loaders][0]#.min()
# df[df.energy ==df.energy.min()]

# %%
def df_argmin(df, col):
    return df[df[col] == df[col].min() ]


# %%
def aggregate_all(loaders, func="mean"):
    data = []
    for l in loaders:
        for e in l.experiments.values():
            for b in e.benchmarks.values():
                #aggregates = b.aggregate(func2=func)
                aggregates = b.get_total_values(aggregate=func)
                #maximum = aggregates.loc[aggregates.index.max()]
                tmp = aggregates#maximum - df_argmin(aggregates, column)
                tmp["platform"] = l.name
                tmp["exp"] = e.name
                tmp["benchmark"] = b.name
                data.append(tmp)
    df = pd.concat(data, sort=True)
    #df = df[["platform", "exp", "benchmark", "timestamp", "energy", "edp"]]
    df = df.reset_index()
    return df


# %%
group_keys = ["platform", "exp", "benchmark"]

# %%
all_aggregates_mean = aggregate_all(loaders)
all_aggregates_mean = all_aggregates_mean[all_aggregates_mean.exp == "power-limit"]
# remove W sign from run column
all_aggregates_mean["run"] = all_aggregates_mean["run"].str[:-1]

# %%
all_aggregates_std = aggregate_all(loaders, "std")
all_aggregates_std = all_aggregates_std[all_aggregates_std.exp == "power-limit"]

# remove W sign from run column
all_aggregates_std["run"] = all_aggregates_std["run"].str[:-1]

# %%
join_index = group_keys + ["run"]
joined = all_aggregates_mean.set_index(join_index).join(all_aggregates_std.set_index(join_index), rsuffix="_std")

joined = joined.reset_index("run")

# all_aggregates_mean = all_aggregates_mean.set_index(group_keys)
# all_aggregates_std = all_aggregates_std.set_index(group_keys)

# %%
all_totals = aggregate_all(loaders, None)
all_totals = all_totals[all_totals.exp == "power-limit"]
# remove W sign from run column
all_totals["run"] = all_totals["run"].str[:-1]

# %%
group_keys

# %%
all(joined.reset_index().set_index(join_index)[optimization_targets].sort_index() == all_totals.groupby(join_index).apply(np.mean)[optimization_targets].sort_index() )

# %%
all_totals#.groupby(join_index).apply(np.std)[optimization_targets]

# %%
len(all_aggregates_mean), len(all_aggregates_std), len(joined)


# %%
def make_table_great_again(table, target):
    
    signs = {
        "timestamp": "t",
        "run": "\left \lceil{{P}}\right \rceil _{{{}}}",
        "edp": "EDP",
        "ed2p": "ED^2P",
        "energy": "E",
    }
    
    units = {
        "timestamp": "s",
        "run": None,
        "edp": "kJs",
        "ed2p":"kJs^2",
        "energy": "kJ",
    }
    
    signs[target] = signs[target]+"_{min}"
    signs["run"] = signs["run"].format(signs[target])
    
    def make_header(sign, unit):
        result = f"${sign}$"
        if unit is not None:
            result += f" $\left[\SI{{}}{{{unit}}}\right]$"
        return  result
    
    
    columns = {s:make_header(signs[s], units[s]) for s in signs}
    
    if "run" in table.columns:
        table["run"] = "\SI{" + table["run"] + "}{W}"
    
    table = table.rename(columns=columns)
    
    table = table.sort_index()
#         columns={
#     "timestamp": "$t_{min}$ $[s]$",
#     "run": "$P_{t_{min}}$",
#     "edp": "$EDP_{min}$ $\left[kJs\right]$",
#     "energy": "$E_{min}$ $\left[kJ\right]$",
#     })
    
    return table

# make_table_great_again(table, target)

# %%

# %%
for experiment, experiment_data in experiment_data.groupby("benchmark"):
    pass

d =  experiment_data[projected_columns]
d = d.reset_index().drop(columns=group_keys).set_index("run")
baseline = d[d.index == d.index.max()]
#baseline.reset_index().drop(columns=group_keys).set_index("run")
(d.divide(baseline.iloc[0], axis=1)-1)*100

# %%
t = (experiment_data.groupby(level=minima.index.names).apply(lambda x: df_argmin(x, target))
                     .reset_index(level=[3,4,5]).drop(columns=group_keys)
                     .reset_index(level=["platform", "exp"]))

t[["timestamp", "timestamp_std"]].apply(lambda x: x)# + "+-" + t["timestamp_std"].to_string()
t["timestamp_std"].map(lambda x: "{:.3f}".format(x))+ " +- " +  t["timestamp"].map(lambda x: "{:.3f}".format(x))

# %%
optimization_targets = ["energy", "timestamp", "edp"]
from IPython.display import display, Math, Latex

long_targets = {
    "energy": "energy"
    , "timestamp": "time"
    , "edp": "EDP"
}

long_platform = {
    "dgx9": "V100",
    "k80-2": "K80",
    "t4-1": "T4"
}

for target in optimization_targets:
    projected_columns = ["run"] + optimization_targets

    for platform, platform_data in joined.groupby(level="platform"):
        print("")
        for experiment, experiment_data in platform_data.groupby(level="exp"):
            print("")
            table = (experiment_data.groupby(level=experiment_data.index.names).apply(lambda x: df_argmin(x, target))
                     .reset_index(level=[3,4,5]).drop(columns=group_keys)
                     .reset_index(level=["platform", "exp"]))
            
            for column in optimization_targets:
                table[column] = "$" + table[column].map(lambda x: "{:.3f}".format(x))+ " \pm " +  table[column+"_std"].map(lambda x: "{:.3f}".format(x)) + "$"
            
            table = table[projected_columns]
            
            formatted_table = make_table_great_again(table, target)
            
            print(formatted_table.to_latex(float_format="%.3f",
                                           escape=False,
                                           column_format="l"+"r"*len(formatted_table.columns),
                                           label=f"tab:min-{long_targets[target]}-{long_platform[platform]}-{experiment}",
                                           caption=f"Optimal (minimizing) power limit with respect to {long_targets[target]}. Measured on {long_platform[platform]}."
                                          ))
            
           # display(formatted_table#.to_latex(float_format="%.3f", escape=False
                                                                 #, caption="This table lists the optimal (minimizing) power limit settings for time, energy and edp respectively as measured on V100 GPUs. Values are obtained by taking the arithmetic mean over all repetitions."
                                                                 #, label="tab:min-dgx-pl"
                                                               # )
            #)
            

# %%
#
baseline[std_projections] = baseline[projected_columns]
baseline[std_projections]

# %%
benchmark_data.groupby("run").mean()
((1/all_totals[(all_totals.platform=="dgx9")&(all_totals.benchmark=="nbody")].groupby(join_index).mean().reset_index().set_index("run").timestamp) **3 ).plot()

# %%
optimization_targets = ["energy", "timestamp", "edp", "ed2p"]
from IPython.display import display, Math, Latex

long_targets = {
    "energy": "energy"
    , "timestamp": "time"
    , "edp": "EDP"
    ,"ed2p": "ED^2P"
}

long_platform = {
    "dgx9": "V100",
    "k80-2": "K80",
    "t4-1": "T4"
}

projected_columns = optimization_targets

for platform, platform_data in all_totals.groupby("platform"):
    print("")
    for experiment, experiment_data in platform_data.groupby("exp"):
        print("")
        for benchmark, benchmark_data in experiment_data.groupby("benchmark"):
            print("")

            #table = (benchmark_data.reset_index()
            #                 .drop(columns=group_keys)
            #                 .set_index("run")
            #         )
            
            benchmark_data = benchmark_data.set_index("run")
            
            means = benchmark_data.groupby("run").mean()
            

            baseline = means[means.index == means.index.max()]
        

            # divide by baseline and convert to percental change
            normalized_data = (benchmark_data.divide(baseline.iloc[0], axis=1)-1)*100
            
            means_norm = normalized_data.groupby("run").mean()
            stds_norm = normalized_data.groupby("run").std()


            for column in optimization_targets:
                means_norm[column] = "$" + means_norm[column].map(lambda x: "{:.3f}".format(x))+ " \pm " +  stds_norm[column].map(lambda x: "{:.3f}".format(x)) + "$"

            table = means_norm[projected_columns]



            formatted_table = make_table_great_again(table, target)

            print(formatted_table.to_latex(float_format="%.3f",
                                           escape=False,
                                           column_format="l"+"r"*len(formatted_table.columns),
                                           label=f"tab:diff-{benchmark}-{long_platform[platform]}-{experiment}",
                                           caption=f"Relative differences (in \%) of different power limits for the {benchmark} benchmark.  Measured on {long_platform[platform]}."
                                          ))
            
           # display(formatted_table#.to_latex(float_format="%.3f", escape=False
                                                                 #, caption="This table lists the optimal (minimizing) power limit settings for time, energy and edp respectively as measured on V100 GPUs. Values are obtained by taking the arithmetic mean over all repetitions."
                                                                 #, label="tab:min-dgx-pl"
                                                               # )
            #)


# %%
optimization_targets = ["energy", "timestamp", "edp"]
from IPython.display import display, Math, Latex

long_targets = {
    "energy": "energy"
    , "timestamp": "time"
    , "edp": "EDP"
}

long_platform = {
    "dgx9": "V100",
    "k80-2": "K80"
}

for target in optimization_targets:
    projected_columns = ["run", target] + [x for x in optimization_targets if x != target]

    for platform, platform_data in joined.groupby(level="platform"):
        print("")
        for experiment, experiment_data in platform_data.groupby(level="exp"):
            print("")
            table = (experiment_data.groupby(level=minima.index.names).apply(lambda x: df_argmin(x, target))
                     .reset_index(level=[3,4,5]).drop(columns=group_keys)
                     .reset_index(level=["platform", "exp"]))
            
            for column in optimization_targets:
                table[column] = "$" + table[column].map(lambda x: "{:.3f}".format(x))+ " $\pm$ " +  table[column+"_std"].map(lambda x: "{:.3f}".format(x)) +"$"
            
            table = table[projected_columns]
            
            formatted_table = make_table_great_again(table, target)
            
            print(formatted_table.to_latex(float_format="%.3f",
                                           escape=False,
                                           column_format="l"+"r"*len(formatted_table.columns),
                                           label=f"tab:min-{long_targets[target]}-{long_platform[platform]}-{experiment}",
                                           caption=f"Optimal (minimizing) power limit with respect to {long_targets[target]}. Measured on {long_platform[platform]}."
                                          ))
            
           # display(formatted_table#.to_latex(float_format="%.3f", escape=False
                                                                 #, caption="This table lists the optimal (minimizing) power limit settings for time, energy and edp respectively as measured on V100 GPUs. Values are obtained by taking the arithmetic mean over all repetitions."
                                                                 #, label="tab:min-dgx-pl"
                                                               # )
            #)


# %%
class Metric:
    def __init__(self, name, internal_name, symbol, unit):
        self.name = name
        self.internal_name = internal_name
        self.symbol = symbol
        self.unit = unit
    
    def __str__(self):
        return f"Metric({self.name} [{self.unit}])"
    
    def __repr__(self):
        return self.__str__()

Metric("Time", "timestamp", "t", "s")
Metric("Run", "timestamp", "t", "s")

# %%

#minima = all_aggregates.groupby(group_keys)#.apply(lambda x: df_argmin(x, "energy"))
#minima = minima[minima.exp == "power-limit"]
#minima.loc["power-limit"]
#minima = minima.drop(columns=group_keys)
minima = all_aggregates.set_index(group_keys)
minima.groupby(level=minima.index.names).apply(lambda x: df_argmin(x, "energy")).reset_index(level=[3,4,5])

# %%
experiment_data.reset_index()[["energy", "run", "timestamp", "edp"]]

# %%
all_aggregates[(all_aggregates.platform == "dgx9") & (all_aggregates.exp == "power-limit") & (all_aggregates.benchmark == "bert")]


# %%
def get_minima(loaders, column):
    data = []
    for l in loaders:
        for e in l.experiments.values():
            for b in e.benchmarks.values():
                tmp = df_argmin(b.aggregate(func="std"), column)
                tmp["platform"] = l.name
                tmp["exp"] = e.name
                tmp["benchmark"] = b.name
                data.append(tmp)
    df = pd.concat(data, sort=True)
    df = df[["platform", "exp", "benchmark", column]]
    df = df.reset_index()
    return df


# %%
min_t

# %%
min_t = get_minima(loaders=loaders, column="timestamp")
min_e = get_minima(loaders=loaders, column="energy")
min_edp = get_minima(loaders=loaders, column="edp")

# %%
save_t = get_savings(loaders=loaders, column="timestamp")
#save_e = get_savings(loaders=loaders, column="energy")
#save_edp = get_savings(loaders=loaders, column="edp")

# %%
save_t


# %%
def select_platform_and_experiment(df):
    return df[(df.platform=="dgx9") & (df.exp=="clock-limit")].set_index("benchmark")


# %%
a = select_platform_and_experiment(min_t)[["timestamp", "run"]].rename(columns={
    "timestamp": "$t_{min}$ $[s]$",
    "run": "$P_{t_{min}}$"
})

# %%
b = select_platform_and_experiment(min_e)[["energy", "run"]].rename(columns={
    "energy": "$E_{min}$ $\left[kJ\right]$",
    "run": "$P_{E_{min}}$"
})

# %%
c = select_platform_and_experiment(min_edp)[["edp", "run"]].rename(columns={
    "edp": "$EDP_{min}$ $\left[kJs\right]$",
    "run": "$P_{EDP_{min}}$"
})


# %%
print(a.join(b).join(c).to_latex(float_format="%.3f", escape=False, caption="This table lists the optimal (minimizing) power limit settings for time, energy and edp respectively as measured on V100 GPUs. Values are obtained by taking the arithmetic mean over all repetitions.", label="tab:min-dgx-pl"))

# %%
for x in a.itertuples():
    print("".join(list(x)))

# %%
[f"{i:.3f}" for i in x if isinstance(i, float)]
    
def format_generator(row):
    for item in row:
        if isinstance(item, str):
            if "MHz" in item:
                yield "$"+item.replace("MHz", "", 1).replace(",","|")+ "$"
            else:
                yield item
        else:
            yield f"${item:.3f}$"

list(format_generator(x))


# %%
def build_table(df, label="", caption=""):
    columns = "r"*(len(df.columns)+1)
    table=f"\\begin{{table}}\n\\centering\n\\begin{{tabular}}{{{columns}}}\n"
    table += " & ".join([df.index.name]+[*df.columns])
    table +="\\\\\n\\hline\n"
    for row in df.itertuples():
        table += " & ".join(format_generator(row))
        table += "\\\\\n"
    
    table+="\\end{tabular}\n"
    table+=f"\\caption{{{caption}}}\n"
    table+=f"\\label{{{label}}}\n"
    
    table+="\\end{table}"
    return table


print(build_table(a))

# %%
print(a.join(b).join(c).to_latex(float_format="%.3f",
                                 escape=False, caption="This table lists the optimal (minimizing) clock limit "
                                 "settings for time, energy and edp respectively as measured on V100 GPUs."
                                 " Values are obtained by taking the arithmetic mean over all repetitions.",
                                 label="tab:min-dgx-cl"))


# %%
def select_platform_and_experiment(df):
    return df[(df.platform=="k80-2") & (df.exp=="clock-limit")].set_index("benchmark")


# %%
a = select_platform_and_experiment(min_t)[["timestamp", "run"]].rename(columns={
    "timestamp": "$t_{min}$ $[s]$",
    "run": "$P_{t_{min}}$"
})
b = select_platform_and_experiment(min_e)[["energy", "run"]].rename(columns={
    "energy": "$E_{min}$ $\left[kJ\right]$",
    "run": "$P_{E_{min}}$"
})
c = select_platform_and_experiment(min_edp)[["edp", "run"]].rename(columns={
    "edp": "$EDP_{min}$ $\left[kJs\right]$",
    "run": "$P_{EDP_{min}}$"
})

# %%
a

# %%
print(a.join(b).join(c).to_latex(float_format="%.3f", escape=False,
                                 caption="This table lists the optimal (minimizing) power limit settings for time, energy and edp respectively "
                                 "as measured on K80 GPUs. Values are obtained by taking the arithmetic mean over all repetitions."
                                 , label="tab:min-k80-pl"))

# %%
print(a.join(b).join(c).to_latex(float_format="%.3f", escape=False,
                                 caption="This table lists the optimal (minimizing) power clock settings for time, energy and edp respectively "
                                 "as measured on K80 GPUs. Values are obtained by taking the arithmetic mean over all repetitions."
                                 , label="tab:min-k80-cl"))

# %%

# %%
df[df.platform=="dgx9"].reset_index().set_index("benchmark").rename(columns={"run":"best time", "timestamp":"time"})

# %%
df = df.rename(columns={"timestamp":"time"})

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%


