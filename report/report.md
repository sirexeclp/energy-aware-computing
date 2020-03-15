---
title:  Report -- Energy Aware Computing
subtitle: "Measuring Energy Consumption of Deep Learning Workloads"
author:
- Felix Grzelka
reference-section-title: Bibliography
geometry: "left=2.5cm,right=3cm,top=2cm,bottom=2cm"
toc: True
toc-title: Table of Contents
no-toc-relocation: True
link-citations: True
bibliography: /home/felix/bib/energy.bib
numbersections: true
header-includes: |
  \usepackage{float}
  \let\origfigure\figure
  \let\endorigfigure\endfigure
  \renewenvironment{figure}[1][2] {
      \expandafter\origfigure\expandafter[H]
  } {
      \endorigfigure
  }
---
\newpage

# Motivation

Training Artificial Deep Neural Networks consumes big ammounts of energy [@Strubell2019].
The time needed to train big (or rather deep), state of the art networks, for problems such as image classification or natural language processing can range from several hours to weeks [@Strubell2019]. During development networks are being trained multiple times to find the best combination of hyper parameters.
To speed up training specialized hardware such as clusters of Graphics Processing Units (GPUs) or Tensor Processing Units (TPUs) are being used [@Jouppi2017].
Nvidia offers servers utilizing its Telsa V100-GPUs to train neural networks.[^dgx]
A smaller server (DGX-1)[^dgx-1] equipped with 8 GPUs and a bigger variant (DGX-2) with 16 GPUs.
Each Telsa V100-GPU has a maximum thermal design power (TDP) of 300W.
The DGX-1 therefore has 4 redundant 1600W power supplies, which can deliver a total power of 3200W. Each GPU has an idle power draw of around 40--50W (or 50--60W when idle but attached to a process), which equates to a total idle power draw (including the rest of the system) of roughly 650--750W. This work focuses on reducing the energy needed for training such networks on GPUs.

<!-- [insert examples of training times for some networks] -->

[^dgx]: [https://www.nvidia.com/en-us/data-center/dgx-systems/](https://www.nvidia.com/en-us/data-center/dgx-systems/) Retrieved: 2020-03-02
[^dgx-1]: [https://www.nvidia.com/en-us/data-center/dgx-1/](https://www.nvidia.com/en-us/data-center/dgx-1/) Retrieved: 2020-03-02

<!-- # Related Works

## @Garcia-Martin2019

ML community shows "lack of interest [for energy aware computing]" because of:

- lack of familiarity with energy estimation
- lack of power models in popular frameworks
  - Tensorflow
  - Caffe2
  - PyTorch

## @Strubell2019

Energy in Deep Learning in NLP
best scores by computationally expensive models

recommendations:
1. report time to retrain and sensitivity to hyperparameters
2. researchers need access to compute resources
3. prioritize model and hardware efficiency

comparing models would require a standardized hardware independet measurement of training time



## @Li2016 -->
\newpage
# Methods


## Time and Sampling {#sec:time}

Two steps of time based preprocessing are necessary.
Timestamps with wall-clock time have been recorded.
However for further analysis only the relative time since the first collected sample of each benchmark run should be used.

\begin{equation}
t_{rel}(i) = t_i - t_0
\end{equation}

Secondly to create averaged curves from multiple runs, the samples need to be interpolated.
See section \ref{sec:jitter} for an analysis of the jitter.

A cubic interpolation was used to resample the signal at even sample distances, with a sample rate of 4Hz.
This sample rate was chosen to be about 2 times the measured average sample rate of the raw data, to avoid information loss caused by interpolation.

Finally to compute an averaged curve, different lengths of signals need to be taken care of.
We can expect, that multiple runs of the same experiment will not always run for the exact same amount of time and therefore produce signals with varying lengths. To compute mean and standard deviation for multiple signals with different lengths masked arrays can be used.

First each signal is padded to the length of the longest signal.
A mask is created that masks out all padding, that has been added.
Finally a numpy masked array is created with the padded signal and the mask.
Numpy masked arrays can calculate the mean at each sample, but will only include values that have not been masked out.



## Calculating Energy

The tools used for this study did not provide a direct way of measuring energy. Instead only the current power draw (or an average over a fixed time period) could be measured.
The total energy needed for training can be calculated as the dot product of the time differences between samples and the sampled power readings.

\begin{equation}
E = \vec{\Delta t}^T \cdot \vec{P}
\label{eq:power}
\end{equation}


To visualize and compare the energy consumed up to a certain point in time of training ($E_s$), the cumulative sum can be used:

\begin{equation}
E_s = \sum^s_{i=0}=\Delta t_i \cdot P_i
\label{eq:cum-power}
\end{equation}

## Energy Delay Product

To give equal weight to energy and execution time the energy delay product (EDP) can be used. It is defined as the product of energy and execution time.

\begin{equation}
EDP = E \cdot \Delta t = P \cdot \Delta t^2
\end{equation}

\newpage
# Experimental Design

## Hard- and Middleware

All experiments were run on a DGX-1 on one of the eight Tesla V100-32GB cards.

Table: Component Versions \label{tab:versions}

| Component       | Version                   |
| --------------- | ------------------------- |
| Host OS         | Ubuntu 18.04.3 LTS        |
| Container OS    | Ubuntu 18.04.3 LTS        |
| (Nvidia) Docker | 19.03.4, build 9013bf583a |
| Nvidia Driver   | 418.116.00                |
| NVIDIA-SMI      | 418.116.00                |
| Cuda            | 10.1                      |
| Python          | 3.6.9                     |
| pip3            | 19.3.1                    |
| tensorflow-gpu  | 1.14.0+nv                 |
| nvidia-ml-py3   | 7.352.0                   |

Table \ref{tab:versions} lists the versions of components used to run the experiments.

## Measurements

Table: Measured Values \label{tab:measured-values}

| Value                          | Unit   |
| ------------------------------ | ------ |
| Timestamp                      | $\mu$S |
| GPU Utilization                | %      |
| Memory Utilization             | %      |
| Streaming Multiprocessor Clock | MHz    |
| Memory Clock                   | MHz    |
| Power State                    | ---    |
| Power Draw                     | mW     |
| Temperature                    | °C     |
| PCIe Tx Throughput             | MB/s   |
| PCIe Rx Throughput             | MB/s   |


Table \ref{tab:measured-values} lists measured values (for each GPU) and their respective units.
Additional the following timestamps have been recorded:

- experiment start
- training start
- epoch start
- epoch end
- training end
- experiment end

## GPyJoules

`GPyJoules` is a novel python based gpu-power-profiling tool, which has been developed
to conduct measurements more easily and without the need to further modify the source code of the benchmark scripts.
GPyJoules is used as a command line tool, to wrap the call of a python script (or module).
It monkey patches keras to automatically add callbacks to every model.
In particular the methods `tensorflow.keras.models.Model.fit` and `tensorflow.keras.models.Model.fit_generator` are being patched.
The added callbacks collect the timestamps above listed during training.

It also spawns a new sub-process, before the training is started.
The sub-process is used to collect measurements using the pip package `nvidia-ml-py3`[^nvidia-ml-py3] which provides the `nvidia_smi` module.
This is a wrapper around the c API of the NVIDIA Management Library (NVML).[^nvml-api]

Our experiments used nvml queries in a while True loop, to query all listed metrics for all 8 GPUs as fast as possible.


## Benchmarks

Table \label{tab:parameters} lists the 3 different networks, used as benchmarks.

> Note that the repositories listed in the row `Code` have been adapted to fit the needs of the benchmarks.

Table: Neural Network Parameters \label{tab:parameters}

--------------------------------------------------------------------------
Network          Parameters                   Dataset          Code
---------------- ---------------------------- ---------------- -----------
ECG Net\         Deep Resnet with relatively  CinC17[^CinC17]\ [^ecg-fork]
[@Hannun2019]    small input and layers       [@Clifford2017]  (forked from[^ecg-original])        

MNIST Small      Network with 2 hidden dense  MNIST[^mnist-db] adapted from[^mnist-code]
                 layers, each 512 neurons
                 wide and 128 batch-size

MNIST Big        Network with 2 hidden dense  MNIST[^mnist-db] adapted from[^mnist-code]
                 layers, each 16384 neurons
                 wide and 32 batch-size
--------------------------------------------------------------------------


The power limits used for training were: 150W, 200W, 250W and 300W.
The sequence of power limits was randomized for each repetition, to prevent effects of previous runs (e.g. higher temperature) from  biasing the successive results. This also distributes changes of system-load not caused by the benchmark evenly over the different constraints.
Each network was trained for 10 epochs after setting a power limit.

[^mnist-db]: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
[^CinC17]: [https://physionet.org/content/challenge-2017/1.0.0/](https://physionet.org/content/challenge-2017/1.0.0/)
[^mnist-code]: [https://keras.io/examples/mnist_cnn/](https://keras.io/examples/mnist_cnn/)
[^ecg-fork]: [https://github.com/sirexeclp/ecg-1](https://github.com/sirexeclp/ecg-1)
[^ecg-original]: [https://github.com/awni/ecg](https://github.com/awni/ecg)


## Jitter {#sec:jitter}

Methods used for data analysis did not require equally spaced samples or used interpolation to reduce the error introduced by jitter.

![Distribution of sample rate using the naive approach \label{fig:old-dist}](fig/old-dist.png)

Figure \ref{fig:old-dist} shows the distribution of the sample rate, with the currently used method.
We observe a mean sampling frequency of $\mu = 2.045Hz$ and a standard deviation of $\sigma = 0.317Hz$.

> Note: This uses only the data from one training run of a singe network. Sampling rates may or may not be different depending on load. This needs further investigation.

Prior tests on a single laptop GTX 2060 showed sampling frequencies of as high as 25Hz, but the actual sampling frequency was about 6Hz, when measuring the smallest peaks (every data point was duplicated multiple times). Nvidia's support website lists a sample rate between 1 to 6 Hz (depending on the GPU) for GPU and memory utilization.[^smi-support]

![Jitter of samples over time \label{fig:old-jitter}](fig/old-method-change-in-fs.png)

Figure \ref{fig:old-jitter} shows the change in sample rate (jitter) over time, for the currently used method.

\newpage
# Results

## Raw Data

> Note: This section only uses the data from the first run, all following sections average data from all runs.

![ECG Power Draw over Time\label{fig:ecg-power-raw}](fig3/ecg-power-raw.pdf)

In figure \ref{fig:ecg-power-raw} we can see the power draw of the ecg net, trained with the different power limits (low to high from top to bottom) over time.
We can clearly see the spikes, which appear in the 300W setting being cut off in the 200W setting and a clear shift of the maximum below 200W in the 150W setting. This is a good indication that setting the power limit worked and that the data can be used for further analysis.

![MNIST-BIG Power Draw over Time\label{fig:mnist-big-power-raw}](fig3/mnist-big-power-raw.pdf)

Figure \ref{fig:mnist-big-power-raw} shows similar results to figure \ref{fig:ecg-power-raw}. The average power draw for the highest power limit seems to be slightly above 200W and slightly below 200W for the 200W limit.
The 150W limit shows a lower average around 150W.
This is to be expected as a power limit set above the actual power draw, should not affect the power draw.

![MNIST-SMALL Power Draw over Time\label{fig:mnist-small-power-raw}](fig3/mnist-small-power-raw.pdf)

Figure \ref{fig:mnist-small-power-raw} shows a power draw of below 100W for all power limits.


## Cumulative Energy

Using equation \ref{eq:cum-power} the cumulative energy was calculated.
The plots in this section show mean and standard deviation of the
cumulative energy using the methods described in section \ref{sec:time}.

![ECG Cumulative Energy over Time\label{fig:ecg-cum-energy}](fig3/ecg-cum-energy.pdf)

Figure \ref{fig:ecg-cum-energy} shows a slower incline for the 150W limit and a slightly lower one for the 200W limit, resulting in overall less consumed energy for the 150W limit at the cost of increased time.

![MNIST-BIG Cumulative Energy over Time\label{fig:mnist-big-cum-energy}](fig3/mnist-big-cum-energy.pdf)

Figure \ref{fig:mnist-big-cum-energy} shows similar results to \ref{fig:ecg-cum-energy} for the "big" mnist experiment, but with bigger effect of the 200W power limit.

![MNIST-SMALL Cumulative Energy over Time\label{fig:mnist-small-cum-energy}](fig3/mnist-small-cum-energy.pdf)

Figure \ref{fig:mnist-small-cum-energy} shows high variance for all power limits.
The averages over all runs are similar for all power limits and well within the expected standard deviation.


## Mean Total Energy

Plots in this section show the mean energy vs. the power limit and a second degree least squares polynomial fit to interpolate values between the tested power limits.

![ECG Mean Total Energy vs. Power Limit\label{fig:ecg-mean-energy}](fig3/ecg-mean-total-energy.pdf)

Figure \ref{fig:ecg-mean-energy} shows the lowest energy consumption, as seen in previous plots, for the 150W power limit.
The parabola fits the data almost perfectly ($R^2$ rounded to 3 decimal places).
$100\%$ of the variance can be explained with this model.

![MNIST-BIG Mean Total Energy vs. Power Limit\label{fig:mnist-big-mean-energy}](fig3/mnist-big-mean-total-energy.pdf)

Figure \ref{fig:mnist-big-mean-energy} shows similar results to \ref{fig:ecg-mean-energy} for the "big" mnist experiment, but the model does not fit the data as good.

![MNIST-SMALL Mean Total Energy vs. Power Limit\label{fig:mnist-small-mean-energy}](fig3/mnist-small-mean-total-energy.pdf)

The proposed energy-model fails for the small mnist network, which can be seen in Figure \ref{fig:mnist-small-mean-energy}.


## Mean Total Time

Plots in this section show the mean execution time vs. the power limit and a second degree least squares polynomial fit to interpolate values between the tested power limits.

![ECG Mean Total Time vs. Power Limit\label{fig:ecg-mean-time}](fig3/ecg-mean-total-time.pdf)

Figure \ref{fig:ecg-mean-time} shows the fastest execution time, for the 300W power limit.
$100\%$ of the variance in the data can be explained with the model.

![MNIST-BIG Mean Total Time vs. Power Limit\label{fig:mnist-big-mean-time}](fig3/mnist-big-mean-total-time.pdf)

Figure \ref{fig:mnist-big-mean-time} shows, again, a worse fit than on the ecg data.

![MNIST-SMALL Mean Total Time vs. Power Limit\label{fig:mnist-small-mean-time}](fig3/mnist-small-mean-total-time.pdf)

The time-model fails on the small mnist network (see Fig. \ref{fig:mnist-small-mean-time}).

## Mean EDP

Plots in this section show the energy delay product (EDP) vs. the power limit and a second degree least squares polynomial fit to interpolate values between the tested power limits.


![ECG Mean EDP vs. Power Limit\label{fig:ecg-mean-edp}](fig3/ecg-mean-edp.pdf)

Figure \ref{fig:ecg-mean-edp} shows the lowest EDP, for the 150W power limit.
$100\%$ of the variance in the data can be explained with the model.

![MNIST-BIG Mean EDP vs. Power Limit\label{fig:mnist-big-mean-edp}](fig3/mnist-big-mean-edp.pdf)

Figure \ref{fig:mnist-big-mean-edp} shows the EDP on the big mnist network.

![MNIST-SMALL Mean EDP vs. Power Limit\label{fig:mnist-small-mean-edp}](fig3/mnist-small-mean-edp.pdf)

The EDP-model fails on the small mnist network (see Fig. \ref{fig:mnist-small-mean-edp}).


# Conclusion

This series of experiments showed that setting a power limit during training can reduce the consumed energy.
The energy savings on the lowest possible power limit (150W) outweigh the increase in execution time, making it overall the best setting, when energy is considered.
This behavior could be shown for two different network architectures, belonging to the family of cov/resnets.
Small networks with power draw always below the lowest power limit can not be optimized using this method.

For the tested networks a quadratic relations between power limit and energy, time and EDP have been shown.


\newpage
# Future Work

## Benchmarks

For future experiments a more diverse zoo of neural network architectures should be used, to represent many more different types of neural networks.

## Clock Limits

Instead of setting power limits, the clock frequency of the GPUs shared multiprocessors can be limited.
First tests showed great potential, as the lowest clock settings resulted in less power draw, than what was achievable using power limits.


## Improved Measurements

Higher sampling rates ($\mu=49.930Hz$) and higher variance ($\sigma=0.570$), but a lower Coefficient of Variance $CV_{new}=1.141 \%$ (compared to $CV_{naive}=15.494\%$) can be achieved using the `nvmlDeviceGetSamples`[^get-samples] function, which returns a buffer of values instead of a singe data point.

![Distribution of sample rate when using the `nvmlDeviceGetSamples` function](fig/get-sample-dist.png)


Table: Types of Sampling Events \label{tab:event-type}

------------------------------------------------------------------
Constant (NVML_)             Description                                                                   
---------------------------- -------------------------------------
TOTAL_POWER_SAMPLES          total power drawn by GPU                                                      

GPU_UTILIZATION_SAMPLES      percent of time during which one or more kernels was executing on the GPU     

MEMORY_UTILIZATION_SAMPLES   percent of time during which global (device) memory was being read or written 

ENC_UTILIZATION_SAMPLES      percent of time during which NVENC remains busy                               

DEC_UTILIZATION_SAMPLES      percent of time during which NVDEC remains busy                               

PROCESSOR_CLK_SAMPLES        processor clock samples                                                       

MEMORY_CLK_SAMPLES           memory clock samples                                                          
------------------------------------------------------------------

Table \ref{tab:event-type} was adapted from the nvml documentation. [^sampling-type]

First testing showed that a polling rate of 0.5 Hz is sufficient for not dropping samples.

![Jitter when using the `nvmlDeviceGetSamples` function](fig/new-method-change-in-fs.png)

[^nvidia-ml-py3]: [https://github.com/nicolargo/nvidia-ml-py3](https://github.com/nicolargo/nvidia-ml-py3) Retrieved: 2020-03-01
[^nvml-api]: [https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html](https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html) Retrieved: 2020-03-01
[^smi-support]: [https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries](https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries) Retrieved: 2020-03-02
[^get-samples]: [https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1gb7d2a6d2a9b4584cd985765d1ff46c94](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1gb7d2a6d2a9b4584cd985765d1ff46c94) Retrieved: 2020-03-02
[^sampling-type]: [https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceStructs.html#group__nvmlDeviceStructs_1gcef9440588e5d249cded88ce3efcc6b5](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceStructs.html#group__nvmlDeviceStructs_1gcef9440588e5d249cded88ce3efcc6b5) Retrieved: 2020-03-02

## Virtual Standard Hardware

A Virtual Standard Hardware could be created.
This would allow to compare different benchmarks conducted on different hardware platforms.
It could also be used to predict energy for different hardware platforms based on benchmarks (or prediction) on one platform.

## Select Best Platform

The work on a Virtual Standard Hardware could further be used to select and run networks on the best fitted hardware in a heterogeneous environment.

# Bonus Plots

## Energy and Time Per Epoch

### Energy

![ECG Energy per Epoch\label{fig:ecg-epoch-energy}](fig3/ecg-epoch-energy-boxplot.pdf)

![MNIST-BIG Energy per Epoch\label{fig:mnist-big-epoch-energy}](fig3/mnist-big-epoch-energy-boxplot.pdf)

![MNIST-SMALL Energy per Epoch\label{fig:mnist-small-epoch-energy}](fig3/mnist-small-epoch-energy-boxplot.pdf)

Figures \ref{fig:ecg-epoch-energy} to \ref{fig:mnist-small-epoch-energy} show the energy per epoch.

### Time

![ECG Time per Epoch\label{fig:ecg-epoch-time}](fig3/ecg-epoch-times-boxplot.pdf)

![MNIST-BIG Time per Epoch\label{fig:mnist-big-epoch-time}](fig3/mnist-big-epoch-times-boxplot.pdf)

![MNIST-SMALL Time per Epoch\label{fig:mnist-small-epoch-time}](fig3/mnist-small-epoch-times-boxplot.pdf)

Figures \ref{fig:ecg-epoch-time} to \ref{fig:mnist-small-epoch-time} show the time per epoch.

# Bibliography