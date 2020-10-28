Data Model
==========

Data are stored in UTF-8 encoded CSV files.
A file system structure is used to group the data.
The structure looks something like the following:

.. .. graphviz::

..    digraph foo {
..         /* paper size in inches */
..       //size="11.0,8.5";
..       /* locate label at top of drawing */
..       labelloc=t;
..       label="My File Tree Drawing";
..       rankdir = LR;
..       /* no directional arrow on connectors */
..       edge [dir=none];
..       /* nodes below are boxes (folders) */
..       node [shape=box];
..       "data-root" -> "experiment0";
..       "experiment0" -> "benchmark0";
..       node [shape=ellipse];
..    }


.. code::

    data-root
        |------experiment0
        |           |------benchmark0
                    |           |-----benchmark-run0
                                |           |--------repetition0
                                |           |            |-----system_info.json
                                |           |            |-----timestamps.csv
                                |           |            |-----gpu-power.csv
                                |           |            | ...
                                |           |
                                |           |--------repetition0
                                |                       |-----system_info.json
                                |                       |-----timestamps.csv
                                |                       |-----gpu-power.csv
                                |                       | ...
                    |           |-----benchmark-run1
                                |           |--------repetition0
                                            |           |-----system_info.json
                                            |           |-----timestamps.csv
                                            |           |-----gpu-power.csv
                                            |           | ...


``data-root`` is the root directory. It contains all experiments.
The name of the root directory is not important.
Below the ``data-root`` are directories for each experiment.
The name of a directory here is the name of that experiment.
Each experiment directory contains one or more benchmarks.
Each benchmark is again a directory, where the name of the benchmark is the file name.
Each benchmark contains multiple runs.
Each benchmark run is also a directory.
The name of each benchmark-run directory should consist of the
settings (power limit, clock limit) used for this run.
Each run contains one or more repetitions.
The repetition directories can just be named numerically:
0, 1, 2 ...
Each repetition directory contains multiple data files.

See below for a real world :ref:`Example`.

Data Files
----------

The files present depend on the data, that could be collected, which depends on the hardware platform.

- ``system_info.json``
    This json file contains information about the system, the benchmark was run on,
    such as os version, cuda version, etc.
    See also :ref:`system_info`.


- ``timestamps.csv``
    This file contains timestamps for epochs and experiment begin and end.
    It has the following fields:
    
    - timestamp
        ISO timestamp string 
    
    - event
        name of the event, the following events are supported

        - experiment_begin
        - train_begin
        - epoch_begin
        - batch_begin
        - batch_end
        - epoch_end
        - train_end
        - experiment_end

        All of these are optional except for ``experiment_begin`` and ``experiment_end``.
    
    - data
        index of the event, set to 0 if not used


- ``power-external.csv``
    Only present if external power measurements have been taken.
    This file has the following fields:
    
    - unnamed index
        an numeric index (you can ignore this)
    - timestamp
        ISO timestamp string, when the measurement was taken
    - d0c0
        power first channel of first device
    - d0c1
        power second channel of first device
    - d1c0
        power first channel of second device
    - d1c1
        power second channel of second device
    
    The naming convention is d{device-index}c{channel-index}.
    Each channel is the power measurement of one of four power supplies and 
    need to be summed together to get the power draw of the whole system at each
    timestep.
    All measurements are in milliwatt.


- ``gpu-power.csv``
    This file should always be present.
    It contains measurements taken using nvidia-smi and the "slow" one sample per call functions.
    Depending on the hardware some fields might be missing.

    - timestamp
        ISO timestamp string, when the measurement was taken
    - util-gpu
        the fraction of time at which at least one cuda kernel was run (in percent)
    - util-mem
        the fraction of used GPU memory (in percent)
    
    - clock-mem
        the current memory clock frequency (in MHz)
    - clock-gpu
        the current streaming processor clock frequency (in MHz)

    - app-clock-mem
        the set application clock (clock limit) for memory (in MHz)
    
    - app-clock-gpu
        the set application clock (clock limit) for streaming multiprocessors (in MHz)
    
    - enforced-power-limit
        the actually enforced power-limit (in mW)
    - total-energy
        the energy consumed until now, since the last reload of the driver (in mJ)
    - power-state
        the power state of the gpu stored as int representation of the corresponding pynvml3 enum
    - power
        the current power draw (in mW)
    - tmp
        the temperature (in °C)
    - pci-tx
        the pci express throughput of send data (bytes/s ?)
    - pci-rx
        the pci express throughput of recieved data (bytes/s ?)


- ``*_samples.csv``
    If the hardware supports the get_samples function the following 
    samples will be collected if available:

    - GPU_UTILIZATION_SAMPLES
    - MEMORY_UTILIZATION_SAMPLES
    - TOTAL_POWER_SAMPLES
        ca. 50Hz sampling frequency power samples

    The file will named like `{sampling_type | lower}_samples.csv`.

    Each of these files has the following fields:
    
    - unnamed index
        an numeric index (you can ignore this)
    - timestamp
        unix timestamp in microsecond resolution (microseconds since 1970)
    - value
        the value of the given sample

Example
-------

.. code::

    .
    ├── clock-limit
    │   ├── bert
    │   │   ├── 877MHz,1065MHz
    │   │   │   ├── 0
    │   │   │   │   ├── gpu-power.csv
    │   │   │   │   ├── gpu_utilization_samples.csv
    │   │   │   │   ├── memory_utilization_samples.csv
    │   │   │   │   ├── power-external.csv
    │   │   │   │   ├── system_info.json
    │   │   │   │   ├── timestamps.csv
    │   │   │   │   └── total_power_samples.csv
    │   │   │   ├── 1
    │   │   │   │   ├── gpu-power.csv
    │   │   │   │   ├── gpu_utilization_samples.csv
    │   │   │   │   ├── memory_utilization_samples.csv
    │   │   │   │   ├── power-external.csv
    │   │   │   │   ├── system_info.json
    │   │   │   │   ├── timestamps.csv
    │   │   │   │   └── total_power_samples.csv
    │   │   │   ├── 2
    │   │   │   │   ├── gpu-power.csv
    │   │   │   │   ├── gpu_utilization_samples.csv
    │   │   │   │   ├── memory_utilization_samples.csv
    │   │   │   │   ├── power-external.csv
    │   │   │   │   ├── system_info.json
    │   │   │   │   ├── timestamps.csv
    │   │   │   │   └── total_power_samples.csv
    │   │   │   ├── 3
    │   │   │   │   ├── gpu-power.csv
    │   │   │   │   ├── gpu_utilization_samples.csv
    │   │   │   │   ├── memory_utilization_samples.csv
    │   │   │   │   ├── power-external.csv
    │   │   │   │   ├── system_info.json
    │   │   │   │   ├── timestamps.csv
    │   │   │   │   └── total_power_samples.csv
    │   │   │   └── 4
    │   │   │       ├── gpu-power.csv
    │   │   │       ├── gpu_utilization_samples.csv
    │   │   │       ├── memory_utilization_samples.csv
    │   │   │       ├── power-external.csv
    │   │   │       ├── system_info.json
    │   │   │       ├── timestamps.csv
    │   │   │       └── total_power_samples.csv
    │   │   ├── 877MHz,1222MHz
    │   │   │   ├── 0
    │   │   │   │   ├── gpu-power.csv
    │   │   │   │   ├── gpu_utilization_samples.csv
    │   │   │   │   ├── memory_utilization_samples.csv
    │   │   │   │   ├── power-external.csv
    │   │   │   │   ├── system_info.json
    │   │   │   │   ├── timestamps.csv
    │   │   │   │   └── total_power_samples.csv
    │   │   │   ├── 1
    │   │   │   │   ├── gpu-power.csv
    │   │   │   │   ├── gpu_utilization_samples.csv
    │   │   │   │   ├── memory_utilization_samples.csv
    │   │   │   │   ├── power-external.csv
    │   │   │   │   ├── system_info.json
    │   │   │   │   ├── timestamps.csv
    │   │   │   │   └── total_power_samples.csv
    │   │   │   ├── 2
    │   │   │   │   ├── gpu-power.csv
    │   │   │   │   ├── gpu_utilization_samples.csv
    │   │   │   │   ├── memory_utilization_samples.csv
    │   │   │   │   ├── power-external.csv
    │   │   │   │   ├── system_info.json
    │   │   │   │   ├── timestamps.csv
    │   │   │   │   └── total_power_samples.csv
    │   │   │   ├── 3
    │   │   │   │   ├── gpu-power.csv
    │   │   │   │   ├── gpu_utilization_samples.csv
    │   │   │   │   ├── memory_utilization_samples.csv
    │   │   │   │   ├── power-external.csv
    │   │   │   │   ├── system_info.json
    │   │   │   │   ├── timestamps.csv
    │   │   │   │   └── total_power_samples.csv
    │   │   │   └── 4
    │   │   │       ├── gpu-power.csv
    │   │   │       ├── gpu_utilization_samples.csv
    │   │   │       ├── memory_utilization_samples.csv
    │   │   │       ├── power-external.csv
    │   │   │       ├── system_info.json
    │   │   │       ├── timestamps.csv
    │   │   │       └── total_power_samples.csv
