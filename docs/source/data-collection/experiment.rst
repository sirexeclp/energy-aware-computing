Experiment
==========

experiment.yaml - file format
-----------------------------

experiment.yaml is read when the experiment is startet.
It lists which benchmarks should be run and which constraints should be applied.

Consider the following example:

.. code:: yaml

    experiments:
    - experiment_name: "power-limit"
        repeat: 5
        power_limits:
        - 150
        - 200
        data_path: /tmp/data-test
        benchmarks:
        - resnet
        - bert
    - experiment_name: "clock-limit"
        repeat: 5
        clock_limits:
        - [ 877, 1530 ]
        - [ 877, 135 ]
        data_path: /tmp/data-test
        benchmarks:
        - mnist-cnn
        - nbody

This example will run two experiments with the names "power-limit" and "clock-limit".
The power_limit experiment will run the benchmarks "resnet" and "bert".
Each benchmarks will be executed five times with 150W and 200W power limits applied.
The data will be saved to ``/tmp/data-test``.

The second experiment will run the benchmarks mnist-cnn and nbody.
Both will be run with clock limits set to 877Mhz (memory) and 1530 (graphics) als well as 877Mhz (memory) and 153 (graphics).
Each benchmark with each clock limit will be run five times.
The data is saved to the same directory as the data from the other experiment.

In total there will be 2 (experiments) * 2 (benchmarks) * 2 (limits) * 5 (technical repeats) = 40 runs



The root key must be named ``experiments`` and should contain a list of the experiments to run.
Each experiment entry in the list has a name (``experiment_name``).
The number of repetitions is given in ``repeat``.
The path in ``data_path`` will be used as the data root path to store collected data in.
Under ``benchmarks`` a list of benchmarks is selected, which should match the filenames of the benchmark definitions (see :ref:benchmarks).
An experiment can either provide a list of ``power_limits`` or a list of ``clock_limits`` to be used.
Each clock_limit must be a list with two entrys of the form ``[memory clock in MHz,  graphics clock in MHz]``.
The order of limits does not matter, as they are applied in random order.

