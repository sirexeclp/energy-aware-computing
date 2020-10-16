Architectural Overview
======================

This pages gives you an overview over this project, the architecture and main components.

The project consists of two main components benchmark execution including data collection and an offline 
data analsysis.

Benchmark execution and data collection
---------------------------------------

The benchmarks (and data collection) are built to be run inside a docker container.
To run all configured benchmarks use the command ``docker-compose up`` in the project root directory.
It will build the necessary container using the dockerfile and run the benchmarks and data collection inside it.

The entrypoint of the dockerfile is ``run.sh``. This script simply loads/executes the module ``gpyjoules.run_benchmark``.

The run benchmark module loads benchmark definitions from the ``benchmarks`` directory and the experiment from the ``experiment.yaml`` file.

See :ref:Experiment and :ref:Benchmark.


Data analsysis
--------------

The data analsysis pipeline is defined in the module ``new_analysis`` wich can be imported into a jupyter notebook
to allow for quick, exploratory data analsysis.
It can also be run as a module, where it will produce the outputs needed for the thesis.
