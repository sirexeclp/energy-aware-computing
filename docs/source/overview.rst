Architectural Overview
======================

This pages gives you an overview over this project, the architecture and main components.

The project consists of two main components: benchmark execution (including data collection) and an offline 
data analsysis.

Benchmark execution and data collection
---------------------------------------

The benchmarks (and data collection) are built to be run inside a docker container.

.. Note::
    You need to have `nvidia-docker`_ installed.


.. _`nvidia-docker`: https://github.com/NVIDIA/nvidia-docker

To run all configured benchmarks use the command ``docker-compose up`` in the project root directory.
It will build the necessary container using the dockerfile and run the benchmarks and data collection inside it.

The entrypoint of the dockerfile is ``run.sh``. This script simply loads/executes the module ``gpyjoules.run_benchmark``.

The run benchmark module loads benchmark definitions from the ``benchmarks`` directory and the experiment from the ``experiment.yaml`` file.

See :ref:`Experiment` and :ref:`Benchmarks`.

The function :func:`gpyjoules.run_benchmark.run_benchmark` is executed for each
benchmark-run.

It sets power an clock limits using pynvml3 and starts a new python interpreter
in a subprocess, which runs the :module:`gpyjoules.g_py_joules` module.
:module:`gpyjoules.g_py_joules` does the following things:

    1. Set correct arguments for wrapped module.
    2. Set ``CUDA_VISIBLE_DEVICES`` environment variable
    3. Set tensorflow log level
    4. Create event and cue for inter process communication
    5. Start data collection (pass event & queue)
    6. Monkey patch keras (pass event & queue)
    7. Chdir if needed
    8. Load and run the benchmark module

Each data collector also runs it's own process.
So if a data collector crashes, it does not affect the others.
Also since each benchmark runs in it's own process, if it fails,
the experiment will just continue with the next benchmark.


Data analsysis
--------------

The data analsysis pipeline is defined in the module ``new_analysis`` wich can be imported into a jupyter notebook
to allow for quick, exploratory data analsysis.
It can also be run as a module, where it will produce the outputs needed for the thesis.
