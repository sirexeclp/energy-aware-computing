Benchmarks
==========

Definitions
-----------

Each benchmark is defined by a yaml file in the ``benchmarks`` directory.

Consider the following example:

.. code:: yaml

    ---
    working_directory: ./
    module: finetune_bert
    args: []


The name of the benchmark is defined by name of the file (excluding the .yaml file extension).
A benchmark definition specifies the python module to be executed, in this case ``finetune_bert``.
The working_directory can be changed or as in this example left as is.
The process, which will later load the module changes its working directory accordingly before attempting to load the module.
The array specified in args will be passed to the module as command line arguments. 