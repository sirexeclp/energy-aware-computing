.. Energy aware computing documentation master file, created by
   sphinx-quickstart on Tue Oct 13 16:25:51 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Energy aware computing's documentation!
==================================================

This is the documentation for the code written for my master thesis "On the Energy Consumption of Deep
Learning Workloads".

Begin by reading the :ref:`Architectural Overview` and :ref:`Data Model`.
Then you can read the in depth description of :ref:`Data Collection`,
which should contain everything you need to know, to run your own benchmarks.

Finally take a look at the data analysis module described in :ref:`Data Analysis`.

All functions are annotated with type hints according to :pep:`484`.
You can find the auto generated documentation build from docstrings and type hints
in the :ref:`modules` section.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   data-model
   data-collection/experiment
   data-collection/benchmarks
   analysis
   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
