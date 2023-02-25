Example using miR-125a maturation efficiency data
=================================================

The data used in this guide along with the necessary Python scripts can be found in the dynasigml_mir125a_example_ GitHub
repository. You can clone it with this command::

    git clone https://github.com/gregorpatof/dynasigml_mir125a_example

.. _dynasigml_mir125a_example: https://github.com/gregorpatof/dynasigml_mir125a_example

The miR-125a maturation efficiency data come from Fang & Bartel (https://doi.org/10.1016/j.molcel.2015.08.015) and have
been explored thoroughly in previous work (https://doi.org/10.1371/journal.pcbi.1010777).
The maturation efficiencies of over 50k miR-125a sequence variants have been measured using an enzymatic assay.

Here are the necessary steps to run DynaSig-ML on the miR-125a data:

.. toctree::
    :maxdepth: 2

    generating_mutations
    computing_signatures
    running_ml
    analysis
    lasso_coefs
    predictions
    parallel_runs
