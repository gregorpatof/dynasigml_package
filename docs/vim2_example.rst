Example using VIM-2 lactamase DMS data
======================================

The data used in this guide along with the necessary Python scripts can be found in the dynasigml_vim2_example_ GitHub
repository. You can clone it with this command::

    git clone https://github.com/gregorpatof/dynasigml_vim2_example

.. _dynasigml_vim2_example: https://github.com/gregorpatof/dynasigml_vim2_example

The fitness data for the deep mutational scan of VIM-2 lactamase comes from this article: https://doi.org/10.7554/eLife.56707.
The authors measured the fitness of bacteria expressing the mutant enzymes under selective pressures from various
concentrations of antibiotics. Thus, it is a direct measure of the VIM-2 sequence variant's ability to degrade
these antibiotics. The fitness data we are using here corresponds to the fitness measured at 128ug/mL ampicillin, at
37 degrees Celsius.

Here are the necessary steps to run DynaSig-ML on the VIM-2 DMS data:

.. toctree::
    :maxdepth: 2

    generating_mutations
    computing_signatures
    running_ml
    analysis
    lasso_coefs
    predictions
    parallel_runs
