Performance analysis
====================

Performance graphs were already generated in the previous step. Here is how they can be regenerated from the saved
DynaSigML_Model objects::

    from dynasigml.dynasig_ml_model import DynaSigML_Model, load_dynasigml_model_from_file

    dsml_model_hard = load_dynasigml_model_from_file("dsml_model_hard.pickle")
    dsml_model_inverted = load_dynasigml_model_from_file("dsml_model_inverted.pickle")

    dsml_model_hard.make_graphs('graphs_hard')
    dsml_model_inverted.make_graphs('graphs_inverted')

The graphs will be located in the folder specified, which will be created if it does not already exist.
For both user-specified ML models and LASSO regression, a scatter plot of predicted versus experimentally measured
values is generated. A plot of predictive (testing) R-squared as a function of the beta parameter for the Dynamical Signature is
also generated. Finally, for LASSO regression, a plot of predictive R-squared as a function of regularization
strength is generated.

.. note::

    If the **save_testing=True** flag is passed to the DynaSigML_Model constructor, text data frames
    are generated for all ML models with the predictions and
    measured values, so that the user can directly access the raw predictions if need be.
