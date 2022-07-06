Running the machine learning models
===================================

When the needed DynaSigDF objects are all computed, a DynaSigML_Model object is built and can automatically train
and test LASSO regression and multilayer perceptron models, and print a report of their performance. In the case
of the VIM-2 dataset, 20 separate DynaSigDF objects are part of the GitHub repository and first need to be combined
together::

    from dynasigml.dynasig_df import combine_pickled_dynasig_dfs
    import glob

    separate_dsdfs = glob.glob("separate_dsdfs/*.pickle")
    combine_pickled_dynasig_dfs(separate_dsdfs, "combined_dsdf")

This will generate a file called "combined_dsdf.pickle" which contains the combined data from the 20 separated DynaSigDF
objects. One can then build the DynaSigML_Model object, and then test and train the machine learning models
automatically::

    from dynasigml.dynasig_ml_model import DynaSigML_Model
    import json

    with open("test_ids_8020.json") as f:
        test_ids = json.load(f)
    dsml_model = DynaSigML_Model("combined_dsdf.pickle", test_ids=test_ids)
    dsml_model.train_test_lasso()
    dsml_model.train_test_mlp()
    dsml_model.performance_report()

The performance report will print the best training and testing performance for both LASSO regression and MLP, in terms
of the coefficient of determination R². In general, a training R² too close to 1 means the model is overfitting the
training data.

.. note::
    When the default DynaSigML_Model constructor is used, the data will be randomly split in 80/20 training/testing
    sets. However, you can supply your own testing set by using the test_ids optional parameter (see detailed code
    documentation for more details). This is done here to ensure comparable results (results may still differ  a little
    as the training of the models is not deterministic).
