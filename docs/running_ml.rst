Running and saving the machine learning models
==============================================

When the needed DynaSigDF objects are all computed, a DynaSigML_Model object is built and can automatically train
and test LASSO regression and other user-specified ML models, and print a report of their performance. In the case
of the miR-125a dataset, 99 separate DynaSigDF objects are part of the GitHub repository and first need to be combined
together::

    from dynasigml.dynasig_df import combine_pickled_dynasig_dfs
    import glob

    separate_dsdfs = glob.glob("split_dsdfs/*.pickle")
    combine_pickled_dynasig_dfs(split_dsdfs, "combined_dsdf")

This will generate a file called "combined_dsdf.pickle" which contains the combined data from the 99 separated DynaSigDF
objects. One can then build the DynaSigML_Model object, and then test and train the machine learning models
automatically. In the following example, we run both the hard benchmark and the inverted benchmark from our previous work
(https://doi.org/10.1371/journal.pcbi.1010777) with LASSO regression, gradient boosting and random forest models::

    from dynasigml.dynasig_ml_model import DynaSigML_Model
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    import json


    def hard_test():
        with open("test_ids_hard.json") as f:
            test_ids = json.load(f)
        with open("train_ids_hard.json") as f:
            train_ids = json.load(f)
        dsml_model = DynaSigML_Model("combined_dsdf.pickle", test_ids=test_ids, train_ids=train_ids, verbose=True,
                                     ml_models=[RandomForestRegressor(),
                                                GradientBoostingRegressor()],
                                     ml_models_labels=["RF", "GBR"], measured_property="maturation efficiency")
        dsml_model.train_test_lasso()
        dsml_model.train_test_ml_models()
        dsml_model.performance_report()
        dsml_model.save_to_file('dsml_model_hard')
        dsml_model.make_graphs('graphs_hard')


    def inverted_test():
        with open("test_ids_inverted.json") as f:
            test_ids = json.load(f)
        with open("train_ids_inverted.json") as f:
            train_ids = json.load(f)
        dsml_model = DynaSigML_Model("combined_dsdf.pickle", test_ids=test_ids, train_ids=train_ids, verbose=True,
                                     ml_models=[RandomForestRegressor(),
                                                GradientBoostingRegressor()],
                                     ml_models_labels=["RF", "GBR"], measured_property="maturation efficiency")
        dsml_model.train_test_lasso()
        dsml_model.train_test_ml_models()
        dsml_model.performance_report()
        dsml_model.save_to_file('dsml_model_inverted')
        dsml_model.make_graphs('graphs_inverted')

    hard_test()
    inverted_test()

The performance report will print the best testing performance for both LASSO regression and the other ML
models (bagged together), in terms of the coefficient of determination R².

The models are saved with the **save_to_file** method. Depending on the dataset size and the number of tested beta
values for the Dynamical Signatures, the training and
testing of LASSO and ML models can take up to a few minutes. Thus, it is convenient to save the whole
DynaSigML_Model object, making further analysis faster since the ML models are already trained.

.. note::
    When the default DynaSigML_Model constructor is used, the data will be randomly split in 80/20 training/testing
    sets. However, we recommend supplying your own carefully constructed training and testing sets, as we have done
    here (see https://doi.org/10.1371/journal.pcbi.1010777 for details). Otherwise, it is not guaranteed that a true
    dynamical signal is captured and overfitting would likely happen.
