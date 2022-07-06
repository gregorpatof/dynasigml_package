Performance analysis and saving the models
==========================================

Depending on the dataset size and the number of tested beta values for the Dynamical Signatures, the training and
testing of LASSO and MLP models can take up to a few minutes. Fortunately, it is possible to save the whole
DynaSigML_Model object, making further analysis faster because the models are already trained::

    from dynasigml.dynasig_ml_model import DynaSigML_Model, load_dynasigml_model_from_file
    import json

    with open("test_ids_8020.json") as f:
        test_ids = json.load(f)
    dsml_model = DynaSigML_Model("combined_dsdf.pickle", test_ids=test_ids)
    dsml_model.test_train_lasso()
    dsml_model.test_train_mlp()
    dsml_model.save_to_file("dsml_model")

This will save the DynaSigML_Model object to a file called dsml_model.pickle. It can then be loaded again, already
trained::

    from dynasigml.dynasig_ml_model import DynaSigML_Model, load_dynasigml_model_from_file

    dsml_model = load_dynasigml_model_from_file("dsml_model.pickle")

Assuming both MLP and LASSO models were trained, the make_graphs() method can be called to generate five graphs
automatically::

    from dynasigml.dynasig_ml_model import DynaSigML_Model, load_dynasigml_model_from_file

    dsml_model = load_dynasigml_model_from_file("dsml_model.pickle")
    dsml_model.make_graphs("graphs_folder")

The five graphs will be located in the folder specified, which will be created if it does not already exist.
For both MLP and LASSO models, a scatter plot of predicted versus experimentally measured values is generated.
A plot of predictive (training) R-squared as a function of the beta parameter for the Dynamical Signature is
also generated. Finally, for LASSO regression, a plot of predictive R-squared as a function of regularization
strength is generated. For both MLP and LASSO, text data frames are also generated with the predictions and
measured values, so that the user can directly access the raw predictions if need be.
