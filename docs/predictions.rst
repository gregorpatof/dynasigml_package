Making new predictions
======================

Once the models are trained and the best parameter combinations have been identified, they can be used to make
predictions about the function of new sequence variants. For the VIM-2 dataset, we have generated 190 double mutants
by combining the 20 point mutations with highest measured fitness. They are located in the **double_mutants** folder.
Since we are not running the full Rosetta protocol that the authors of the VIM-2 dataset used, we have to use a
predictor based on the Dynamical Signatures only::

    from dynasigml.dynasig_df import DynaSigDF
    from dynasigml.dynasig_ml_model import DynaSigML_Model
    import json

    with open("test_ids_8020.json") as f:
        test_ids = json.load(f)
    dsml_model = DynaSigML_Model("combined_dsdf.pickle", test_ids=test_ids, predictor_columns=["dynasig"])
    dsml_model.train_test_lasso()
    dsml_model.train_test_mlp()
    dsml_model.make_graphs('graphs_folder_dynasig_only')
    dsml_model.save_to_file("dsml_dynasig_only")

We also have to generate the Dynamical Signatures for the double mutants::

    from dynasigml.dynasig_df import DynaSigDF
    from dynasigml.dynasig_ml_model import load_dynasigml_model_from_file
    import glob

    filenames_list = glob.glob("double_mutants/*.pdb")
    exp_data = [0 for x in filenames_list]
    exp_labels = ["dummy"]
    dsml = load_dynasigml_model_from_file("dsml_dynasig_only.pickle")
    best_beta_values = dsml.get_best_beta_values()
    dsdf = DynaSigDF(filenames_list, exp_data, exp_labels, "dsdf_double_mutants", beta_values=best_beta_values)

This should take around 10 minutes to run, but the resulting **dsdf_double_mutants.pickle** file is already part of the
repository, so you do not have to run it to execute the next step.

We can then use the saved LASSO and MLP models to make predictions about the double mutants::

    from dynasigml.dynasig_df import load_pickled_dynasig_df
    from dynasigml.dynasig_ml_model import load_dynasigml_model_from_file

    dsdf_double_mutants = load_pickled_dynasig_df("dsdf_double_mutants.pickle")
    dsml_model = load_dynasigml_model_from_file("dsml_dynasig_only.pickle")
    all_file_ids = dsdf_double_mutants.get_file_ids()

    data_lasso = dsdf_double_mutants.get_data_array(all_file_ids, beta=dsml_model.get_best_beta_lasso())
    data_lasso = data_lasso[:, 2:] # reject first two columns (beta value and dummy experimental measure)
    predictions_lasso = dsml_model.predict_lasso(data_lasso)

    data_mlp = dsdf_double_mutants.get_data_array(all_file_ids, beta=dsml_model.get_best_beta_mlp())
    data_mlp = data_mlp[:, 2:]
    predictions_mlp = dsml_model.predict_mlp(data_mlp)

    lasso_zscores = (predictions_lasso - np.mean(predictions_lasso)) / np.std(predictions_lasso)
    mlp_zscores = (predictions_mlp - np.mean(predictions_mlp)) / np.std(predictions_mlp)
    with open("predictions_double_mutants.df", "w") as f:
        f.write("file_id pred_lasso pred_mlp pred_lasso_zscore pred_mlp_zscore sum_zscores\n")
        for file_id, pred_lasso, pred_mlp, lasso_z, mlp_z in zip(all_file_ids, predictions_lasso, predictions_mlp,
                                                                 lasso_zscores, mlp_zscores):
            f.write("{} {} {} {} {} {}\n".format(file_id, pred_lasso, pred_mlp, lasso_z, mlp_z, lasso_z + mlp_z))

The predicted fitness values are now saved in the **predictions_double_mutants.df** file, along with the Z-scores for
the LASSO and MLP predictions, and the sum of both Z-scores. The highest Z-score sum is indicative of both models
agreeing on this variant leading to high evolutionary fitness.
