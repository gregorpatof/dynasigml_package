Making new predictions
======================

Once the models are trained and the best parameter combinations have been identified, they can be used to make
predictions about the function of new sequence variants. Once you have *in silico* mutated structures of theoretical
variants, simply compute a DynaSigDF object from them, as was show in :ref:`computing_signatures`.


You can then use your saved DynaSigML_Model objects to make predictions about the new variants::

    from dynasigml.dynasig_df import load_pickled_dynasig_df
    from dynasigml.dynasig_ml_model import load_dynasigml_model_from_file

    dsdf_new_variants = load_pickled_dynasig_df("dsdf_new_variants.pickle")
    dsml_model = load_dynasigml_model_from_file("dsml_inverted.pickle")
    all_file_ids = dsdf_new_variants.get_file_ids()

    # get the data with the beta scaling factor used by the best LASSO model
    data_lasso = dsdf_double_mutants.get_data_array(all_file_ids, beta=dsml_model.get_best_beta_lasso())
    data_lasso = data_lasso[:, 2:] # reject first two columns (beta value and dummy experimental measure)
    predictions_lasso = dsml_model.predict_lasso(data_lasso)

    # same as above, but uses the best user-supplied ML model instead of LASSO regression
    data_ml = dsdf_double_mutants.get_data_array(all_file_ids, beta=dsml_model.get_best_beta_ml())
    data_ml = data_ml[:, 2:]
    predictions_ml = dsml_model.predict_ml(data_ml)

    lasso_zscores = (predictions_lasso - np.mean(predictions_lasso)) / np.std(predictions_lasso)
    ml_zscores = (predictions_ml - np.mean(predictions_ml)) / np.std(predictions_ml)
    with open("predictions_new_variants.df", "w") as f:
        f.write("file_id pred_lasso pred_ml pred_lasso_zscore pred_ml_zscore sum_zscores\n")
        for file_id, pred_lasso, pred_ml, lasso_z, ml_z in zip(all_file_ids, predictions_lasso, predictions_ml,
                                                                 lasso_zscores, mlp_zscores):
            f.write("{} {} {} {} {} {}\n".format(file_id, pred_lasso, pred_ml, lasso_z, ml_z, lasso_z + ml_z))

The predicted fitness values are now saved in the **predictions_double_mutants.df** file, along with the Z-scores for
the LASSO and ML predictions, and the sum of both Z-scores. The highest Z-score sum is indicative of both models
agreeing on this variant leading to high maturation efficiency.
