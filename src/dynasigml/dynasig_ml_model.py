from dynasigml.dynasig_df import load_pickled_dynasig_df
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import BaseEstimator, clone
import numpy as np
import random
import matplotlib.pyplot as plt
import warnings
import os
import pickle
from nrgten.encom import ENCoM


def load_dynasigml_model_from_file(filename):
    with open(filename, 'rb') as f:
        dsml = pickle.load(f)
    return dsml


class DynaSigML_Model:
    """Class used to train and test machine learning models from a saved DynaSigDF object.

        The DynaSigML_Model class can automatically train and test ML models (LASSO regression and multilayer
        perceptrons) using Dynamical Signatures saved as a DynaSigDF object. It also automatically generates a
        performance report, plots testing performance and can map the LASSO coefficients back on a PDB structure.

        Attributes:
            dynasigdf (DynaSigDF): The data frame of Dynamical Signatures and other experimental measures, as a
                                   DynaSigDF object.
            pred_cols (list): The indices of the columns (variables) to be used as predictors in the ML models.
            targ_col (int): The index of the outcome to predict.
            n (int): The total number of observations (sequence variants).
            file_ids (list): The list of file identifiers for the sequence variants (generated with id_func).
            test_ids (list): The file identifiers of sequence variants in the testing set.
            train_ids (list): The file identifiers of sequence variants in the training set.
            alphas (list): The list of different regularization strengths to be used in LASSO regression.
            max_iter_lasso (int): Maximum number of LASSO iterations.
            lasso_stats_dict (dict): Performance of LASSO models in dictionary format.
            standardization_dict (dict): Transformation to apply to new predictors to standardize them (mean,
                                         standard deviation for every column) for each beta value.
            verbose (bool): If True, information will be printed throughout the training and testing of ML models.
            save_testing (bool): If True, testing set predictions will be saved for every trained model.
    """

    def __init__(self, dynasigdf_file, alphas=None, betas=None, test_ids=None, train_ids=None, id_func=None,
                 test_prop=0.2, predictor_columns=None, target_column=None, verbose=False, save_testing=False,
                 max_iter_lasso=1000, ml_models=None, ml_models_labels=None, measured_property="value"):
        """Constructor for the DynaSigML_Model class.

        Args:
            dynasigdf_file (str): The filename for the DynaSigDF object.
            alphas (list, optional): The list of different regularization strengths to be used in LASSO regression.
            betas (list, optional): The list of thermodynamic scaling factors to use. If not supplied, all values in the
                                    DynaSigDF object are tested.
            test_ids (list, optional): The file identifiers of sequence variants in the testing set.
            train_ids (list, optional): The file identifiers of sequence variants in the training set. Can be omitted
                                        if test_ids is supplied, but test_ids cannot be omitted if train_ids is
                                        supplied.
            id_func: (function, optional): function to be ran on the input files to generate the filename IDs (which are
                                          used to identify each DynaSig in the final dataframe). If not supplied,
                                          defaults to just the filename at the end of the path, minus the extension.
            test_prop: (float, optional): proportion of variants to use for the testing set (random sampling), if
                                          test_ids is not supplied.
            predictor_columns (list, optional): A list of str names of the columns to be used as predictors in the ML
                                                models. To select the whole Dynamical Signature, include 'dynasig'.
            target_column (str, optional): The str name of the column to predict.
            verbose (bool, optional): If True, info will be printed.
            save_testing (bool, optional): If True, testing set predictions are saved.
            max_iter_lasso (int, optional): Maximum number of LASSO iterations.
            ml_models (list, optional): Additional sklearn ML models to run in addition to LASSO
            ml_models_labels (list, optional): Labels (str) for added ML models
            measured_property (str, optional): the name of the measured and predicted property
            """
        self.dynasigdf = load_pickled_dynasig_df(dynasigdf_file)
        if predictor_columns is None:
            predictor_columns = []
            for label in self.dynasigdf.exp_labels[1:]:
                predictor_columns.append(self.dynasigdf.get_column_index(label))
            for index in self.dynasigdf.get_dynasig_indices():
                predictor_columns.append(index)
        else:
            assert isinstance(predictor_columns, list)
            tmp_columns = []
            for col in predictor_columns:
                if isinstance(col, int):
                    tmp_columns.append(col)
                elif isinstance(col, str):
                    if col == 'dynasig':
                        for index in self.dynasigdf.get_dynasig_indices():
                            tmp_columns.append(index)
                    elif col == 'svib':
                        tmp_columns.append(self.dynasigdf.get_svib_index())
                    else:
                        tmp_columns.append(self.dynasigdf.get_column_index(col))
            predictor_columns = tmp_columns
        self.pred_cols = predictor_columns
        if target_column is None:
            target_column = 1
        elif isinstance(target_column, str):
            target_column = self.dynasigdf.get_column_index(target_column)
        else:
            assert isinstance(target_column, int)
        self.targ_col = target_column
        if id_func is None:
            id_func = lambda x: x.split('/')[-1]
        self.id_func = id_func
        if alphas is None:
            alphas = [2**x for x in range(-15, 1)]
        self.n = len(self.dynasigdf.files_list)
        self.file_ids = [self.id_func(filename) for filename in self.dynasigdf.files_list]
        if train_ids is not None:
            if test_ids is None:
                raise ValueError("Training set ids specified but not testing set ids (the reverse should be done).")
        elif test_ids is not None:
            train_ids = set([x for x in self.file_ids if x not in test_ids])
            assert len(train_ids) + len(test_ids) == self.n
        else:
            n_test = int(test_prop*self.n + 1)
            test_ids = random.sample(self.file_ids, n_test)
            train_ids = [x for x in self.file_ids if x not in test_ids]
        self.test_ids = test_ids
        self.train_ids = train_ids
        self.alphas = alphas
        if betas is None:
            self.beta_values = self.dynasigdf.beta_values
        else:
            self.beta_values = betas
        self.max_iter_lasso = max_iter_lasso
        self.verbose = verbose
        self.save_testing = save_testing
        self.measured_property = measured_property
        self.lasso_stats_dict = dict()
        self.standardization_dict = dict()
        if ml_models is not None:
            if ml_models_labels is None:
                raise ValueError("Additional ML models supplied but no matching labels " +
                                 "(add ml_models_labels=['your_label'] to the constructor).")
            if isinstance(ml_models, list):
                assert isinstance(ml_models_labels, list)
                assert len(ml_models) == len(ml_models_labels)
                if len(ml_models_labels) != len(set(ml_models_labels)):
                    raise ValueError("ml_models_labels are not unique (each string identifier needs to be)")
                for i in range(len(ml_models)):
                    assert isinstance(ml_models[i], BaseEstimator)
                    assert isinstance(ml_models_labels[i], str)
            else:
                assert isinstance(ml_models, BaseEstimator)
                ml_models = [ml_models]
                if isinstance(ml_models_labels, list):
                    assert len(ml_models_labels) == 1
                else:
                    assert isinstance(ml_models_labels, str)
                    ml_models_labels = [ml_models_labels]
        elif ml_models_labels is not None:
            raise ValueError("ml_models_labels supplied but no ml_models supplied")
        self.ml_models = ml_models
        self.ml_models_labels = ml_models_labels
        self.ml_models_dict = dict()

    def print_verbose(self, s):
        if self.verbose:
            print(s)

    def save_to_file(self, filename):
        self.id_func = None
        with open('{}.pickle'.format(filename), 'wb') as f:
            pickle.dump(self, f)

    def get_train_test_std(self, beta):
        train_data = self.dynasigdf.get_data_array(self.train_ids, beta)
        if len(self.test_ids) > 0:
            test_data = self.dynasigdf.get_data_array(self.test_ids, beta)
        else:
            test_data = None
        return standardize_data(train_data, test_data, self.pred_cols)

    def _fit_model(self, model, train_data, test_data, stats_dict):
        warnings.filterwarnings("error")
        try:
            model.fit(train_data[:, self.pred_cols], train_data[:, self.targ_col])
            stats_dict['Converged'] = True
        except ConvergenceWarning:
            warnings.filterwarnings("ignore")
            model.fit(train_data[:, self.pred_cols], train_data[:, self.targ_col])
            stats_dict['Converged'] = False
        stats_dict['Training_R2'] = model.score(train_data[:, self.pred_cols], train_data[:, self.targ_col])
        stats_dict['model'] = model
        if test_data is not None:
            stats_dict['Testing_R2'] = model.score(test_data[:, self.pred_cols], test_data[:, self.targ_col])

            if self.save_testing:
                stats_dict['test_preds'] = [test_data[:, self.targ_col], model.predict(test_data[:, self.pred_cols])]
        warnings.filterwarnings("default")

    def train_test_lasso(self):
        for beta in self.beta_values:
            train_data, test_data, standardization = self.get_train_test_std(beta)
            self.standardization_dict[beta] = standardization
            self.lasso_stats_dict[beta] = dict()
            for alpha in self.alphas:
                self.lasso_stats_dict[beta][alpha] = dict()
                lasso_mod = Lasso(alpha=alpha, selection='random', max_iter=self.max_iter_lasso)
                self._fit_model(lasso_mod, train_data, test_data, self.lasso_stats_dict[beta][alpha])
                self.print_verbose("Finished training LASSO model with beta={} and alpha={}".format(beta, alpha))

    def train_test_ml_models(self):
        for label, model in zip(self.ml_models_labels, self.ml_models):
            self.ml_models_dict[label] = dict()
            for beta in self.beta_values:
                train_data, test_data, standardization = self.get_train_test_std(beta)
                self.ml_models_dict[label][beta] = dict()
                mod = clone(model)
                self._fit_model(mod, train_data, test_data, self.ml_models_dict[label][beta])
                self.print_verbose("Finished training {} model with beta={}".format(label, beta))

    def get_best_params_lasso(self):
        best_testing = float('-Inf')
        best_params_testing = [None, None, None]
        best_test_model = None
        train_best_test = None
        for beta in self.lasso_stats_dict:
            for alpha in self.lasso_stats_dict[beta]:
                test_r2 = self.lasso_stats_dict[beta][alpha]['Testing_R2']
                train_r2 = self.lasso_stats_dict[beta][alpha]['Training_R2']
                if test_r2 > best_testing:
                    best_testing = test_r2
                    best_params_testing = [beta, alpha, self.lasso_stats_dict[beta][alpha]['Converged']]
                    train_best_test = train_r2
                    best_test_model = self.lasso_stats_dict[beta][alpha]['model']
        return best_testing, best_params_testing, train_best_test, best_test_model

    def get_best_params_ml(self):
        best_testing = float('-Inf')
        best_beta_testing = None
        train_best_test = None
        best_test_model = None
        best_label_testing = None
        for label in self.ml_models_dict:
            for beta in self.ml_models_dict[label]:
                train_r2 = self.ml_models_dict[label][beta]['Training_R2']
                test_r2 = self.ml_models_dict[label][beta]['Testing_R2']
                if test_r2 > best_testing:
                    best_testing = test_r2
                    best_beta_testing = beta
                    train_best_test = train_r2
                    best_label_testing = label
                    best_test_model = self.ml_models_dict[label][beta]['model']
        return best_testing, best_beta_testing, train_best_test, best_test_model, best_label_testing

    def performance_report(self):
        if self.lasso_stats_dict is not None:
            print("LASSO regression performance:")
            print("\n".join(get_performance_report_lasso(*self.get_best_params_lasso())))
        if self.ml_models_dict is not None:
            print("Other ML models performance:")
            print("\n".join(get_performance_report_ml(*self.get_best_params_ml())))

    def get_best_beta_values(self):
        best_betas = []
        if len(self.lasso_stats_dict) > 0:
            best_betas.append(self.get_best_beta_lasso())
        if len(self.ml_models_dict) > 0:
            beta_ml = self.get_best_beta_ml()
            if beta_ml != best_betas[0]:
                best_betas.append(beta_ml)
        return best_betas

    def get_best_beta_lasso(self):
        return self.get_best_params_lasso()[1][0]

    def get_best_beta_ml(self):
        return self.get_best_params_ml()[3]

    def make_graphs(self, folder):
        if not os.path.isdir(folder):
            os.mkdir(folder)
        if len(self.lasso_stats_dict) != 0:
            self._make_lasso_graphs(folder)
        if len(self.ml_models_dict) != 0:
            self._make_ml_graphs(folder)

    def _make_plot(self, betas, test_r2s, best_beta_testing, folder, label, best_test_model, best_testing, model_dict):
        plt.clf()
        plt.plot(np.log(betas), test_r2s)
        plt.ylim(ymin=0, ymax=1)
        plt.xlabel("Log beta")
        plt.ylabel("Predictive R²")
        plt.savefig("{}/{}_testing_r2_beta.png".format(folder, label))

        test_data = self.dynasigdf.get_data_array(self.test_ids, best_beta_testing)
        train_data = self.dynasigdf.get_data_array(self.train_ids, best_beta_testing)
        train_data, test_data, standardization = standardize_data(train_data, test_data, self.pred_cols)
        self.standardization_dict[best_beta_testing] = standardization
        preds = best_test_model.predict(test_data[:, self.pred_cols])
        reals = test_data[:, self.targ_col]
        plt.clf()
        plt.scatter(preds, reals, alpha=0.5)
        plt.xlabel("Predicted {}".format(self.measured_property))
        plt.ylabel("Experimental {}".format(self.measured_property))
        text = "R² = {:.2f}\nEF10% = {:.2f}".format(best_testing, _get_ef(reals, preds))
        plt.gca().text(0.05, 0.95, text, transform=plt.gca().transAxes,
                       fontsize=14, verticalalignment='top')
        plt.savefig("{}/{}_testing_performance.png".format(folder, label))
        indices = np.argsort(-preds)
        with open("{}/{}_testing_performance.df".format(folder, label), "w") as f:
            f.write("variant predicted measured\n")
            for i in indices:
                f.write("{} {} {}\n".format(self.test_ids[i], preds[i], reals[i]))
        if self.save_testing:
            with open("{}/{}_testing_preds_full.df".format(folder, label), "w") as f:
                f.write("beta variant predicted measured\n")
                for beta in model_dict:
                    sreals, spreds = model_dict[beta]['test_preds']
                    for variant, pred, real in zip(self.test_ids, spreds, sreals):
                        f.write("{} {} {} {}\n".format(beta, variant, pred, real))

    def _make_ml_graphs(self, folder):
        for label in self.ml_models_dict:
            best_beta_testing = None
            betas = []
            test_r2s = []
            best_testing = float('-Inf')
            best_test_model = None
            for beta in self.ml_models_dict[label]:
                betas.append(beta)
                test_r2s.append(self.ml_models_dict[label][beta]['Testing_R2'])
                if test_r2s[-1] > best_testing:
                    best_testing = test_r2s[-1]
                    best_test_model = self.ml_models_dict[label][beta]['model']
                    best_beta_testing = beta
            self._make_plot(betas, test_r2s, best_beta_testing, folder, label,
                            best_test_model, best_testing, self.ml_models_dict[label])

    def _make_lasso_graphs(self, folder):
        best_testing, best_params_testing, train_best_test, best_test_model = self.get_best_params_lasso()
        best_beta = best_params_testing[0]
        x = []
        test_r2 = []
        for alpha in self.lasso_stats_dict[best_beta]:
            x.append(np.log2(alpha))
            te_r2 = self.lasso_stats_dict[best_beta][alpha]['Testing_R2']
            if te_r2 < 0:
                te_r2 = 0
            test_r2.append(te_r2)
        plt.clf()
        plt.plot(x, test_r2)
        plt.ylim(ymin=0, ymax=1)
        plt.xlabel("Log₂ regularization strength")
        plt.ylabel("Predictive R²")
        plt.savefig("{}/lasso_testing_r2_alpha.png".format(folder))

        test_data = self.dynasigdf.get_data_array(self.test_ids, best_beta)
        train_data = self.dynasigdf.get_data_array(self.train_ids, best_beta)
        train_data, test_data, standardization = standardize_data(train_data, test_data, self.pred_cols)
        self.standardization_dict[best_beta] = standardization
        preds = best_test_model.predict(test_data[:, self.pred_cols])
        reals = test_data[:, self.targ_col]
        plt.clf()
        plt.scatter(preds, reals, alpha=0.5)
        plt.xlabel("Predicted {}".format(self.measured_property))
        plt.ylabel("Experimental {}".format(self.measured_property))
        text = "R² = {:.2f}\nEF10% = {:.2f}".format(best_testing, _get_ef(reals, preds))
        plt.gca().text(0.05, 0.95, text, transform=plt.gca().transAxes,
                       fontsize=14, verticalalignment='top')
        plt.savefig("{}/lasso_testing_performance.png".format(folder))
        indices = np.argsort(-preds)
        with open("{}/lasso_testing_performance.df".format(folder), "w") as f:
            f.write("variant predicted measured\n")
            for i in indices:
                f.write("{} {} {}\n".format(self.test_ids[i], preds[i], reals[i]))

        plt.clf()
        betas = []
        test_r2 = []
        for beta in self.lasso_stats_dict:
            max_r2 = 0
            for alpha in self.lasso_stats_dict[beta]:
                r2 = self.lasso_stats_dict[beta][alpha]['Testing_R2']
                if r2 > max_r2:
                    max_r2 = r2
            betas.append(np.log(beta))
            test_r2.append(max_r2)
        plt.plot(betas, test_r2)
        plt.ylim(ymin=0, ymax=1)
        plt.xlabel("Log beta")
        plt.ylabel("Best predictive R²")
        plt.savefig("{}/lasso_testing_r2_beta.png".format(folder))

        if self.save_testing:
            with open("{}/lasso_testing_preds_full.df".format(folder), "w") as f:
                f.write("beta alpha variant predicted measured\n")
                for beta in self.lasso_stats_dict:
                    for alpha in self.lasso_stats_dict[beta]:
                        sreals, spreds = self.lasso_stats_dict[beta][alpha]['test_preds']
                        for variant, pred, real in zip(self.test_ids, spreds, sreals):
                            f.write("{} {} {} {} {}\n".format(beta, alpha, variant, pred, real))

    def predict_lasso(self, data_array):
        assert len(data_array[0]) == len(self.pred_cols)
        best_params = self.get_best_params_lasso()
        best_lasso_model = best_params[-1]
        best_beta = best_params[1][0]
        standardization = self.standardization_dict[best_beta]
        assert len(standardization) == len(self.pred_cols)
        for i in range(data_array.shape[1]):
            mean, sd = standardization[i]
            data_array[:, i] -= mean
            if sd > 0:
                data_array[:, i] /= sd
        return best_lasso_model.predict(data_array)

    def predict_ml(self, data_array):
        assert len(data_array[0]) == len(self.pred_cols)
        best_params = self.get_best_params_ml()
        best_ml_model = best_params[-1]
        best_beta = best_params[1]
        standardization = self.standardization_dict[best_beta]
        assert len(standardization) == len(self.pred_cols)
        for i in range(data_array.shape[1]):
            mean, sd = standardization[i]
            data_array[:, i] -= mean
            if sd > 0:
                data_array[:, i] /= sd
        return best_ml_model.predict(data_array)

    def map_coefficients(self, wt_pdb_file, output_pdb_file, beta=None, alpha=None, added_massdef=None,
                         added_atypes=None):
        if added_massdef is None:
            enc = ENCoM(wt_pdb_file, solve=False)
        else:
            assert added_atypes is not None
            enc = ENCoM(wt_pdb_file, solve=False, added_massdef=added_massdef, added_atypes=added_atypes)
        n = len(enc.mol.masses)
        if beta is None:
            assert alpha is None
            lasso_mod = self.get_best_params_lasso()[-1]
        else:
            assert alpha is not None
            lasso_mod = self.lasso_stats_dict[beta][alpha]['model']
        coefs_og = lasso_mod.coef_[-n:]
        coefs = np.array(coefs_og)
        coefs /= np.max(np.abs(coefs))
        enc.set_bfactors(np.abs(coefs))
        enc.set_occupancy(coefs)
        enc._write_to_file(output_pdb_file)
        return coefs_og


def get_performance_report_lasso(best_testing, best_params_testing, train_best_test, best_test_model):
    report = ["Best testing (predictive) R²: {:.2f} with alpha={} and beta={} (associated training R²:{:.2f})"
              .format(best_testing, best_params_testing[1], best_params_testing[0], train_best_test)]
    return report


def get_performance_report_ml(best_testing, best_beta_testing, train_best_test, best_test_model, best_label_testing):
    report = ["Best testing (predictive) R²: {:.2f} with {} model, beta={}, (associated training R²:{:.2f})".format(
        best_testing, best_label_testing, best_beta_testing, train_best_test)]
    return report


def standardize_data(train_data, test_data, cols):
    standardization = np.zeros((len(cols), 2))
    count = 0
    for col in cols:
        if test_data is not None:
            mean = np.mean(np.concatenate((train_data[:, col], test_data[:, col])))
            sd = np.std(np.concatenate((train_data[:, col], test_data[:, col])))
            standardization[count] = [mean, sd]
            for data in [train_data, test_data]:
                data[:, col] -= mean
                if sd != 0:
                    data[:, col] /= sd
        else:
            mean = np.mean(train_data[:, col])
            sd = np.std(train_data[:, col])
            standardization[count] = [mean, sd]
            train_data[:, col] -= mean
            if sd != 0:
                train_data[:, col] /= sd
        count += 1
    return train_data, test_data, standardization


def adjusted_r2(r2, n, p):
    return 1 - (1-r2)*(n-1)/(n-p-1)


def _get_ef(reals, preds, prop=0.1):
    data = np.zeros((len(reals), 2))
    data[:, 0] = reals
    data[:, 1] = preds
    data = data[data[:, 1].argsort()]
    max_index = int(len(data)*prop+1)
    thresh = sorted(data[:, 0])[-max_index+1]
    count = 0
    for i in range(max_index):
        if data[-i, 0] >= thresh:
            count += 1
    return float(count)/(max_index-1) / prop
