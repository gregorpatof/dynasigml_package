from dynasigml.dynasig_df import load_pickled_dynasig_df
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
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
            n_layers (int): The number of hidden layers in multilayer perceptrons (MLPs).
            layer_size (int): The size of each MLP hidden layer.
            max_iter_lasso (int): Maximum number of LASSO iterations.
            max_iter_mlp (int): Maximum number of MLP iterations.
            lasso_stats_dict (dict): Performance of LASSO models in dictionary format.
            mlp_stats_dict (dict): Performance of MLP models in dictionary format.
            standardization (ndarray): Transformation to apply to new predictors to standardize them (mean,
                                       standard deviation for every column).
    """

    def __init__(self, dynasigdf_file, alphas=None, test_ids=None, train_ids=None, id_func=None, test_prop=0.2,
                 n_layers=2, layer_size=None, predictor_columns=None, target_column=None, max_iter_lasso=1000,
                 max_iter_mlp=200):
        """Constructor for the DynaSigML_Model class.

        Args:
            dynasigdf_file (str): The filename for the DynaSigDF object.
            alphas (list, optional): The list of different regularization strengths to be used in LASSO regression.
            test_ids (list, optional): The file identifiers of sequence variants in the testing set.
            train_ids (list, optional): The file identifiers of sequence variants in the training set. Can be omitted
                                        if test_ids is supplied, but test_ids cannot be omitted if train_ids is
                                        supplied.
            id_func: (function, optional): function to be ran on the input files to generate the filename IDs (which are
                                          used to identify each DynaSig in the final dataframe). If not supplied,
                                          defaults to just the filename at the end of the path, minus the extension.
            test_prop: (float, optional): proportion of variants to use for the testing set (random sampling), if
                                          test_ids is not supplied.
            n_layers (int, optional): The number of hidden layers in multilayer perceptrons (MLPs).
            layer_size (int, optional): The size of each MLP hidden layer.
            predictor_columns (list, optional): A list of str names of the columns to be used as predictors in the ML
                                                models. To select the whole Dynamical Signature, include 'dynasig'.
            target_column (str, optional): The str name of the column to predict.
            max_iter_lasso (int, optional): Maximum number of LASSO iterations.
            max_iter_mlp (int, optional): Maximum number of MLP iterations.
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
        if layer_size is None:
            layer_size = len(self.pred_cols)
        if id_func is None:
            id_func = lambda x: x.split('/')[-1]
        if alphas is None:
            alphas = [2**x for x in range(-15, 1)]
        self.n = len(self.dynasigdf.files_list)
        self.file_ids = [id_func(filename) for filename in self.dynasigdf.files_list]
        if train_ids is not None:
            assert len(train_ids) + len(test_ids) == self.n
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
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.max_iter_lasso = max_iter_lasso
        self.max_iter_mlp = max_iter_mlp
        self.lasso_stats_dict = dict()
        self.mlp_stats_dict = dict()
        self.standardization_dict = dict()

    def save_to_file(self, filename):
        self.id_func = None
        with open('{}.pickle'.format(filename), 'wb') as f:
            pickle.dump(self, f)

    def train_test_lasso(self):
        for beta in self.dynasigdf.beta_values:
            train_data = self.dynasigdf.get_data_array(self.train_ids, beta)
            test_data = self.dynasigdf.get_data_array(self.test_ids, beta)
            train_data, test_data, standardization = standardize_data(train_data, test_data, self.pred_cols)
            self.standardization_dict[beta] = standardization
            self.lasso_stats_dict[beta] = dict()
            for alpha in self.alphas:
                self.lasso_stats_dict[beta][alpha] = dict()
                lasso_mod = Lasso(alpha=alpha, selection='random', max_iter=self.max_iter_lasso)
                warnings.filterwarnings("error")
                try:
                    lasso_mod.fit(train_data[:, self.pred_cols], train_data[:, self.targ_col])
                    self.lasso_stats_dict[beta][alpha]['Converged'] = True
                except ConvergenceWarning:
                    warnings.filterwarnings("ignore")
                    lasso_mod.fit(train_data[:, self.pred_cols], train_data[:, self.targ_col])
                    self.lasso_stats_dict[beta][alpha]['Converged'] = False
                self.lasso_stats_dict[beta][alpha]['Training_R2'] = lasso_mod.score(train_data[:, self.pred_cols], train_data[:, self.targ_col])
                self.lasso_stats_dict[beta][alpha]['Testing_R2'] = lasso_mod.score(test_data[:, self.pred_cols], test_data[:, self.targ_col])
                self.lasso_stats_dict[beta][alpha]['model'] = lasso_mod
        warnings.filterwarnings("default")

    def train_linear_model(self):
        for beta in self.dynasigdf.beta_values:
            train_data = self.dynasigdf.get_data_array(self.train_ids, beta)
            test_data = self.dynasigdf.get_data_array(self.test_ids, beta)
            train_data, test_data, standardization = standardize_data(train_data, test_data, self.pred_cols)
            self.standardization_dict[beta] = standardization
            all_data = np.concatenate((train_data, test_data))
            mod = LinearRegression()
            mod.fit(all_data[:, self.pred_cols], all_data[:, self.targ_col])
            print(beta, adjusted_r2(mod.score(all_data[:, self.pred_cols], all_data[:, self.targ_col]), len(all_data), all_data.shape[1]-2))

    def train_test_mlp(self):
        layer_sizes = []
        for i in range(self.n_layers):
            layer_sizes.append(self.layer_size)
        layer_sizes = tuple(layer_sizes)
        for beta in self.dynasigdf.beta_values:
            train_data = self.dynasigdf.get_data_array(self.train_ids, beta)
            test_data = self.dynasigdf.get_data_array(self.test_ids, beta)
            train_data, test_data, standardization = standardize_data(train_data, test_data, self.pred_cols)
            self.standardization_dict[beta] = standardization
            self.mlp_stats_dict[beta] = dict()
            mlp_mod = MLPRegressor(layer_sizes, max_iter=self.max_iter_mlp)
            warnings.filterwarnings("error")
            try:
                mlp_mod.fit(train_data[:, self.pred_cols], train_data[:, self.targ_col])
                self.mlp_stats_dict[beta]['Converged'] = True
            except ConvergenceWarning:
                warnings.filterwarnings("ignore")
                mlp_mod.fit(train_data[:, self.pred_cols], train_data[:, self.targ_col])
                self.mlp_stats_dict[beta]['Converged'] = False
            self.mlp_stats_dict[beta]['Training_R2'] = mlp_mod.score(train_data[:, self.pred_cols], train_data[:, self.targ_col])
            self.mlp_stats_dict[beta]['Testing_R2'] = mlp_mod.score(test_data[:, self.pred_cols], test_data[:, self.targ_col])
            self.mlp_stats_dict[beta]['model'] = mlp_mod
        warnings.filterwarnings("default")

    def get_best_params_lasso(self):
        best_training = float('-Inf')
        best_testing = float('-Inf')
        best_params_training = [None, None, None]
        best_params_testing = [None, None, None]
        train_best_test = None
        best_test_model = None
        for beta in self.lasso_stats_dict:
            for alpha in self.lasso_stats_dict[beta]:
                train_r2 = self.lasso_stats_dict[beta][alpha]['Training_R2']
                test_r2 = self.lasso_stats_dict[beta][alpha]['Testing_R2']
                if train_r2 > best_training:
                    best_training = train_r2
                    best_params_training = [beta, alpha, self.lasso_stats_dict[beta][alpha]['Converged']]
                if test_r2 > best_testing:
                    best_testing = test_r2
                    best_params_testing = [beta, alpha, self.lasso_stats_dict[beta][alpha]['Converged']]
                    train_best_test = train_r2
                    best_test_model = self.lasso_stats_dict[beta][alpha]['model']
        return best_training, best_params_training, best_testing, best_params_testing, train_best_test, best_test_model

    def get_best_params_mlp(self):
        best_training = float('-Inf')
        best_testing = float('-Inf')
        best_beta_training = None
        best_beta_testing = None
        train_best_test = None
        best_test_model = None
        for beta in self.mlp_stats_dict:
            train_r2 = self.mlp_stats_dict[beta]['Training_R2']
            test_r2 = self.mlp_stats_dict[beta]['Testing_R2']
            if train_r2 > best_training:
                best_training = train_r2
                best_beta_training = beta
            if test_r2 > best_testing:
                best_testing = test_r2
                best_beta_testing = beta
                train_best_test = train_r2
                best_test_model = self.mlp_stats_dict[beta]['model']
        return best_training, best_beta_training, best_testing, best_beta_testing, train_best_test, best_test_model

    def performance_report(self):
        if self.lasso_stats_dict is not None:
            print("LASSO regression performance:")
            print("\n".join(get_performance_report_lasso(*self.get_best_params_lasso())))
        if self.mlp_stats_dict is not None:
            print("MLP regressor performance:")
            print("\n".join(get_performance_report_mlp(*self.get_best_params_mlp())))

    def get_best_beta_values(self):
        best_betas = []
        if self.lasso_stats_dict is not None:
            best_betas.append(self.get_best_beta_lasso())
        if self.mlp_stats_dict is not None:
            beta_mlp = self.get_best_beta_mlp()
            if beta_mlp != best_betas[0]:
                best_betas.append(beta_mlp)
        return best_betas

    def get_best_beta_lasso(self):
        return self.get_best_params_lasso()[3][0]

    def get_best_beta_mlp(self):
        return self.get_best_params_mlp()[3]



    def make_graphs(self, folder):
        if not os.path.isdir(folder):
            os.mkdir(folder)
        if self.lasso_stats_dict is not None:
            self._make_lasso_graphs(folder)
        if self.mlp_stats_dict is not None:
            self._make_mlp_graphs(folder)

    def _make_mlp_graphs(self, folder):
        best_training, best_beta_training, best_testing, best_beta_testing, train_best_test, best_test_model = \
        self.get_best_params_mlp()
        betas = []
        test_r2 = []
        for beta in self.mlp_stats_dict:
            betas.append(np.log(beta))
            test_r2.append(self.mlp_stats_dict[beta]['Testing_R2'])
        plt.clf()
        plt.plot(betas, test_r2)
        plt.ylim(ymin=0, ymax=1)
        plt.xlabel("Log beta")
        plt.ylabel("Predictive R²")
        plt.savefig("{}/mlp_testing_r2_beta.png".format(folder))

        test_data = self.dynasigdf.get_data_array(self.test_ids, best_beta_testing)
        train_data = self.dynasigdf.get_data_array(self.train_ids, best_beta_testing)
        train_data, test_data, standardization = standardize_data(train_data, test_data, self.pred_cols)
        self.standardization_dict[best_beta_testing] = standardization
        preds = best_test_model.predict(test_data[:, self.pred_cols])
        reals = test_data[:, self.targ_col]
        plt.clf()
        plt.scatter(preds, reals, alpha=0.5)
        plt.xlabel("Predicted value")
        plt.ylabel("Experimental value")
        text = "R² = {:.2f}\nEF10% = {:.2f}".format(best_testing, _get_ef(reals, preds))
        plt.gca().text(0.05, 0.95, text, transform=plt.gca().transAxes,
                       fontsize=14, verticalalignment='top')
        plt.savefig("{}/mlp_testing_performance.png".format(folder))
        indices = np.argsort(-preds)
        with open("{}/mlp_testing_performance.df".format(folder), "w") as f:
            f.write("variant predicted measured\n")
            for i in indices:
                f.write("{} {} {}\n".format(self.test_ids[i], preds[i], reals[i]))

    def _make_lasso_graphs(self, folder):
        best_training, best_params_training, best_testing, best_params_testing, train_best_test, best_test_model = \
        self.get_best_params_lasso()
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
        plt.xlabel("Predicted value")
        plt.ylabel("Experimental value")
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

    def predict_lasso(self, data_array):
        assert len(data_array[0]) == len(self.pred_cols)
        best_params = self.get_best_params_lasso()
        best_lasso_model = best_params[-1]
        best_beta = best_params[3][0]
        standardization = self.standardization_dict[best_beta]
        assert len(standardization) == len(self.pred_cols)
        for i in range(data_array.shape[1]):
            mean, sd = standardization[i]
            data_array[:, i] -= mean
            if sd > 0:
                data_array[:, i] /= sd
        return best_lasso_model.predict(data_array)

    def predict_mlp(self, data_array):
        assert len(data_array[0]) == len(self.pred_cols)
        best_params = self.get_best_params_mlp()
        best_mlp_model = best_params[-1]
        best_beta = best_params[3]
        standardization = self.standardization_dict[best_beta]
        assert len(standardization) == len(self.pred_cols)
        for i in range(data_array.shape[1]):
            mean, sd = standardization[i]
            data_array[:, i] -= mean
            if sd > 0:
                data_array[:, i] /= sd
        return best_mlp_model.predict(data_array)

    def map_coefficients(self, wt_pdb_file, output_pdb_file):
        enc = ENCoM(wt_pdb_file, solve=False)
        n = len(enc.mol.masses)
        lasso_mod = self.get_best_params_lasso()[-1]
        coefs = lasso_mod.coef_[-n:]
        coefs = np.array(coefs)
        coefs /= np.max(np.abs(coefs))
        enc.set_bfactors(np.abs(coefs))
        enc.set_occupancy(coefs)
        enc._write_to_file(output_pdb_file)




def get_performance_report_lasso(best_training, best_params_training, best_testing, best_params_testing,
                                 train_best_test, best_test_model):
    report = ["Best training R²: {:.2f} with alpha={} and beta={}".format(best_training, best_params_training[1],
                                                                              best_params_training[0])]
    report.append("Best testing (predictive) R²: {:.2f} with alpha={} and beta={} (associated training R²:{:.2f})"
                  .format(best_testing, best_params_testing[1], best_params_testing[0], train_best_test))
    return report


def get_performance_report_mlp(best_training, best_beta_training, best_testing, best_beta_testing, train_best_test,
                           best_test_model):
    report = ["Best training R²: {:.2f} with beta={}".format(best_training, best_beta_training)]
    report.append("Best testing (predictive) R²: {:.2f} with beta={} (associated training R²:{:.2f})".format(
                  best_testing, best_beta_testing, train_best_test))
    return report


def standardize_data(train_data, test_data, cols):
    standardization = np.zeros((len(cols), 2))
    count = 0
    for col in cols:
        mean = np.mean(np.concatenate((train_data[:, col], test_data[:, col])))
        sd = np.std(np.concatenate((train_data[:, col], test_data[:, col])))
        standardization[count] = [mean, sd]
        for data in [train_data, test_data]:
            data[:, col] -= mean
            if sd != 0:
                data[:, col] /= sd
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





