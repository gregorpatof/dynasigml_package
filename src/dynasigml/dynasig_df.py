import numpy as np
import pickle


def load_pickled_dynasig_df(pickle_file):
    with open(pickle_file, 'rb') as f:
        dsdf = pickle.load(f)
    return dsdf


def combine_pickled_dynasig_dfs(pickle_files_list, output_name):
    assert isinstance(pickle_files_list, list)
    assert len(pickle_files_list) >= 2
    combined_dsdf = load_pickled_dynasig_df(pickle_files_list[0])
    combined_dsdf.outname = output_name
    for pickle_file in pickle_files_list[1:]:
        dsdf = load_pickled_dynasig_df(pickle_file)
        combined_dsdf.add_other_dynasig_df(dsdf)
    combined_dsdf.save_to_file()
    return combined_dsdf


class DynaSigDF:
    """Class representing a set of Dynamical Signatures (DynaSigs), which can be outputted as a data frame.

    The DynasigDF class allows for the parallel computation of DynaSigs, stores them with minimal disk space using NumPy
    .npy binary files, and implements methods to easily combine the separate DynaSigDFs into one big dataframe for
    further analysis.

    Attributes:
        files_list (list): The list of sequence variants PDB files for which to compute the Dynamical Signatures.
        exp_measures (list): A list of experimental measures, in the same order as files_list. There can be more than
                             one measure per variant (in which case exp_measures is a list of lists).
        exp_labels (list): Labels for the experimental measures, in the order they appear in each sub_list. If there is
                           only one measure per variant, type can be str.
        n_exp (int): The number of experimental measures per variant.
        outname (str): The output name where the DynaSigDF object will be saved (with the .pickle extension).
        id_func (function): The function that gives the variant identifier from the PDB filename. By default, it gives
                            the name of the file without the path to the directory and without the extension.
        beta_values (list): The list of beta values for the Entropic Signatures computed. By default, beta=1 is the
                            only value tested.
        models (list): The list of ENM models to run. By default, only ENCoM is run.
        models_labels (list): The labels for the ENM models.
        added_atypes_list (list): Can be used to add custom atom types definitions. Needs to be of the
                                  same length as 'files_list' (some elements can be None).
        added_massdef_list (list): Same as 'added_atypes_list', for mass definition files.
        use_svib (bool): If True, the vibrational entropy will be computed and added as a potential
                         predictor variable. Defaults to False because usually the Entropic Signature is
                         enough for the model to capture the vibrational entropy.
        index_dict (dict): Dictionary of parameter combinations for every row index in the data frame.
        params_dict (dict): Dictionary of data frame row indices for every combination of parameters.
        dynasigs_masslabels (list): List of labels for the masses in the DynaSig. Every PDB file must generate the
                                    same labels.
        data_array (ndarray): Storage of all the data as a data frame (each observation on a row), using a NumPy
                              2D array.
    """

    def __init__(self, files_list, exp_measures, exp_labels, output_name, id_func=None, beta_values=None, models=None,
                 models_labels=None, added_atypes_list=None, added_massdef_list=None, use_svib=False):
        """Constructor for the DynaSigDF class.

        Args:
            files_list (list): The list of PDB files on which to compute DynaSigs.
            exp_measures (list): List of experimental measures, matching the files_list. If many experimental measures
                                 are used, has to be a list of lists: [[measure1_file1, measure2_file1],
                                 [measure1_file2, measure2_file2], ...]
            exp_labels (list): The labels for each of the experimental measures. If only one measure is supplied, can
                               be a string instead of a list.
            output_name (str): the output name to use for the saved DynaSigDF object.
            id_func (function, optional): function to be ran on the input files to generate the filename IDs (which are
                                          used to identify each DynaSig in the final dataframe). If not supplied,
                                          defaults to just the filename at the end of the path, minus the extension.
            beta_values (list, optional): If supplied, the list of beta values for the computation of DynaSigs. Positive
                                          values will use the Entropic Signature, any negative value will compute the
                                          MSF (predicted B-factors).
            models (list, optional): If supplied, list of ENM models (need to inherit from the ENM metaclass from the
                                     NRGTEN package. Defaults to ENCoM.
            models_labels (list, optional): List of model labels (to identify the model used in the dataframe).
            added_atypes_list (list, optional): Can be used to add custom atom types definitions. Needs to be of the
                                                same length as 'files_list' (some elements can be None).
            added_massdef_list (list, optional): Same as 'added_atypes_list', for mass definition files.
            use_svib (bool, optional): If True, the vibrational entropy will be computed and added as a potential
                                       predictor variable. Defaults to False because usually the Entropic Signature is
                                       enough for the model to capture the vibrational entropy.
        """
        if models is None:
            from nrgten.encom import ENCoM
            models = [ENCoM]
            models_labels = ["ENCoM"]
        if beta_values is None:
            beta_values = [1]
        if id_func is None:
            id_func = lambda x: x.split('/')[-1]
        assert isinstance(beta_values, list) and isinstance(models, list) and isinstance(models_labels, list)
        assert len(models) == len(models_labels)
        assert len(files_list) == len(exp_measures)
        if isinstance(exp_measures[0], list):
            assert isinstance(exp_labels, list)
            assert len(exp_labels) == len(exp_measures[0])
            self.n_exp = len(exp_measures[0])
        else:
            self.n_exp = 1
        if added_atypes_list is not None:
            assert len(added_atypes_list) == len(files_list) == len(added_massdef_list)
        self.files_list = files_list
        self.exp_measures = exp_measures
        self.exp_labels = exp_labels
        self.outname = output_name
        self.models = models
        self.models_labels = models_labels
        self.beta_values = beta_values
        self.id_func = id_func
        self.added_atypes_list = added_atypes_list
        self.added_massdef_list = added_massdef_list
        self.use_svib = use_svib
        self.index_dict = dict()
        self.params_dict = dict()
        self._fill_index_params_dicts()
        self.data_array = None
        self.dynasigs_masslabels = None
        self._compute_dynasig_df()
        self._pickle()

    def _pickle(self):
        self.id_func = None
        self.models = [None for x in self.models]
        with open('{}.pickle'.format(self.outname), 'wb') as f:
            pickle.dump(self, f)

    def rename(self, new_name):
        self.outname = new_name

    def save_to_file(self):
        self._pickle()

    def _fill_index_params_dicts(self):
        index = 0
        for filename_id in [self.id_func(x) for x in self.files_list]:
            self.params_dict[filename_id] = dict()
            for mod_lab in self.models_labels:
                self.params_dict[filename_id][mod_lab] = dict()
                for beta in self.beta_values:
                    self.params_dict[filename_id][mod_lab][beta] = index
                    self.index_dict[index] = [filename_id, mod_lab, beta]
                    index += 1

    def get_file_ids(self):
        return [x for x in self.params_dict]

    def get_data_array(self, file_ids, beta, mod_lab=None):
        if mod_lab is None:
            assert len(self.models_labels) == 1
            mod_lab = self.models_labels[0]
        n_add = 1  # beta value
        if self.use_svib:
            n_add += 1
        n_add += self.n_exp
        new_array = np.zeros((len(file_ids), self.data_array.shape[1]))
        for i, file_id in enumerate(file_ids):
            index = self.params_dict[file_id][mod_lab][beta]
            new_array[i] = self.data_array[index]
        return new_array

    def get_column_index(self, exp_label):
        for i, lab in enumerate(self.exp_labels):
            if lab == exp_label:
                return 1 + i

    def get_svib_index(self):
        if not self.use_svib:
            raise ValueError("Called get_svib_index() on DynaSigDF with use_svib=False")
        return 1 + self.n_exp

    def get_dynasig_indices(self):
        if self.use_svib:
            start = 2 + self.n_exp
        else:
            start = 1 + self.n_exp
        return [x for x in range(start, start + len(self.dynasigs_masslabels))]

    def _compute_dynasig_df(self):
        first_flag = True
        counter = 0
        for i, filename in enumerate(self.files_list):
            filename_id = self.id_func(filename)
            for mod, mod_lab in zip(self.models, self.models_labels):
                if self.added_atypes_list is None:
                    enm = mod(filename)
                else:
                    # This allows to have unique custom residue(s) (ligand for example) per file
                    enm = mod(filename, added_atypes=self.added_atypes_list[i],
                              added_massdef=self.added_massdef_list[i])

                # only the mass name to allow for mutations
                masslabels = [x.split('.')[-1] for x in enm.get_mass_labels()]
                if first_flag:
                    n_add = 1 # beta value
                    if self.use_svib:
                        n_add += 1
                    n_add += self.n_exp
                    self.data_array = np.zeros((np.max([x+1 for x in self.index_dict]), len(masslabels)+n_add))
                    first_flag = False
                for beta_val in self.beta_values:
                    assert counter == self.params_dict[filename_id][mod_lab][beta_val]

                    if self.dynasigs_masslabels is None:
                        self.dynasigs_masslabels = masslabels
                    else:
                        _validate_masslabels(masslabels, self.dynasigs_masslabels)

                    self.data_array[counter][0] = beta_val
                    if isinstance(self.exp_measures[i], list):
                        for j in range(self.n_exp):
                            self.data_array[counter][1+j] = self.exp_measures[i][j]
                    else:
                        self.data_array[counter][1] = self.exp_measures[i]
                    if self.use_svib:
                        self.data_array[counter][1+self.n_exp] = enm.compute_vib_entropy(beta=beta_val)
                        self.data_array[counter][2+self.n_exp:] = enm.compute_bfactors_boltzmann(beta=beta_val)
                    else:
                        self.data_array[counter][1+self.n_exp:] = enm.compute_bfactors_boltzmann(beta=beta_val)
                    counter += 1

    def add_other_dynasig_df(self, other):
        if not self._properties_are_matching(other):
            raise ValueError("Trying to combine two DynaSigDF objects with unmatched properties: {} {}".format(
                             self.outname, other.outname))
        n = np.max([x+1 for x in self.index_dict])

        for other_index in other.index_dict:
            self.index_dict[other_index + n] = other.index_dict[other_index].copy()
        for other_file_id in other.params_dict:
            if other_file_id in self.params_dict:
                raise ValueError("This file ID present in both DynaSigDFs: {}".format(other_file_id))
            self.params_dict[other_file_id] = other.params_dict[other_file_id].copy()
            for mod_lab in self.params_dict[other_file_id]:
                for beta in self.params_dict[other_file_id][mod_lab]:
                    self.params_dict[other_file_id][mod_lab][beta] += n
        self.data_array = np.concatenate((self.data_array, other.data_array))
        self.files_list = self.files_list + other.files_list
        self.exp_measures = self.exp_measures + other.exp_measures

    def _properties_are_matching(self, other):
        assert isinstance(other, DynaSigDF)
        matching = True
        if self.n_exp != other.n_exp:
            matching = False
        if self.beta_values != other.beta_values:
            matching = False
        if self.models_labels != other.models_labels:
            matching = False
        if self.use_svib != other.use_svib:
            matching = False
        return matching


def _validate_masslabels(new_labels, old_labels):
    assert len(new_labels) == len(old_labels)
    for new, old in zip(new_labels, old_labels):
        if new != old:
            raise ValueError("problem with masslabels in compute_dynasigs_df: {} not equal to {}".format(new, old))
