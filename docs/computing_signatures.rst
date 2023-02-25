Computing Dynamical Signatures
==============================

To compute the ENCoM Dynamical Signatures from the list of mutated PDB files, we need to build a DynaSigDF
("Dynamical Signature Data Frame") object. The DynaSigDF constructor needs the list of files, and a matching
list of experimental measures (what the machine learning models will try to predict later on). For the miR-125a
dataset, the maturatino efficiency measure along with the MC-Fold enthalpy of folding are stored in the
**data_mir125.df** file. ::

    from dynasigml.dynasig_df import DynaSigDF
    import glob
    import json

    def load_data(filename):
    with open(filename) as f:
        lines = f.readlines()
    data_dict = dict()
    for line in lines[1:]:
        ll = line.split()
        data_dict[ll[2]] = [float(ll[0]), float(ll[1])]
    return data_dict


    files_list = glob.glob('mir125a_variants/*.pdb')
    beta_values = [np.e ** (x / 2) for x in range(-6, 7)]
    data_dict = load_data('data_mir125.df')
    exp_data = []
    for fn in files_list:
        mutid = fn.split('.')[0].split('mir125a_')[-1]
        exp_data.append(data_dict[mutid])
    # eff is for maturation efficiency
    DynaSigDF(files_list, exp_data, ["eff", "mcfold_energy"], "combined_dsdf".format(index), beta_values=beta_values)


.. note::

    The above code will take between 24 and 48 hours to run on a modern laptop (without parallelization). However, you do
    not need to run it as the saved DynaSigDF objects that contain every Dynamical Signature are part of the
    dynasigml_mir125a_example repository. For more details on how to run the computations yourself, including parallelization,
    see the Running in parallel section.
