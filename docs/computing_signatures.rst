Computing Dynamical Signatures
==============================

To compute the ENCoM Dynamical Signatures from the list of mutated PDB files, we need to build a DynaSigDF
("Dynamical Signature Data Frame") object. The DynaSigDF constructor needs the list of files, and a matching
list of experimental measures (what the machine learning models with try to predict later on). For the VIM-2
dataset, the fitness measure along with other properties extracted from the supplementary data are stored as
a Python dictionary in json format in the **vim2_data_dict.json** file, with the corresponding labels in the
**vim2_exp_labels.json** file. ::

    from dynasigml.dynasig_df import DynaSigDF
    import glob
    import json

    filenames_list = glob.glob("mutants_modeller/*.pdb")
    with open("vim2_data_dict.json") as f:
        data_dict = json.load(f)
    with open("vim2_exp_labels.json") as f:
        exp_labels = json.load(f)
    data_list = []
    for filename in filenames_list:
        variant_id = filename.split('/')[-1].split('.')[0].split('_')[-1]
        data_list.append(data_dict[variant_id])
    dsdf = DynaSigDF(filenames_list, data_list, exp_labels, "full_dsdf_vim2")

.. note::

    The above code will take quite a while to run, somewhere around 4 hours on a modern machine. However, you do
    not need to run it as the saved DynaSigDF objects that contain every Dynamical Signature are part of the
    dynasigml_vim2_example repository. For more details on how to run the computations yourself, including parallelization,
    see the Running in parallel section.
