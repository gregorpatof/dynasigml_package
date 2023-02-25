Mapping LASSO coefficients on the structure
===========================================

The advantage of the LASSO model is the possibility to map the learned coefficients back on the studied biomolecule's
structure in order to gain insights about what dynamical features can explain its function. Starting from the
WT PDB file, the coefficients can be mapped to the B-factors and occupancy columns, which can then be used in
PyMOL to visualize the coefficients mapped in 3D::

    from dynasigml.dynasig_ml_model import DynaSigML_Model, load_dynasigml_model_from_file

    dsml_model_inverted = load_dynasigml_model_from_file("dsml_model_inverted.pickle")
    dsml_model_inverted.map_coefficients("mir125a_variants/mir125a_WT.pdb", "coefficients_inverted.pdb")

    dsml_model_hard = load_dynasigml_model_from_file("dsml_model_hard.pickle")
    dsml_model_hard.map_coefficients("mir125a_variants/mir125a_WT.pdb", "coefficients_hard.pdb")

In order to visualize the coefficients in 3D, the new PDB file needs to be opened in PyMOL and these two commands
need to be entered::

    spectrum q, blue_white_red, minimum=-1, maximum=1
    cartoon putty

This will show the backbone of the biomolecule with varying diameter corresponding to the absolute value of the
coefficient (works better on proteins than RNA). The color will be blue for negative coefficients, white for zero
coefficients and red for positive coefficients.