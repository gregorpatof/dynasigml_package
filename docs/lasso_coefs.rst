Mapping LASSO coefficients on the structure
===========================================

The advantage of the LASSO model is the possibility to map the learned coefficients back on the studied biomolecule's
structure in order to gain insights about what dynamical features can explain its function. Starting from the
WT PDB file, the coefficients can be mapped to the B-factors and occupancy columns, which can then be used in
PyMOL to visualize the coefficients mapped in 3D::

    from dynasigml.dynasig_ml_model import DynaSigML_Model, load_dynasigml_model_from_file

    dsml_model = load_dynasigml_model_from_file("dsml_model.pickle")
    dsml_model.map_coefficients("mutants_modeller/4bz3_WT.pdb", "coefficients.pdb")

In order to visualize the coefficients in 3D, the new PDB file needs to be opened in PyMOL and these two commands
need to be entered::

    spectrum q, blue_white_red, minimum=-1, maximum=1
    cartoon putty

This will show the backbone of the protein with varying diameter corresponding to the absolute value of the
coefficient. The color will be blue for negative coefficients, white for zero coefficients and red for positive
coefficients.