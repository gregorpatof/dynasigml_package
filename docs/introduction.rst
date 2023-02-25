Introduction
============

:ref:`how_to_cite`

DynaSig-ML ("Dynamical Signatures - Machine Learning") is a Python
package allowing the easy study of dynamics-function relationships in biomolecules.

DynaSig-ML automatically computes and
stores Dynamical Signatures of sequence variants using ENCoM (Elastic Network Contact Model), a sequence-sensitive
coarse-grained normal mode analysis model. These Dynamical Signatures, along with optional additional data (for example
the ddG of folding of the mutation), are then used to automatically train LASSO
multilinear regression, in addition to any number of user-specified machine learning models, provided thay are implemented
in sckikit-learn. The LASSO coefficients are automatically mapped back on the biomolecules'
structure and can be easily visualized using PyMOL, leading to biological insights.

The guide provides examples using
experimental miR-125a (a human microRNA) maturation efficiency data, as described in our previous work
(https://doi.org/10.1371/journal.pcbi.1010777). The method is generalizable
to any biomolecule on which mutational data exists, and for which an input structure is known or can be predicted with confidence.
