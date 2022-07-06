Introduction
============

:ref:`how_to_cite`

DynaSig-ML ("Dynamical Signatures - Machine Learning") is a Python
package allowing the easy study of dynamics-function relationships in biomolecules.

DynaSig-ML automatically computes and
stores Dynamical Signatures of sequence variants using ENCoM (Elastic Network Contact Model), a sequence-sensitive
coarse-grained normal mode analysis model. These Dynamical Signatures, along with optional additional data (for example
the ddG of folding of the mutation), are then used to automatically train two types of machine learning models: LASSO
multilinear regression and multilayer perceptrons. The LASSO coefficients are automatically mapped back on the biomolecules'
structure and can be easily visualized using PyMOL, leading to biological insights.

The guide provides examples using
deep mutational scan data about the enzymatic efficiency of VIM-2 lactamase (CITATION), but the method is generalizable
to any biomolecule on which mutational data exists, and for which an input structure is known or can be predicted with confidence.
