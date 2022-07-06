Generating mutations
====================

The first step in order to run DynaSig-ML is to build a collection of mutated structures, one for every
variant in the experimental dataset. The structure of VIM-2 lactamase was solved by X-ray crystallography
(PDB ID 4bz3) and is the starting point in this example. We made every mutation from position 31 to 262 inclusively
(the other parts are not part of the solved structure) that had a fitness measurement from the DMS study using
the MODELLER mutate_model.py script (https://salilab.org/modeller/wiki/Mutate%20model ). These 4343 structures,
along with the cleaned WT structure, are already present in the **mutants_modeller** folder from the
**dynasigml_vim2_example** repository.

For other applications, many tools can be used in the generation of point mutations. In general, we recommend
that the backbone should change the least possible between the WT and mutated structures. The reason for this
is that the effect of the mutation will be better captured by ENCoM, since it will only affect the sidechains
and thus only the long-range interaction potential.
