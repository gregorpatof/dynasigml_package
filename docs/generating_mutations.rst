Generating mutations
====================

The first step in order to run DynaSig-ML is to build a collection of mutated structures, one for every
variant in the experimental dataset. The predicted structure of WT miR-125a, using the MC-Fold | MC-Sym pipeline, is the
starting point in this example.
We made every of the 29 478 variants which adopt the WT 2D structure using ModeRNA. These structures,
along with the predicted WT structure, are already present in the **mir125a_variants** folder from the
**dynasigml_mir125a_example** repository.

For other applications, many tools can be used in the generation of point mutations. In general, we recommend
that the backbone should change the least possible between the WT and mutated structures. The reason for this
is that the effect of the mutation will be better captured by ENCoM, since it will only affect the sidechains
and thus only the long-range interaction potential.
