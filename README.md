# ibis-publication
Ibis package for external use (released with publication)
Authors: Mathusan Gunabalasingam and Norman Spencer

## Installation
Install the package via pip symlinks.
```
conda env create -f environment.yml
conda activate ibispub
pip install -e .
```
Replace the contents of the `Ibis/Models` with the Models.zip file obtained from [Zenodo](https://zenodo.org/records/14246984)

## Training
The scripts for training the following models are included:
1. [IBIS-Enzyme](https://github.com/magarveylab/ibis-transformer-training/tree/main/training/ibis_enzyme): A Transformer-based model designed to embed protein sequences by capturing patterns associated with enzyme commission numbers, proteins involved in specialized metabolism, and RiPP classes. Additionally, the model predicts residues corresponding to domains within multi-modular assembly proteins, such as Type I polyketide synthases (T1PKS) and non-ribosomal peptide synthetases (NRPS), as well as propeptide designation.
2. [IBIS-Domain](https://github.com/magarveylab/ibis-transformer-training/tree/main/training/ibis_domain): A Transformer-based model that embeds protein domains by identifying patterns relevant to substrate prediction for adenylation and acyltransferase domains. It also distinguishes between functional and inactive states for ketosynthase, ketoreductase, enoylreductase, and dehydratase domains. Furthermore, the model encodes domain subclasses, such as thiolation domains involved in β-branching cascades within polyketide biosynthesis.
3. [IBIS-SM]
4. [IBIS-BGC]