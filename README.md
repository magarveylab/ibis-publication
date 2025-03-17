# ibis-publication
Ibis package for external use (released with publication)
Authors: Mathusan Gunabalasingam and Norman Spencer

## Citation
If you use this code or models in your research, please cite our work:

## Installation
Install the package via pip symlinks.
```
conda env create -f environment.yml
conda activate ibispub
pip install -e .
```
Replace the contents of the `Ibis/Models` with the Models.zip file obtained from [Zenodo](10.5281/zenodo.14246984)

## Qdrant Setup 
IBIS inference piplelines make use of [Qdrant](https://qdrant.tech/) embedding databases to perform ANN lookups. Since individual system setups will vary, we recommend setting up Qdrant locally using a docker container as described in the [documentation](https://qdrant.tech/documentation/quickstart/). The Qdrant databases required for inference are provided as Qdrant snapshots (QdrantSnapshots.zip) at the accompanying [Zenodo Repository](10.5281/zenodo.14246984). Once Qdrant is set up on your local machine, you may restore the databases as follows:
```bash
# Note, it may be necessary to change file paths or port numbers in accordance with your local setup.
# This script assumes that the snapshots will be placed in a local folder Ibis/Utilities/Qdrant/PrepareSnapshots/snapshots/. Please adjust as necessary.
python Ibis/Utilities/Qdrant/PrepareSnapshots/restore.py
```

## Training
The scripts for training the following models are included:
1. [IBIS-Enzyme](https://github.com/magarveylab/ibis-transformer-training/tree/main/training/ibis_enzyme): A Transformer-based model designed to embed protein sequences by capturing patterns associated with enzyme commission numbers, proteins involved in specialized metabolism, and RiPP classes. Additionally, the model predicts residues corresponding to domains within multi-modular assembly proteins (e.g., Type I polyketide synthases, non-ribosomal peptide synthetase), as well as propeptide designation.
2. [IBIS-Domain](https://github.com/magarveylab/ibis-transformer-training/tree/main/training/ibis_domain): A Transformer-based model that embeds protein domains by identifying patterns relevant to substrate prediction for adenylation and acyltransferase domains. It also distinguishes between functional and inactive states for ketosynthase, ketoreductase, enoylreductase, and dehydratase domains. Furthermore, the model encodes domain subclasses, such as thiolation domains involved in β-branching cascades within polyketide biosynthesis.
3. [IBIS-SM]
4. [IBIS-BGC]

## Inference
High-level inference functions are provided for each module of IBIS. To completely annotate a genome with all components of IBIS, one need only run the following function:
```python
from Ibis.Analysis import run_ibis_on_genomes

# Directory to save results to
save_dir = "/path/to/results"

# list of nucleotide fasta filenames
filenames = ["test1.fasta", "test2.fasta"]

run_ibis_on_genomes(
    nuc_fasta_filenames = filenames,
    output_dir = save_dir,
    gpu_id = 0, # optional, but will significantly accelerate inference.
    cpu_cores = 1 # for parallelization of CPU-based tasks, such as Pyrodigal.
)
```
Many users may want to use individual modules of IBIS, without having to perform complete genome annotation. For example, it may be desirable to generate IBIS-Enzyme embeddings for all proteins without performing primary metabolic assignment, calling BGCs, etc. To facilitate this process, each module of IBIS is self-contained and the underlying order-of-operations of `Ibis.Analysis.run_on_genomes()` describes the necessary prerequisites for each function. A sample use-case is provided below.


## Web Platform
A dedicated website for presenting processed genomes from NCBI will be launched soon. In future updates, users will be able to submit internal genomes directly through the platform.
