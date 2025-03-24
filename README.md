# ibis-publication
Integrated Biosynthetic Inference Suite (IBIS) \
Authors: Mathusan Gunabalasingam and Norman Spencer

## Citation
If you use this code or models in your research, please cite our work:

```
@article{yourpaper2025,
  author    = {Your Name and Co-Authors},
  title     = {Title of Your Paper},
  journal   = {Journal Name},
  volume    = {XX},
  number    = {X},
  pages     = {XX--XX},
  year      = {2025},
  doi       = {10.XXXX/XXXXXX}
}
```

## Installation

### Inference-Only Installation
1. Install the Package via Pip Symlinks:
    - Create and activate the Conda environment, then install the package in editable mode:
```
    conda env create -f environment.yml
    conda activate ibispub_test
    pip install -e .
```
2. Set Up Models:
    - Download Models.zip from [Zenodo (10.5281/zenodo.14246984)](https://zenodo.org/doi/10.5281/zenodo.14246984).
    - Replace the contents of the `Ibis/Models` directory with the extracted files.
3. Set Up Qdrant
    - Install Qdrant and restore the Qdrant reference databases from the provided snapshots. Look under **Qdrant Setup** for more details.

### Training Installation
If you plan to fine-tune or retrain the models used in this work, install the following supplementary packages, which implement our custom Multi-Task Multi-Dataset Training approach:
1. [predacons](https://github.com/magarveylab/ibis-transformer-training.git) enables Transformer-based training.
2. [omnicons](https://github.com/magarveylab/ibis-transformer-training.git) supports Graphormer-based training.

#### Installation Notes

Please be aware that the Conda environment provided with this installation contains core dependencies for PyTorch (including lightning, geometric, etc.), ONNX, and torchscript compatible with CUDA 11.1. If you plan on using GPU-accelerated inference or performing training, please be aware that different CUDA versions may require you to modify the provided conda environment to be compatible with your hardware. Should you experience any issues using these dependencies, please ensure that the package versions are compatible with your hardware.

## Qdrant Setup 
IBIS inference piplelines utilize [Qdrant](https://qdrant.tech/) embedding databases for approximate nearest neighbor (ANN) lookups. Since system configurations may vary, we recommend setting up Qdrant locally using a Docker container, following the [official documentation](https://qdrant.tech/documentation/quickstart/).

The required Qdrant databases for inference are provided as QdrantSnapshots.zip in the accompanying [Zenodo repository](https://zenodo.org/doi/10.5281/zenodo.14246984)

### Restoring Qdrant Databases
To restore the Qdrant databases, ensure that the snapshot files are placed in the expected directory and run the following script:
```
# Adjust file paths or port numbers based on your local setup.
# This script assumes that snapshots are located in:
# Ibis/Utilities/Qdrant/PrepareSnapshots/snapshots/
python Ibis/Utilities/Qdrant/PrepareSnapshots/restore.py
```
Adjust paths or configurations as necessary to match your environment.

## Training
The following training scripts are included for model development and fine-tuning:
1. [IBIS-Enzyme](https://github.com/magarveylab/ibis-transformer-training/tree/main/training/ibis_enzyme)
    - A Transformer-based model designed to embed protein sequences by capturing patterns associated with enzyme commission numbers, specialized metabolism proteins, and RiPP classes.
    - Predicts residues corresponding to domains within multi-modular assembly proteins (e.g., Type I polyketide synthases, non-ribosomal peptide synthetases).
    - Identifies propeptide regions in RiPPs.
2. [IBIS-Domain](https://github.com/magarveylab/ibis-transformer-training/tree/main/training/ibis_domain)
    - A Transformer-based model that embeds protein domains for substrate prediction in adenylation and acyltransferase domains.
    - Differentiates between functional and inactive states in ketosynthase, ketoreductase, enoylreductase, and dehydratase domains.
    - Encodes domain subclasses, such as thiolation domains involved in β-branching cascades within polyketide biosynthesis.
3. [IBIS-SM](https://github.com/magarveylab/ibis-graphormer-training/tree/main/training/ibis_sm)
    - A Graphormer-based model designed to predict biosynthetic gene cluster (BGC) boundaries.
4. [IBIS-BGC](https://github.com/magarveylab/ibis-graphormer-training/tree/main/training/ibis_bgc)
    - A Graphormer-based model that generates BGC embeddings for fast, large-scale comparative analysis.

## Inference

### Complete Genome Annotation with IBIS
IBIS provides high-level inference functions for streamlined genome annotation. To fully annotate a genome using all IBIS components, simply run the following function:
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
Adjust gpu_id and cpu_cores based on your system configuration to optimize performance.

### Modular Genome Annotation with Individual IBIS Components

IBIS allows users to run individual modules without performing full genome annotation. For example, users may want to generate IBIS-Enzyme embeddings for all proteins and predict EC numbers without assigning primary metabolism or detecting BGCs.

Each IBIS module is self-contained, and the sequence of operations in `Ibis.Analysis.run_on_genomes` defines the necessary prerequisites for each function. The following example demonstrates a modular workflow:

```python

from Ibis.Analysis import (
    setup_working_directories,
)
from Ibis import (
    Prodigal,
    ProteinEmbedder,
    ProteinDecoder,
)

# Directory to save results to
save_dir = "/path/to/results"

# list of nucleotide fasta filenames
filenames = ["test1.fasta", "test2.fasta"]

# setup working directories
setup_working_directories(filenames=filenames, output_dir=save_dir)

prodigal_preds_created = Prodigal.parallel_run_on_files(
    filenames=filenames,
    output_dir=save_dir,
    cpu_cores=4,
)

# compute protein embeddings for all detected proteins
protein_embs_created = ProteinEmbedder.run_on_files(
    filenames=filenames,
    output_dir=save_dir,
    prodigal_preds_created=prodigal_preds_created,
    gpu_id=0,
)

# compute ec predictions for all proteins.
ec_preds_created = ProteinDecoder.run_on_files(
    filenames=filenames,
    output_dir=save_dir,
    protein_embs_created=protein_embs_created,
    decode_fn=ProteinDecoder.decode_ec,
    decode_name="ec",
)
```

This approach allows flexible use of IBIS modules based on specific research needs. Adjust parameters such as cpu_cores and gpu_id to optimize performance for your system.








### Selecting IBIS Modules for Partial Annotation

Each IBIS function returns a boolean value indicating whether it has been successfully completed. Downstream functions require confirmation that all prerequisite steps have been performed.

For example, in the previous workflow:
* EC number prediction requires protein embeddings.
* Protein embeddings require ORF predictions using Pyrodigal.

This structure allows users to efficiently determine the minimum required modules to achieve their desired level of annotation.

### Single-Threaded IBIS Module Execution

While all IBIS modules support parallel computation, single-threaded execution is also available for cases where parallelization is not needed or possible. Each module includes a single-threaded equivalent.

For example, to run Prodigal on a single nucleotide FASTA file:

```python
from Ibis.Analysis import (
    setup_working_directories,
)
from Ibis import Prodigal

nuc_fasta_fp = "/path/to/nuc_fasta/file.fasta"
Prodigal.run_on_single_file(nuc_fasta_fp, output_dir=save_dir)

```
This option ensures flexibility for different computational environments.


## Web Platform
A dedicated website for presenting processed genomes from NCBI will be launched soon. In future updates, users will be able to submit internal genomes directly through the platform.
