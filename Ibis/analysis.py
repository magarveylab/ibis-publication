from Ibis import (
    Prodigal,
    ProteinEmbedder,
    ProteinDecoder,
    SecondaryMetabolismPredictor,
    DomainPredictor,
    DomainEmbedder,
)
import json
import os
from typing import List, Union


def setup_working_directories(filenames: List[str], output_dir: str):
    for fp in filenames:
        basename = os.basename(fp)
        os.makedirs(f"{output_dir}/{basename}", exist_ok=True)


def run_ibis_on_genome(
    nuc_fasta_filenames: List[str],
    output_dir: str,
    gpu_id: int = 0,
    cpu_cores: int = 1,
):
    # this function will be used to model airflow pipeline
    # setup working directories
    setup_working_directories(
        filenames=nuc_fasta_filenames, output_dir=output_dir
    )
    # prodigal prediction
    prodigal_filenames = Prodigal.parallel_run_on_nuc_fasta_fps(
        filenames=fasta_filenames, output_dir=output_dir, cpu_cores=cpu_cores
    )
    # compute protein embeddings
    protein_embedding_filenames = ProteinEmbedder.run_on_prodigal_fps(
        filenames=prodigal_filenames, output_dir=output_dir, gpu_id=gpu_id
    )
    # compute ec predictions
    ec_pred_filenames = ProteinDecoder.decode_ec_from_embedding_fps(
        filenames=protein_embedding_filenames, output_dir=output_dir
    )
    # compute ko predictions
    ko_pred_filenames = ProteinDecoder.decode_ko_from_embedding_fps(
        filenames=protein_embedding_filenames, output_dir=output_dir
    )
    ###
    # **TO DO** compute primary metabolism predictions
    ###
    # compute bgc boundaries
    bgc_filenames = SecondaryMetabolismPredictor.run_on_embedding_fps(
        filenames=protein_embedding_filenames,
        output_dir=output_dir,
        gpu_id=gpu_id,
    )
    # compute gene family predictions

    # compute gene predictions

    # compute domain predictions
    domain_pred_filenames = DomainPredictor.run_on_bgc_fps(
        filenames=bgc_filenames,
        output_dir=output_dir,
        gpu_id=gpu_id,
        cpu_cores=cpu_cores,
    )
    # compute domain embeddings
    domain_embedding_filenames = DomainEmbedder.run_on_domain_pred_fps(
        filenames=domain_pred_filenames, output_dir=output_dir, gpu_id=gpu_id
    )
    # compute a domain substrate predictions

    # compute at domain substrate predictions

    # compute ks domain functional predictions

    # compute kr domain functional predictions

    # compute dh domain functional predictions

    # compute er domain functional predictions

    # compute t domain subclass predictions

    # compute propeptide predictions

    # compute metabolism embeddings
