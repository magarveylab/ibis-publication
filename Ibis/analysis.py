from Ibis import (
    Prodigal,
    ProteinEmbedder,
    ProteinDecoder,
    SecondaryMetabolismPredictor,
    DomainPredictor,
    DomainEmbedder,
    DomainDecoder,
    PropeptidePredictor,
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
    ec_pred_filenames = ProteinDecoder.decode_from_embedding_fps(
        filenames=protein_embedding_filenames,
        output_dir=output_dir,
        decode_fn=ProteinDecoder.decode_ec,
        decode_name="ec",
    )
    # compute ko predictions
    ko_pred_filenames = ProteinDecoder.decode_from_embedding_fps(
        filenames=protein_embedding_filenames,
        output_dir=output_dir,
        decode_fn=ProteinDecoder.decode_ko,
        decode_name="ko",
    )
    ######################################################
    # **TO DO** compute primary metabolism predictions
    ######################################################
    # compute bgc boundaries
    bgc_filenames = SecondaryMetabolismPredictor.run_on_embedding_fps(
        filenames=protein_embedding_filenames,
        output_dir=output_dir,
        gpu_id=gpu_id,
    )
    # compute gene family predictions
    gene_family_pred_filenames = ProteinDecoder.decode_from_bgc_filenames(
        filenames=bgc_filenames,
        output_dir=output_dir,
        decode_fn=ProteinDecoder.decode_gene_family,
        decode_name="gene_family",
    )
    # compute gene predictions
    gene_pred_filenames = ProteinDecoder.decode_from_bgc_filenames(
        filenames=bgc_filenames,
        output_dir=output_dir,
        decode_fn=ProteinDecoder.decode_gene,
        decode_name="gene",
    )
    # compute molecule predictions (ripps and bacteriocins)
    mol_pred_filenames = ProteinDecoder.decode_from_bgc_filenames(
        filenames=bgc_filenames,
        output_dir=output_dir,
        decode_fn=ProteinDecoder.decode_molecule,
        decode_name="molecule",
    )
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
    # compute adenylation predictions
    adenylation_pred_filenames = DomainDecoder.decode_from_embedding_fps(
        filenames=domain_embedding_filenames,
        output_dir=output_dir,
        decode_fn=DomainDecoder.decode_adenylation,
        target_domain="A",
    )
    # compute acyltransferase predictions
    acyltransferase_pred_filenames = DomainDecoder.decode_from_embedding_fps(
        filenames=domain_embedding_filenames,
        output_dir=output_dir,
        decode_fn=DomainDecoder.decode_acyltransferase,
        target_domain="AT",
    )
    # compute ketosynthase domain functional predictions
    ketosynthase_pred_filenames = DomainDecoder.decode_from_embedding_fps(
        filenames=domain_embedding_filenames,
        output_dir=output_dir,
        decode_fn=DomainDecoder.decode_ketosynthase,
        target_domain="KS",
    )
    # compute ketoreductase domain functional predictions
    ketoreductase_pred_filenames = DomainDecoder.decode_from_embedding_fps(
        filenames=domain_embedding_filenames,
        output_dir=output_dir,
        decode_fn=DomainDecoder.decode_ketoreductase,
        target_domain="KR",
    )
    # compute dehydratase domain functional predictions
    dehydratase_pred_filenames = DomainDecoder.decode_from_embedding_fps(
        filenames=domain_embedding_filenames,
        output_dir=output_dir,
        decode_fn=DomainDecoder.decode_dehydratase,
        target_domain="DH",
    )
    # compute enoylreductase domain functional predictions
    enoylreductase_pred_filenames = DomainDecoder.decode_from_embedding_fps(
        filenames=domain_embedding_filenames,
        output_dir=output_dir,
        decode_fn=DomainDecoder.decode_enoylreductase,
        target_domain="ER",
    )
    # compute thiolation domain subclass predictions
    thiolation_pred_filenames = DomainDecoder.decode_from_embedding_fps(
        filenames=domain_embedding_filenames,
        output_dir=output_dir,
        decode_fn=DomainDecoder.decode_thiolation,
        target_domain="T",
    )
    # compute propeptide predictions
    propeptide_pred_filenames = PropeptidePredictor.run_on_mol_pred_fps(
        filenames=mol_pred_filenames,
        output_dir=output_dir,
        gpu_id=gpu_id,
        cpu_cores=cpu_cores,
    )
    # compute metabolism embeddings
