import os
from typing import Dict, List, Optional

from Ibis import (
    DomainDecoder,
    DomainEmbedder,
    DomainPredictor,
    PrimaryMetabolismPredictor,
    Prodigal,
    PropeptidePredictor,
    ProteinDecoder,
    ProteinEmbedder,
    SecondaryMetabolismEmbedder,
    SecondaryMetabolismPredictor,
)


def setup_working_directories(filenames: List[str], output_dir: str):
    for fp in filenames:
        basename = os.path.basename(fp)
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
        filenames=nuc_fasta_filenames,
        output_dir=output_dir,
        cpu_cores=cpu_cores,
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
    # compute primary metabolism predictions
    primary_metabolism_pred_filenames = (
        PrimaryMetabolismPredictor.parallel_run_on_ko_pred_fps(
            filenames=ko_pred_filenames,
            output_dir=output_dir,
            cpu_cores=cpu_cores,
        )
    )
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
    bgc_embedding_filenames = (
        SecondaryMetabolismEmbedder.run_on_domain_embedding_fps(
            filenames=domain_embedding_filenames,
            output_dir=output_dir,
            gpu_id=gpu_id,
        )
    )


def get_filelookup(nuc_fasta_filename: str, output_dir: str) -> Dict[str, str]:
    name = os.path.basename(nuc_fasta_filename)
    filelookup = {
        "prodigal_fp": f"{output_dir}/{name}/prodigal.json",
        "protein_embedding_fp": f"{output_dir}/{name}/protein_embedding.pkl",
        "bgc_pred_fp": f"{output_dir}/{name}/bgc_predictions.json",
        "primary_metabolism_pred_fp": f"{output_dir}/{name}/primary_metabolism_predictions.json",
        "ec_pred_fp": f"{output_dir}/{name}/ec_predictions.json",
        "ko_pred_fp": f"{output_dir}/{name}/ko_predictions.json",
        "gene_family_pred_fp": f"{output_dir}/{name}/gene_family_predictions.json",
        "gene_pred_fp": f"{output_dir}/{name}/gene_predictions.json",
        "mol_pred_fp": f"{output_dir}/{name}/molecule_predictions.json",
        "domain_pred_fp": f"{output_dir}/{name}/domain_predictions.json",
        "domain_embedding_fp": f"{output_dir}/{name}/domain_embedding.pkl",
        "adenylation_pred_fp": f"{output_dir}/{name}/A_predictions.json",
        "acyltransferase_pred_fp": f"{output_dir}/{name}/AT_predictions.json",
        "ketosynthase_pred_fp": f"{output_dir}/{name}/KS_predictions.json",
        "ketoreductase_pred_fp": f"{output_dir}/{name}/KR_predictions.json",
        "dehydratase_pred_fp": f"{output_dir}/{name}/DH_predictions.json",
        "enoylreductase_pred_fp": f"{output_dir}/{name}/ER_predictions.json",
        "thiolation_pred_fp": f"{output_dir}/{name}/T_predictions.json",
    }
    # check if missing files
    for k, v in filelookup.items():
        if os.path.exists(v) == False:
            return None
    else:
        return filelookup


def upload_to_knowledge_graph(
    nuc_fasta_filename: str, output_dir: str, genome_id: Optional[int] = None
):
    # file paths
    filelookup = get_filelookup(
        nuc_fasta_filename=nuc_fasta_filename, output_dir=output_dir
    )
    if filelookup is None:
        raise ValueError("Missing files")
    # upload contigs
    contigs_uploaded = Prodigal.upload_contigs_from_fp(
        prodigal_fp=filelookup["prodigal_fp"]
    )
    # upload genomes
    genomes_uploaded = Prodigal.upload_genome_from_fp(
        prodigal_fp=filelookup["prodigal_fp"],
        nuc_fasta_fp=nuc_fasta_filename,
        genome_id=genome_id,
        contigs_uploaded=contigs_uploaded,
    )
    # upload orfs
    orfs_uploaded = Prodigal.upload_orfs_from_fp(
        prodigal_fp=filelookup["prodigal_fp"],
        contigs_uploaded=contigs_uploaded,
    )
    # upload primary metabolism
    primary_metabolism_uploaded = (
        PrimaryMetabolismPredictor.upload_primary_metabolism_from_fp(
            primary_metabolism_pred_fp=filelookup[
                "primary_metabolism_pred_fp"
            ],
            genome_id=genome_id,
            orfs_uploaded=orfs_uploaded,
            genome_uploaded=genomes_uploaded,
        )
    )
    # upload secondary metabolism
    bgc_uploaded = SecondaryMetabolismPredictor.upload_bgcs_from_fp(
        nuc_fasta_fp=nuc_fasta_filename,
        bgc_pred_fp=filelookup["bgc_pred_fp"],
        genome_id=genome_id,
        orfs_uploaded=orfs_uploaded,
        contigs_uploaded=contigs_uploaded,
        genome_uploaded=genomes_uploaded,
    )
    # upload protein embeddings and ec1 annotations
    protein_embs_uploaded = ProteinEmbedder.upload_protein_embeddings_from_fp(
        prodigal_fp=filelookup["prodigal_fp"],
        protein_embedding_fp=filelookup["protein_embedding_fp"],
        bgc_pred_fp=filelookup["bgc_pred_fp"],
        primary_metabolism_pred_fp=filelookup["primary_metabolism_pred_fp"],
        orfs_uploaded=orfs_uploaded,
    )
    # upload ec predictions
    ec_uploaded = ProteinDecoder.upload_protein_decoding_from_fp(
        ec_pred_fp=filelookup["ec_pred_fp"],
        label_type="EC4Label",
        protein_embs_uploaded=protein_embs_uploaded,
    )
    # upload ko predictions
    ko_uploaded = ProteinDecoder.upload_protein_decoding_from_fp(
        ko_pred_fp=filelookup["ko_pred_fp"],
        label_type="KeggOrthologLabel",
        protein_embs_uploaded=protein_embs_uploaded,
    )
    # upload gene family predictions
    gene_family_uploaded = ProteinDecoder.upload_protein_decoding_from_fp(
        gene_family_pred_fp=filelookup["gene_family_pred_fp"],
        label_type="GeneFamilyLabel",
        protein_embs_uploaded=protein_embs_uploaded,
    )
    # upload gene predictions
    gene_uploaded = ProteinDecoder.upload_protein_decoding_from_fp(
        gene_pred_fp=filelookup["gene_pred_fp"],
        label_type="GeneLabel",
        protein_embs_uploaded=protein_embs_uploaded,
    )
    # upload molecule predictions
    mol_uploaded = ProteinDecoder.upload_protein_decoding_from_fp(
        mol_pred_fp=filelookup["mol_pred_fp"],
        label_type="BioactivePeptideLabel",
        protein_embs_uploaded=protein_embs_uploaded,
    )
    # upload domain predictions
    domains_uploaded = DomainPredictor.upload_domains_from_fp(
        domain_pred_fp=filelookup["domain_pred_fp"],
        prodigal_fp=filelookup["prodigal_fp"],
        orfs_uploaded=orfs_uploaded,
    )
    # upload domain embeddings
    domain_embs_uploaded = DomainEmbedder.upload_domain_embeddings_from_fp(
        domain_pred_fp=filelookup["domain_pred_fp"],
        domain_embedding_fp=filelookup["domain_embedding_fp"],
        domains_uploaded=domains_uploaded,
    )
    # upload adenylation predictions
    adenylation_uploaded = DomainDecoder.upload_domain_decoding_from_fp(
        knn_fp=filelookup["adenylation_pred_fp"],
        label_type="SubstrateLabel",
        domain_embs_uploaded=domain_embs_uploaded,
    )
    # upload acyltransferase predictions
    acyltransferase_uploaded = DomainDecoder.upload_domain_decoding_from_fp(
        knn_fp=filelookup["acyltransferase_pred_fp"],
        label_type="SubstrateLabel",
        domain_embs_uploaded=domain_embs_uploaded,
    )
    # upload ketosynthase predictions
    ketosynthase_uploaded = DomainDecoder.upload_domain_decoding_from_fp(
        knn_fp=filelookup["ketosynthase_pred_fp"],
        label_type="DomainFunctionalLabel",
        domain_embs_uploaded=domain_embs_uploaded,
    )
    # upload ketoreductase predictions
    ketoreductase_uploaded = DomainDecoder.upload_domain_decoding_from_fp(
        knn_fp=filelookup["ketoreductase_pred_fp"],
        label_type="DomainFunctionalLabel",
        domain_embs_uploaded=domain_embs_uploaded,
    )
    # upload dehydratase predictions
    dehydratase_uploaded = DomainDecoder.upload_domain_decoding_from_fp(
        knn_fp=filelookup["dehydratase_pred_fp"],
        label_type="DomainFunctionalLabel",
        domain_embs_uploaded=domain_embs_uploaded,
    )
    # upload enoylreductase predictions
    enoylreductase_uploaded = DomainDecoder.upload_domain_decoding_from_fp(
        knn_fp=filelookup["enoylreductase_pred_fp"],
        label_type="DomainFunctionalLabel",
        domain_embs_uploaded=domain_embs_uploaded,
    )
    # upload thiolation predictions
    thiolation_uploaded = DomainDecoder.upload_domain_decoding_from_fp(
        knn_fp=filelookup["thiolation_pred_fp"],
        label_type="DomainSubclassLabel",
        domain_embs_uploaded=domain_embs_uploaded,
    )
