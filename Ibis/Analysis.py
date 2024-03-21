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


def setup_working_directories(
    filenames: List[str], output_dir: str
) -> List[str]:
    basenames = []
    for fp in filenames:
        name = os.path.basename(fp)
        os.makedirs(f"{output_dir}/{name}", exist_ok=True)
        basenames.append(name)
    return basenames


def run_ibis_on_genomes(
    nuc_fasta_filenames: List[str],
    output_dir: str,
    gpu_id: int = 0,
    cpu_cores: int = 1,
):
    # this function will be used to model airflow pipeline
    # setup working directories
    basenames = setup_working_directories(
        filenames=nuc_fasta_filenames, output_dir=output_dir
    )
    # prodigal prediction
    prodigal_preds_created = Prodigal.parallel_run_on_files(
        filenames=nuc_fasta_filenames,
        output_dir=output_dir,
        cpu_cores=cpu_cores,
    )
    # compute protein embeddings
    protein_embs_created = ProteinEmbedder.run_on_files(
        filenames=basenames,
        output_dir=output_dir,
        prodigal_preds_created=prodigal_preds_created,
        gpu_id=gpu_id,
    )

    # compute ec predictions
    ec_preds_created = ProteinDecoder.run_on_files(
        filenames=basenames,
        output_dir=output_dir,
        protein_embs_created=protein_embs_created,
        decode_fn=ProteinDecoder.decode_ec,
        decode_name="ec",
    )
    # compute ko predictions
    ko_preds_created = ProteinDecoder.run_on_files(
        filenames=basenames,
        output_dir=output_dir,
        protein_embs_created=protein_embs_created,
        decode_fn=ProteinDecoder.decode_ko,
        decode_name="ko",
    )
    # compute primary metabolism predictions
    primary_metab_preds_created = (
        PrimaryMetabolismPredictor.parallel_run_on_files(
            filenames=basenames,
            output_dir=output_dir,
            prodigal_preds_created=prodigal_preds_created,
            ec_preds_created=ec_preds_created,
            ko_preds_created=ko_preds_created,
            cpu_cores=cpu_cores,
        )
    )
    # compute bgc boundaries
    bgc_preds_created = SecondaryMetabolismPredictor.run_on_files(
        filenames=basenames,
        output_dir=output_dir,
        prodigal_preds_created=prodigal_preds_created,
        protein_embs_created=protein_embs_created,
        gpu_id=gpu_id,
    )
    # compute gene family predictions
    gene_family_preds_created = ProteinDecoder.trimmed_run_on_files(
        filenames=basenames,
        output_dir=output_dir,
        prodigal_preds_created=prodigal_preds_created,
        protein_embs_created=protein_embs_created,
        bgc_preds_created=bgc_preds_created,
        decode_fn=ProteinDecoder.decode_gene_family,
        decode_name="gene_family",
    )
    # compute gene predictions
    gene_preds_created = ProteinDecoder.trimmed_run_on_files(
        filenames=basenames,
        output_dir=output_dir,
        prodigal_preds_created=prodigal_preds_created,
        protein_embs_created=protein_embs_created,
        bgc_preds_created=bgc_preds_created,
        decode_fn=ProteinDecoder.decode_gene,
        decode_name="gene",
    )
    # compute molecule predictions (ripps and bacteriocins)
    mol_preds_created = ProteinDecoder.trimmed_run_on_files(
        filenames=basenames,
        output_dir=output_dir,
        prodigal_preds_created=prodigal_preds_created,
        protein_embs_created=protein_embs_created,
        bgc_preds_created=bgc_preds_created,
        decode_fn=ProteinDecoder.decode_molecule,
        decode_name="molecule",
    )
    # compute domain predictions
    domain_preds_created = DomainPredictor.run_on_files(
        filenames=basenames,
        output_dir=output_dir,
        prodigal_preds_created=prodigal_preds_created,
        bgc_preds_created=bgc_preds_created,
        gpu_id=gpu_id,
        cpu_cores=cpu_cores,
    )
    # compute domain embeddings
    domain_embs_created = DomainEmbedder.run_on_files(
        filenames=basenames,
        output_dir=output_dir,
        domain_preds_created=domain_preds_created,
        gpu_id=gpu_id,
    )
    # compute adenylation predictions
    adenylation_preds_created = DomainDecoder.run_on_files(
        filenames=basenames,
        output_dir=output_dir,
        domain_embs_created=domain_embs_created,
        decode_fn=DomainDecoder.decode_adenylation,
        target_domain="A",
    )
    # compute acyltransferase predictions
    acyltransferase_preds_created = DomainDecoder.run_on_files(
        filenames=basenames,
        output_dir=output_dir,
        domain_embs_created=domain_embs_created,
        decode_fn=DomainDecoder.decode_acyltransferase,
        target_domain="AT",
    )
    # compute ketosynthase domain functional predictions
    ketosynthase_preds_created = DomainDecoder.run_on_files(
        filenames=basenames,
        output_dir=output_dir,
        domain_embs_created=domain_embs_created,
        decode_fn=DomainDecoder.decode_ketosynthase,
        target_domain="KS",
    )
    # compute ketoreductase domain functional predictions
    ketoreductase_preds_created = DomainDecoder.run_on_files(
        filenames=basenames,
        output_dir=output_dir,
        domain_embs_created=domain_embs_created,
        decode_fn=DomainDecoder.decode_ketoreductase,
        target_domain="KR",
    )
    # compute dehydratase domain functional predictions
    dehydratase_preds_created = DomainDecoder.run_on_files(
        filenames=basenames,
        output_dir=output_dir,
        domain_embs_created=domain_embs_created,
        decode_fn=DomainDecoder.decode_dehydratase,
        target_domain="DH",
    )
    # compute enoylreductase domain functional predictions
    enoylreductase_preds_created = DomainDecoder.run_on_files(
        filenames=basenames,
        output_dir=output_dir,
        domain_embs_created=domain_embs_created,
        decode_fn=DomainDecoder.decode_enoylreductase,
        target_domain="ER",
    )
    # compute thiolation domain subclass predictions
    thiolation_preds_created = DomainDecoder.run_on_files(
        filenames=basenames,
        output_dir=output_dir,
        domain_embs_created=domain_embs_created,
        decode_fn=DomainDecoder.decode_thiolation,
        target_domain="T",
    )
    # compute propeptide predictions
    propeptide_preds_created = PropeptidePredictor.run_on_files(
        filenames=basenames,
        output_dir=output_dir,
        prodigal_preds_created=prodigal_preds_created,
        mol_preds_created=mol_preds_created,
        gpu_id=gpu_id,
        cpu_cores=cpu_cores,
    )
    # compute metabolism embeddings
    bgc_embs_created = SecondaryMetabolismEmbedder.run_on_files(
        filenames=basenames,
        output_dir=output_dir,
        prodigal_preds_created=prodigal_preds_created,
        protein_embs_created=protein_embs_created,
        domain_preds_created=domain_preds_created,
        domain_embs_created=domain_embs_created,
        bgc_preds_created=bgc_preds_created,
        gpu_id=gpu_id,
    )


def get_filelookup(nuc_fasta_filename: str, output_dir: str) -> Dict[str, str]:
    name = os.path.basename(nuc_fasta_filename)
    filelookup = {
        "log_dir": f"{output_dir}/{name}/logs",
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
        "propeptide_pred_fp": f"{output_dir}/{name}/propeptide_predictions.json",
        "bgc_embedding_fp": f"{output_dir}/{name}/bgc_embedding.pkl",
    }
    os.makedirs(filelookup["log_dir"], exist_ok=True)
    # check if missing files
    missing_files = [
        k for k, v in filelookup.items() if os.path.exists(v) == False
    ]
    if len(missing_files) > 0:
        raise ValueError(f"Missing files: {missing_files}")
    else:
        return filelookup


def upload_to_knowledge_graph(
    nuc_fasta_filename: str, output_dir: str, genome_id: Optional[int] = None
):
    # this function will be used to model airflow pipeline
    # get files
    filelookup = get_filelookup(
        nuc_fasta_filename=nuc_fasta_filename, output_dir=output_dir
    )
    # upload contigs
    contigs_uploaded = Prodigal.upload_contigs_from_files(
        prodigal_fp=filelookup["prodigal_fp"],
        log_dir=filelookup["log_dir"],
    )
    # upload genomes
    genomes_uploaded = Prodigal.upload_genome_from_files(
        prodigal_fp=filelookup["prodigal_fp"],
        log_dir=filelookup["log_dir"],
        nuc_fasta_fp=nuc_fasta_filename,
        genome_id=genome_id,
        contigs_uploaded=contigs_uploaded,
    )
    # upload orfs
    orfs_uploaded = Prodigal.upload_orfs_from_files(
        prodigal_fp=filelookup["prodigal_fp"],
        log_dir=filelookup["log_dir"],
        contigs_uploaded=contigs_uploaded,
    )
    # upload primary metabolism
    primary_metabolism_uploaded = (
        PrimaryMetabolismPredictor.upload_primary_metabolism_from_files(
            primary_metabolism_pred_fp=filelookup[
                "primary_metabolism_pred_fp"
            ],
            genome_id=genome_id,
            orfs_uploaded=orfs_uploaded,
            genome_uploaded=genomes_uploaded,
        )
    )
    # upload secondary metabolism
    bgc_uploaded = SecondaryMetabolismPredictor.upload_bgcs_from_files(
        nuc_fasta_fp=nuc_fasta_filename,
        bgc_pred_fp=filelookup["bgc_pred_fp"],
        genome_id=genome_id,
        orfs_uploaded=orfs_uploaded,
        contigs_uploaded=contigs_uploaded,
        genome_uploaded=genomes_uploaded,
    )
    # upload protein embeddings and ec1 annotations
    protein_embs_uploaded = (
        ProteinEmbedder.upload_protein_embeddings_from_files(
            prodigal_fp=filelookup["prodigal_fp"],
            protein_embedding_fp=filelookup["protein_embedding_fp"],
            bgc_pred_fp=filelookup["bgc_pred_fp"],
            primary_metabolism_pred_fp=filelookup[
                "primary_metabolism_pred_fp"
            ],
            orfs_uploaded=orfs_uploaded,
        )
    )
    # upload ec predictions
    ec_uploaded = ProteinDecoder.upload_protein_decoding_from_files(
        knn_fp=filelookup["ec_pred_fp"],
        label_type="EC4Label",
        protein_embs_uploaded=protein_embs_uploaded,
    )
    # upload ko predictions
    ko_uploaded = ProteinDecoder.upload_protein_decoding_from_files(
        knn_fp=filelookup["ko_pred_fp"],
        label_type="KeggOrthologLabel",
        protein_embs_uploaded=protein_embs_uploaded,
    )
    # upload gene family predictions
    gene_family_uploaded = ProteinDecoder.upload_protein_decoding_from_files(
        knn_fp=filelookup["gene_family_pred_fp"],
        label_type="GeneFamilyLabel",
        protein_embs_uploaded=protein_embs_uploaded,
    )
    # upload gene predictions
    gene_uploaded = ProteinDecoder.upload_protein_decoding_from_files(
        knn_fp=filelookup["gene_pred_fp"],
        label_type="GeneLabel",
        protein_embs_uploaded=protein_embs_uploaded,
    )
    # upload molecule predictions
    mol_uploaded = ProteinDecoder.upload_protein_decoding_from_files(
        knn_fp=filelookup["mol_pred_fp"],
        label_type="BioactivePeptideLabel",
        protein_embs_uploaded=protein_embs_uploaded,
    )
    # upload domain predictions
    domains_uploaded = DomainPredictor.upload_domains_from_files(
        domain_pred_fp=filelookup["domain_pred_fp"],
        prodigal_fp=filelookup["prodigal_fp"],
        orfs_uploaded=orfs_uploaded,
    )
    # upload domain embeddings
    domain_embs_uploaded = DomainEmbedder.upload_domain_embeddings_from_files(
        domain_pred_fp=filelookup["domain_pred_fp"],
        domain_embedding_fp=filelookup["domain_embedding_fp"],
        domains_uploaded=domains_uploaded,
    )
    # upload adenylation predictions
    adenylation_uploaded = DomainDecoder.upload_domain_decoding_from_files(
        knn_fp=filelookup["adenylation_pred_fp"],
        label_type="SubstrateLabel",
        domain_embs_uploaded=domain_embs_uploaded,
    )
    # upload acyltransferase predictions
    acyltransferase_uploaded = DomainDecoder.upload_domain_decoding_from_files(
        knn_fp=filelookup["acyltransferase_pred_fp"],
        label_type="SubstrateLabel",
        domain_embs_uploaded=domain_embs_uploaded,
    )
    # upload ketosynthase predictions
    ketosynthase_uploaded = DomainDecoder.upload_domain_decoding_from_files(
        knn_fp=filelookup["ketosynthase_pred_fp"],
        label_type="DomainFunctionalLabel",
        domain_embs_uploaded=domain_embs_uploaded,
    )
    # upload ketoreductase predictions
    ketoreductase_uploaded = DomainDecoder.upload_domain_decoding_from_files(
        knn_fp=filelookup["ketoreductase_pred_fp"],
        label_type="DomainFunctionalLabel",
        domain_embs_uploaded=domain_embs_uploaded,
    )
    # upload dehydratase predictions
    dehydratase_uploaded = DomainDecoder.upload_domain_decoding_from_files(
        knn_fp=filelookup["dehydratase_pred_fp"],
        label_type="DomainFunctionalLabel",
        domain_embs_uploaded=domain_embs_uploaded,
    )
    # upload enoylreductase predictions
    enoylreductase_uploaded = DomainDecoder.upload_domain_decoding_from_files(
        knn_fp=filelookup["enoylreductase_pred_fp"],
        label_type="DomainFunctionalLabel",
        domain_embs_uploaded=domain_embs_uploaded,
    )
    # upload thiolation predictions
    thiolation_uploaded = DomainDecoder.upload_domain_decoding_from_files(
        knn_fp=filelookup["thiolation_pred_fp"],
        label_type="DomainSubclassLabel",
        domain_embs_uploaded=domain_embs_uploaded,
    )
    # upload propeptide predictions
    propeptides_uploaded = PropeptidePredictor.upload_propetides_from_files(
        propeptide_pred_fp=filelookup["propeptide_pred_fp"],
        orfs_uploaded=orfs_uploaded,
    )
    # upload bgc embedding
    bgc_embs_uploaded = (
        SecondaryMetabolismEmbedder.upload_bgc_embeddings_from_files(
            nuc_fasta_fp=nuc_fasta_filename,
            bgc_embedding_fp=filelookup["bgc_embedding_fp"],
            bgcs_uploaded=bgc_uploaded,
        )
    )
