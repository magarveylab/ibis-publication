import os
from typing import List

from Ibis import (
    DomainDecoder,
    DomainEmbedder,
    DomainPredictor,
    ModulePredictor,
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
    orfs_prepared = SecondaryMetabolismPredictor.parallel_prepare_orfs_for_pipeline_from_files(
        filenames=basenames,
        output_dir=output_dir,
        prodigal_preds_created=prodigal_preds_created,
        protein_embs_created=protein_embs_created,
        cpu_cores=cpu_cores,
    )
    internal_orf_annos_prepared = (
        SecondaryMetabolismPredictor.run_internal_metabolism_pipeline_on_files(
            filenames=basenames,
            output_dir=output_dir,
            orfs_prepared=orfs_prepared,
            gpu_id=gpu_id,
        )
    )
    proximity_based_bgcs_prepared = SecondaryMetabolismPredictor.parallel_call_bgcs_by_proximity_from_files(
        filenames=basenames,
        output_dir=output_dir,
        internal_orf_annos_prepared=internal_orf_annos_prepared,
        cpu_cores=cpu_cores,
    )
    mibig_orf_annos_prepared = (
        SecondaryMetabolismPredictor.run_mibig_metabolism_pipeline_on_files(
            filenames=basenames,
            output_dir=output_dir,
            orfs_prepared=orfs_prepared,
            proximity_based_bgcs_prepared=proximity_based_bgcs_prepared,
            gpu_id=gpu_id,
        )
    )
    bgc_preds_created = SecondaryMetabolismPredictor.parallel_call_bgcs_by_chemotype_from_files(
        filenames=basenames,
        output_dir=output_dir,
        orfs_prepared=orfs_prepared,
        internal_orf_annos_prepared=internal_orf_annos_prepared,
        mibig_orf_annos_prepared=mibig_orf_annos_prepared,
        cpu_cores=cpu_cores,
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
        prodigal_preds_created=prodigal_preds_created,
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
    # compute modules
    module_preds_created = ModulePredictor.run_on_files(
        filenames=basenames,
        output_dir=output_dir,
        domain_preds_created=domain_preds_created,
        adenylation_preds_created=adenylation_preds_created,
        acyltransferase_preds_created=acyltransferase_preds_created,
        ketosynthase_preds_created=ketosynthase_preds_created,
        ketoreductase_preds_created=ketoreductase_preds_created,
        dehydratase_preds_created=dehydratase_preds_created,
        enoylreductase_preds_created=enoylreductase_preds_created,
        thiolation_preds_created=thiolation_preds_created,
        cpu_cores=cpu_cores,
    )
