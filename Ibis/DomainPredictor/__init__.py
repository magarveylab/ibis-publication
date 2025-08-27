import json
import os
from typing import List

from tqdm import tqdm

from Ibis.DomainPredictor.pipeline import DomainPredictorPipeline

########################################################################
# General functions
########################################################################


def run_on_protein_sequences(
    sequences: List[str], gpu_id: int = 0, cpu_cores: int = 1
):
    pipeline = DomainPredictorPipeline(gpu_id=gpu_id, cpu_cores=cpu_cores)
    return pipeline.run(sequences)


########################################################################
# Airflow inference functions
########################################################################


def run_on_files(
    filenames: List[str],
    output_dir: str,
    prodigal_preds_created: bool,
    bgc_preds_created: bool,
    gpu_id: int,
    cpu_cores: int = 1,
) -> bool:
    if prodigal_preds_created == False:
        raise ValueError("Prodigal predictions not created")
    if bgc_preds_created == False:
        raise ValueError("BGC predictions not created")
    # load pipeline
    pipeline = DomainPredictorPipeline(gpu_id=gpu_id, cpu_cores=cpu_cores)
    # analysis
    for name in tqdm(filenames, leave=False, desc="Running DomainPredictor"):
        export_fp = f"{output_dir}/{name}/domain_predictions.json"
        if os.path.exists(export_fp) == False:
            # load sequence lookup
            prodigal_fp = f"{output_dir}/{name}/prodigal.json"
            sequence_lookup = {}
            for protein in json.load(open(prodigal_fp)):
                contig_id = protein["contig_id"]
                contig_start = protein["contig_start"]
                contig_stop = protein["contig_stop"]
                orf_id = f"{contig_id}_{contig_start}_{contig_stop}"
                sequence_lookup[orf_id] = protein["sequence"]
            # find orfs from modular systems
            bgc_fp = f"{output_dir}/{name}/bgc_predictions.json"
            sequences_to_run = set()
            for cluster in json.load(open(bgc_fp)):
                internal_chemotypes = cluster["internal_chemotypes"]
                if (
                    "TypeIPolyketide" in internal_chemotypes
                    or "NonRibosomalPeptide" in internal_chemotypes
                ):
                    for orf_id in cluster["orfs"]:
                        sequences_to_run.add(sequence_lookup[orf_id])
            # analysis
            if len(sequences_to_run) > 0:
                out = pipeline.run(list(sequences_to_run))
            else:
                out = []
            with open(export_fp, "w") as f:
                json.dump(out, f)
    del pipeline
    return True


########################################################################
# Airflow upload functions
########################################################################


def upload_domains_from_files(
    domain_pred_fp: str, prodigal_fp: str, log_dir: str, orfs_uploaded: bool
):
    from Ibis.DomainPredictor.upload import upload_domains

    log_fp = f"{log_dir}/domain_uploaded.json"
    if os.path.exists(log_fp) == False:
        # domain lookup
        domain_lookup = {}
        for p in json.load(open(domain_pred_fp)):
            domain_lookup[p["protein_id"]] = p["regions"]
        # upload domains
        orfs = []
        for p in json.load(open(prodigal_fp)):
            protein_id = p["protein_id"]
            orfs.append(
                {
                    "contig_id": p["contig_id"],
                    "protein_id": protein_id,
                    "contig_start": p["contig_start"],
                    "contig_stop": p["contig_stop"],
                    "domains": domain_lookup.get(protein_id, []),
                }
            )
        upload_domains(orfs=orfs, orfs_uploaded=orfs_uploaded)
        json.dump({"upload": True}, open(log_fp, "w"))
    return True
