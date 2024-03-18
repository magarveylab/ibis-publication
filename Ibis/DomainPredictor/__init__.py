from Ibis.DomainPredictor.pipeline import DomainPredictorPipeline
from tqdm import tqdm
import json
import os
from typing import List


def run_on_protein_sequences(
    sequences: List[str], gpu_id: int = 0, cpu_cores: int = 1
):
    pipeline = DomainPredictorPipeline(gpu_id=gpu_id, cpu_cores=cpu_cores)
    return pipeline.run(sequences)


def run_on_bgc_fps(
    filenames: List[str], output_dir: str, gpu_id: int, cpu_cores: int = 1
):
    domain_pred_filenames = []
    # load pipeline
    pipeline = DomainPredictorPipeline(gpu_id=gpu_id, cpu_cores=cpu_cores)
    # analysis
    for bgc_fp in tqdm(filenames):
        name = os.path.basename(os.path.dirname(bgc_fp))
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
        domain_pred_filenames.append(export_fp)
    del pipeline
    return domain_pred_filenames


def upload_domains_from_fp():
    pass
