import json
import os
from functools import partial
from multiprocessing import Pool
from typing import List

from tqdm import tqdm

from Ibis.PrimaryMetabolismPredictor.annotation import (
    KOAnnotator,
    annotate_enzyme_orfs_with_pathways,
)
from Ibis.PrimaryMetabolismPredictor.preprocess import (
    merge_protein_annotations,
)
from Ibis.PrimaryMetabolismPredictor.upload import upload_predicted_pathways

########################################################################
# Airflow inference functions
########################################################################


def run_on_single_file(
    filename: str,
    output_dir: str,
    ec_homology_cutoff: float = 0.6,
    ko_homology_cutoff: float = 0.2,
    module_score: float = 0.7,
    allow_inf_ec: bool = True,
) -> bool:
    export_fp = os.path.join(
        output_dir, filename, "primary_metabolism_predictions.json"
    )
    if os.path.exists(export_fp) == False:
        prodigal_fp = os.path.join(output_dir, filename, "prodigal.json")
        ec_pred_fp = os.path.join(output_dir, filename, "ec_predictions.json")
        ko_pred_fp = os.path.join(output_dir, filename, "ko_predictions.json")
        # do things.
        annots = merge_protein_annotations(
            prodigal_fp=prodigal_fp,
            ko_pred_fp=ko_pred_fp,
            ec_pred_fp=ec_pred_fp,
        )
        ec_results = annotate_enzyme_orfs_with_pathways(
            orfs=annots,
            homology_score_threshold=ec_homology_cutoff,
            module_completeness_threshold=module_score,
            annotate_kegg=True,
        )
        ko_annotator = KOAnnotator(
            allow_inferred_kegg_ecs=allow_inf_ec,
            ec_homology_cutoff=ec_homology_cutoff,
            ko_homology_cutoff=ko_homology_cutoff,
            module_completeness_threshold=module_score,
        )
        ko_results = ko_annotator.run_annotation(genome_orfs=annots)
        out = {"ko_results": ko_results, "ec_results": ec_results}
        with open(export_fp, "w") as json_data:
            json.dump(out, json_data)
    return True


########################################################################
# Airflow upload functions
########################################################################


def parallel_run_on_files(
    filenames: List[str],
    output_dir: str,
    prodigal_preds_created: bool,
    ec_preds_created: bool,
    ko_preds_created: bool,
    cpu_cores: int = 1,
) -> bool:
    if prodigal_preds_created == False:
        raise ValueError("Prodigal predictions not created")
    if ec_preds_created == False:
        raise ValueError("EC predictions not created")
    if ko_preds_created == False:
        raise ValueError("KO predictions not created")
    funct = partial(run_on_single_file, output_dir=output_dir)
    pool = Pool(cpu_cores)
    process = pool.imap_unordered(funct, filenames)
    out = [p for p in tqdm(process, total=len(filenames), leave=False)]
    pool.close()
    return True


def upload_primary_metabolism_from_files(
    primary_metabolism_pred_fp: str,
    log_dir: str,
    genome_id: int,
    orfs_uploaded: bool,
    genome_uploaded: bool,
):
    if isinstance(genome_id, int) == False:
        return False
    log_fp = f"{log_dir}/primary_metabolism_uploaded.json"
    if os.path.exists(log_fp) == False:
        preds = []
        dat = json.load(open(primary_metabolism_pred_fp))
        for ress in dat.values():
            for res in ress:
                spider_pwy_id = res["neo4j_id"]
                prediction_id = f"{genome_id}_{spider_pwy_id}"
                candidate_orf_ids = set()
                for cand_orfs in res["candidate_orfs"].values():
                    candidate_orf_ids.update(cand_orfs)
                preds.append(
                    {
                        "prediction_id": prediction_id,
                        "genome_id": genome_id,
                        "pathway_id": spider_pwy_id,
                        "module_completeness_score": res["completeness_score"],
                        "detected_labels": res["matched_criteria"],
                        "missing_labels": res["missing_criteria"],
                        "orf_ids": list(candidate_orf_ids),
                    }
                )
        upload_predicted_pathways(
            preds=preds,
            orfs_uploaded=orfs_uploaded,
            genome_uploaded=genome_uploaded,
        )
        json.dump({"upload": True}, open(log_fp, "w"))
    return True
