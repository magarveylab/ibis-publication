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


def run_on_ko_pred_fp(
    ko_pred_fp: str,
    output_dir: str,
    ec_homology_cutoff: float = 0.6,
    ko_homology_cutoff: float = 0.2,
    module_score: float = 0.7,
    allow_inf_ec: bool = True,
):
    filename = os.path.basename(os.path.dirname(ko_pred_fp))
    export_fp = os.path.join(
        output_dir, filename, "primary_metabolism_predictions.json"
    )
    prodigal_fp = os.path.join(output_dir, filename, "prodigal.json")
    ec_pred_fp = os.path.join(output_dir, filename, "ec_predictions.json")
    embed_fp = os.path.join(output_dir, filename, "protein_embedding.pkl")
    if os.path.exists(export_fp) == False:
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
        json.dump(
            {"ko_results": ko_results, "ec_results": ec_results},
            open(export_fp, "w"),
        )
    return export_fp


def parallel_run_on_ko_pred_fps(
    filenames: List[str],
    output_dir: str,
    cpu_cores: int = 1,
):
    funct = partial(run_on_ko_pred_fp, output_dir=output_dir)
    pool = Pool(cpu_cores)
    process = pool.imap_unordered(funct, filenames)
    out = [p for p in tqdm(process, total=len(filenames))]
    pool.close()
    return out


def upload_primary_metabolism_from_fp(
    primary_metabolism_pred_fp: str,
    genome_id: int,
    orfs_uploaded: bool,
    genome_uploaded: bool,
):
    if isinstance(genome_id, int):
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
                        "module_completeness_score": res["completeness_score"],
                        "detected_labels": res["matched_criteria"],
                        "missing_labels": res["missing_criteria"],
                        "orf_ids": list(candidate_orf_ids),
                    }
                )
        return upload_predicted_pathways(
            preds=preds,
            orfs_uploaded=orfs_uploaded,
            genome_uploaded=genome_uploaded,
        )
    else:
        return False
