import json
from typing import List, Optional

from tqdm import tqdm

from Ibis.PrimaryMetabolismPredictor.datastructs import PredictedPathwayDict
from Ibis.Utilities.knowledge_graph import (
    batchify,
    run_cypher,
    stringfy_dicts,
    upload_rel_batch,
)


def reshape_results(
    result_fp: str,
    genome_id: int,
) -> List[PredictedPathwayDict]:
    results_to_upload = []
    dat = json.load(open(result_fp))
    for category, ress in dat.items():
        for res in ress:
            spider_pwy_id = res["neo4j_id"]
            prediction_id = f"{genome_id}_{spider_pwy_id}"
            candidate_orf_ids = []
            for req, cand_orfs in res["candidate_orfs"].items():
                candidate_orf_ids.extend(cand_orfs)
            results_to_upload.append(
                {
                    "prediction_id": prediction_id,
                    "module_completeness_score": res["completeness_score"],
                    "detected_labels": res["matched_criteria"],
                    "missing_labels": res["missing_criteria"],
                    "orf_ids": candidate_orf_ids,
                }
            )
    return results_to_upload


def upload_predicted_pathways(
    preds: List[PredictedPathwayDict],
    batch_size: int = 1000,
):
    pred_dict = {
        p["prediction_id"]: {
            "prediction_id": p["prediction_id"],
            "pathway_id": int(p["prediction_id"].split("_")[-1]),
            "genome_id": int(p["prediction_id"].split("_")[0]),
            "module_completeness_score": p["module_completeness_score"],
            "detected_labels": p["detected_labels"],
            "missing_labels": p["missing_labels"],
            "orf_ids": p["orf_ids"],
        }
        for p in preds
    }
    for batch in tqdm(
        batchify(list(pred_dict.values()), bs=batch_size),
        desc="Uploading pathway predictions",
    ):
        keys = [
            "prediction_id",
            "module_completeness_score",
            "detected_labels",
            "missing_labels",
        ]
        to_upload = [{k: v for k, v in x.items() if k in keys} for x in batch]
        stringified_batch = stringfy_dicts(to_upload, keys=keys)
        run_cypher(
            f"""
            UNWIND {stringified_batch} as row
            MERGE (p:PredictedPathway {{prediction_id: row.prediction_id}})
            ON CREATE
                SET
                    p.module_completeness_score = row.module_completeness_score,
                    p.detected_labels = row.detected_labels,
                    p.missing_labels = row.missing_labels 
            ON MATCH
                SET
                    p.module_completeness_score = row.module_completeness_score,
                    p.detected_labels = row.detected_labels,
                    p.missing_labels = row.missing_labels
            """
        )
    # create the relationships between the predicted pathway and other nodes
    genome_ppath_rels = []
    ppath_path_rels = []
    ppath_orf_rels = []
    for p in pred_dict.values():
        genome_ppath_rels.append(
            {"genome_id": p["genome_id"], "prediction_id": p["prediction_id"]}
        )
        ppath_path_rels.append(
            {
                "prediction_id": p["prediction_id"],
                "pathway_id": p["pathway_id"],
            }
        )
        ppath_orf_rels.extend(
            [
                {"prediction_id": p["prediction_id"], "orf_id": o}
                for o in p["orf_ids"]
            ]
        )
    # upload relationships between genome and predicted pathway
    upload_rel_batch(
        n1_node_type="Genome",
        n1_property="genome_id",
        n1_value_key="genome_id",
        n2_node_type="PredictedPathway",
        n2_property="prediction_id",
        n2_value_key="prediction_id",
        rel_name="genome_to_pred_pathway",
        values=genome_ppath_rels,
        bs=batch_size,
    )
    # upload relationships between predicted pathway and pathway
    upload_rel_batch(
        n1_node_type="PredictedPathway",
        n1_property="prediction_id",
        n1_value_key="prediction_id",
        n2_node_type="Pathway",
        n2_property="pathway_id",
        n2_value_key="pathway_id",
        rel_name="pred_pathway_to_pathway",
        values=ppath_path_rels,
        bs=batch_size,
    )
    # upload relationships between predicted pathway and orfs
    upload_rel_batch(
        n1_node_type="PredictedPathway",
        n1_property="prediction_id",
        n1_value_key="prediction_id",
        n2_node_type="Orf",
        n2_property="orf_id",
        n2_value_key="orf_id",
        rel_name="pred_pathway_to_orf",
        values=ppath_orf_rels,
        bs=batch_size,
    )


def upload_primary_metabolism(fp: str, genome_id: int, bs: int = 1000):
    to_upload = reshape_results(result_fp=fp, genome_id=genome_id)
    upload_predicted_pathways(preds=to_upload, batch_size=bs)
    return True
