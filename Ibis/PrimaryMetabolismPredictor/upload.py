from typing import List

from tqdm import tqdm

from Ibis.PrimaryMetabolismPredictor.datastructs import PredictedPathwayDict
from Ibis.Utilities.knowledge_graph import batchify, run_cypher, stringfy_dicts


def upload_predicted_pathways(
    preds: List[PredictedPathwayDict],
    orfs_uploaded: bool,
    genome_uploaded: bool,
    bs: int = 1000,
) -> bool:
    if len(preds) > 0:
        batches = batchify(preds, bs=bs)
        for batch in tqdm(
            batches,
            bs=bs,
            desc="Uploading pathway predictions",
            leave=False,
        ):
            batch_str = stringfy_dicts(
                batch,
                keys=[
                    "prediction_id",
                    "module_completeness_score",
                    "detected_labels",
                    "missing_labels",
                ],
            )
            run_cypher(
                f"""
                UNWIND {batch_str} as row
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
    # reformat relationships between the predicted pathway and other nodes
    genome_ppath_rels = []
    ppath_path_rels = []
    ppath_orf_rels = []
    for p in preds:
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
    if genome_uploaded and len(genome_ppath_rels) > 0:
        batches = batchify(genome_ppath_rels, bs=bs)
        for batch in tqdm(
            batches,
            desc="Adding relationships between genomes and predicted pathways",
            leave=False,
        ):
            batch_str = stringfy_dicts(
                batch, keys=["genome_id", "prediction_id"]
            )
            run_cypher(
                f"""UNWIND {batch_str} as row
                    MATCH (n: Genome {{genome_id: row.genome_id}}),
                        (m: PredictedPathway {{prediction_id: row.prediction_id}})
                    MERGE (n)-[r: genome_to_pred_pathway]->(m)
            """
            )
    # upload relationships between predicted pathway and pathway
    if len(ppath_path_rels) > 0:
        batches = batchify(ppath_path_rels, bs=bs)
        for batch in tqdm(
            batches,
            desc="Adding relationships between predicted pathways and pathway labels",
            leave=False,
        ):
            batch_str = stringfy_dicts(
                batch, keys=["prediction_id", "pathway_id"]
            )
            run_cypher(
                f"""Unwind {batch_str} as row
                Match (n: PredictedPathway {{prediction_id: row.prediction_id}}),
                      (m: Pathway {{pathway_id: row.pathway_id}})
                Merge (n)-[r: pred_pathway_to_pathway]->(m)
            """
            )
    # upload relationships between predicted pathway and orfs
    if orfs_uploaded and len(ppath_orf_rels) > 0:
        batches = batchify(ppath_orf_rels, bs=bs)
        for batch in tqdm(
            batches,
            desc="Adding relationships between predicted pathways and orfs",
            leave=False,
        ):
            batch_str = stringfy_dicts(batch, keys=["prediction_id", "orf_id"])
            run_cypher(
                f"""Unwind {batch_str} as row
                Match (n: PredictedPathway {{prediction_id: row.prediction_id}}),
                    (m: Orf {{orf_id: row.orf_id}})
                Merge (n)-[r: pred_pathway_to_orf]->(m)
            """
            )
    return True
