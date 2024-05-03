from typing import List, TypedDict

from tqdm import tqdm

from Ibis.Utilities.knowledge_graph import batchify, run_cypher, stringfy_dicts


class KNNDict(TypedDict):
    label: str
    reference_id: int
    similarity: float
    homology: float
    rank: int


class KNNData(TypedDict):
    query_id: int
    predictions: List[KNNDict]


knn_meta_lookup = {
    "SubstrateLabel": {
        "rel": "domain_anno_to_substrate",
        "process": "ran_substrate_knn",
    },
    "DomainSubclassLabel": {
        "rel": "domain_anno_to_subclass",
        "process": "ran_subclass_knn",
    },
    "DomainFunctionalLabel": {
        "rel": "domain_anno_to_functional",
        "process": "ran_functional_knn",
    },
}


def upload_knn(
    annotations: List[KNNData], label_type: str, bs: int = 1000
) -> bool:
    if label_type not in knn_meta_lookup:
        raise ValueError(f"Invalid Label Type: {label_type}")
    rel = knn_meta_lookup[label_type]["rel"]
    process = knn_meta_lookup[label_type]["process"]
    if len(annotations) == 0:
        return False
    batches = batchify(annotations, bs=bs)
    for batch in tqdm(
        batches, desc=f"Uploading {label_type} knn rels", leave=False
    ):
        batch_str = stringfy_dicts(batch, keys=["query_id"])
        # update orf annotation
        run_cypher(
            f"""
            UNWIND {batch_str} as row
            MERGE (n: DomainAnnotation {{hash_id: row.query_id}})
            ON MATCH
                SET n.date = date(),
                    n.{process} = True
        """
        )
    # add knn relationships
    input_data = []
    for a in annotations:
        for k in a["predictions"]:
            k["query_id"] = a["query_id"]
            input_data.append(k)
    if len(input_data) == 0:
        return False
    batches = batchify(input_data, bs=bs)
    for batch in tqdm(batches, desc=f"Adding {label_type} knn rels"):
        batch_str = stringfy_dicts(
            batch,
            keys=[
                "query_id",
                "label",
                "reference_id",
                "similarity",
                "homology",
                "rank",
            ],
        )
        run_cypher(
            f"""
                UNWIND {batch_str} as row
                MATCH (n: DomainAnnotation {{hash_id: row.query_id}}),
                      (m: {label_type} {{label: row.label}})
                MERGE (n)-[r:{rel}]->(m)
                ON CREATE
                    SET r.similarity = row.similarity,
                        r.homology = row.homology,
                        r.reference_id = row.reference_id,
                        r.rank = row.rank,
                        r.date = date()
                ON MATCH
                    SET r.similarity = row.similarity,
                        r.homology = row.homology,
                        r.reference_id = row.reference_id,
                        r.rank = row.rank,
                        r.date = date()
            """
        )
    return True
