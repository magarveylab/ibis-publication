from typing import List, TypedDict

from tqdm import tqdm

from Ibis.Utilities.knowledge_graph import batchify, run_cypher, stringfy_dicts


class PropeptideDict(TypedDict):
    protein_id: int
    protein_start: int
    protein_stop: int
    trimmed_sequence: str


def upload_propeptides(
    propeptides: List[PropeptideDict], orfs_uploaded: bool, bs: int = 100
) -> bool:
    if len(propeptides) == 0:
        return False
    # add propeptide ids
    for p in propeptides:
        protein_id = p["protein_id"]
        protein_start = p["protein_start"]
        protein_stop = p["protein_stop"]
        p["propeptide_id"] = f"{protein_id}_{protein_start}_{protein_stop}"
    # upload propeptides
    batches = batchify(propeptides, bs=bs)
    for batch in tqdm(
        batches, desc="Uploading propeptides", leave=False, leave=False
    ):
        # add propeptide
        batch_str = stringfy_dicts(
            batch,
            keys=[
                "propeptide_id",
                "protein_start",
                "protein_stop",
                "trimmed_sequence",
            ],
        )
        run_cypher(
            f"""
                UNWIND {batch_str} as row
                MERGE (p: Propeptide {{propeptide_id: row.propeptide_id}})
                SET p.protein_start = row.protein_start,
                    p.protein_stop = row.protein_stop,
                    p.trimmed_sequence = row.trimmed_sequence
            """
        )
    if orfs_uploaded:
        for batch in tqdm(
            batches,
            desc="Adding relationships between orfs and propeptides",
            leave=False,
        ):
            # connect to orf
            batch_str = stringfy_dicts(
                batch,
                keys=[
                    "propeptide_id",
                    "protein_id",
                ],
            )
            run_cypher(
                f"""
                UNWIND {batch_str} as row
                MATCH (o: Orf {{hash_id: row.protein_id}}),
                      (p: Propeptide {{propeptide_id: row.propeptide_id}})
                MERGE (o)-[:orf_to_propeptide]->(p)
            """
            )
    return True
