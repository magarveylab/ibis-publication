from Ibis.Utilities.knowledge_graph import (
    upload_embeddings,
    run_cypher,
    batchify,
    stringfy_dicts,
)
from tqdm import tqdm
import numpy as np
from typing import List, TypedDict


class OrfDict(TypedDict):
    protein_id: int
    embedding: np.array


def upload_protein_embeddings(
    orfs: List[OrfDict], orfs_uploaded: bool, bs: int = 1000
):
    # upload embeddings
    unique = {}
    for o in orfs:
        hash_id = o["protein_id"]
        embedding = o["embedding"].tolist()
        unique[hash_id] = {"hash_id": hash_id, "embedding": embedding}
    upload_embeddings(node_type="OrfEmbedding", data=list(unique.values()))
    # connect embeddings to orfs
    if orfs_uploaded:
        # upload rels
        batches = batchify(rels)
        for batch in tqdm(batch, desc="Uploading orf to embedding rels"):
            batch_str = stringfy_dicts(batch, keys=["protein_id"])
            run_cypher(
                f"""
                UNWIND {batch_str} as row
                MATCH (n: Orf {{hash_id: row.protein_id}}),
                      (m: OrfEmbedding {{hash_id: row.protein_id}})
                MERGE (n)-[r: orf_to_embedding]->(m)
            """
            )
