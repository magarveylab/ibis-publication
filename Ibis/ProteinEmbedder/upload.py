from Ibis.Utilities.knowledge_graph import (
    upload_embeddings,
    run_cypher,
    batchify,
    stringfy_dicts,
)
import numpy as np
from typing import List, TypedDict


class OrfDict(TypedDict):
    contig_id: int
    contig_start: int
    contig_stop: int
    protein_id: int
    embedding: np.array


def upload_protein_embeddings(
    orfs: List[OrfDict], uploaded_orfs: bool, bs: int = 1000
):
    # upload embeddings
    unique = {}
    for o in orfs:
        hash_id = o["protein_id"]
        embedding = o["embedding"].tolist()
        unique[hash_id] = {"hash_id": hash_id, "embedding": embedding}
    upload_embeddings(node_type="OrfEmbedding", data=list(unique.values()))
    # connect embeddings to orfs
    if uploaded_orfs:
        # reformat rels
        rels = []
        for o in orfs:
            contig_id = o["contig_id"]
            contig_start = o["contig_start"]
            contig_stop = o["contig_stop"]
            orf_id = f"{contig_id}_{contig_start}_{contig_stop}"
            hash_id = o["protein_id"]
            rels.append({"orf_id": orf_id, "hash_id": hash_id})
        # upload rels
        batches = batchify(rels)
        for batch in tqdm(batch, desc="Uploading orf to embedding rels"):
            batch_str = stringfy_dicts(batch, keys=["orf_id", "hash_id"])
            run_cypher(
                f"""
                UNWIND {batch_str} as row
                MATCH (n: Orf {{orf_id: row.orf_id}}),
                      (m: OrfEmbedding {{hash_id: row.hash_id}})
                MERGE (n)-[r: orf_to_embedding]->(m)
            """
            )
