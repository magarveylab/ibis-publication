from typing import List, TypedDict

import numpy as np
from tqdm import tqdm

from Ibis.Utilities.knowledge_graph import (
    batchify,
    run_cypher,
    stringfy_dicts,
    upload_embeddings,
)


class EmbeddingDict(TypedDict):
    protein_id: int
    protein_start: int
    protein_stop: int
    hash_id: int
    embedding: np.array


def upload_domain_embeddings(
    domains: List[EmbeddingDict], domains_uploaded: bool, bs: int = 1000
) -> bool:
    # upload embeddings
    unique = {}
    for d in domains:
        protein_id = d["protein_id"]
        protein_start = d["protein_start"]
        protein_stop = d["protein_stop"]
        domain_id = f"{protein_id}_{protein_start}_{protein_stop}"
        d["domain_id"] = domain_id
        hash_id = d["hash_id"]
        embedding = d["embedding"].tolist()
        unique[hash_id] = {"hash_id": hash_id, "embedding": embedding}
    if len(unique) > 0:
        upload_embeddings(
            node_type="DomainEmbedding", data=list(unique.values())
        )
    # connect embeddings to domains
    if domains_uploaded and len(domains) > 0:
        batches = batchify(domains, bs=bs)
        for batch in tqdm(batches, desc="Uploading domain to embedding rels"):
            batch_str = stringfy_dicts(batch, keys=["domain_id", "hash_id"])
            run_cypher(
                f"""
                UNWIND {batch_str} as row
                MATCH (n: Domain {{domain_id: row.domain_id}}),
                      (m: DomainEmbedding {{hash_id: row.hash_id}})
                MERGE (n)-[r: domain_to_embedding]->(m)
            """
            )
    return True


def initialize_domain_annotations(
    hash_ids: List[int], embedding_uploaded: bool, bs: int = 1000
):
    if len(hash_ids) > 0:
        batches = batchify(hash_ids, bs=bs)
        for batch in tqdm(batches, desc="Initialize domain annotations"):
            run_cypher(
                f"""
                UNWIND {batch} as row
                MERGE (n: DomainAnnotation {{hash_id: row.hash_id}})
                ON CREATE
                    SET n.date = date(),
                        n.ran_substrate_knn = False,
                        n.ran_subclass_knn = False,
                        n.ran_functional_knn = False,
            """
            )
        if embedding_uploaded:
            for batch in tqdm(
                batches, desc="Connect domain to annotation rels"
            ):
                run_cypher(
                    f"""
                    UNWIND {batch} as row
                    MATCH (n: DomainEmbedding {{hash_id: row.hash_id}}),
                          (m: DomainAnnotation {{hash_id: row.hash_id}})
                    MERGE (n)-[r: domain_embedding_to_annotation]->(m)
                """
                )
    return True
