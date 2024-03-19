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
    embedding: np.array


class ECDict(TypedDict):
    protein_id: int
    ec1_label: str
    ec1_score: float
    is_enzyme: bool


def upload_protein_embeddings(
    orfs: List[EmbeddingDict], orfs_uploaded: bool, bs: int = 1000
) -> bool:
    if len(orfs) == 0:
        return False
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
        batches = batchify(orfs, bs=bs)
        for batch in tqdm(
            batches, desc="Adding relationships between orfs and embeddings"
        ):
            batch_str = stringfy_dicts(batch, keys=["protein_id"])
            run_cypher(
                f"""
                UNWIND {batch_str} as row
                MATCH (n: Orf {{hash_id: row.protein_id}}),
                      (m: OrfEmbedding {{hash_id: row.protein_id}})
                MERGE (n)-[r: orf_to_embedding]->(m)
            """
            )
    return True


def upload_ec1_annotations(
    orfs: List[ECDict], embedding_uploaded: bool, bs: int = 1000
) -> bool:
    if len(orfs) == 0:
        return False
    batches = batchify(orfs, bs=bs)
    for batch in tqdm(batches, desc="Uploading ec1 annotations"):
        batch_str = stringfy_dicts(
            batch,
            keys=["protein_id", "ec1_label", "ec1_score", "is_enzyme"],
        )
        run_cypher(
            f"""
            UNWIND {batch_str} as row
            MERGE (n: OrfAnnotation {{hash_id: row.hash_id}})
            ON CREATE
                SET n.date = date(),
                    n.is_enzyme = row.is_enzyme,
                    n.ec1_label = row.ec1_label,
                    n.ec1_score = row.ec1_score,
                    n.ran_ec4_knn = False,
                    n.ran_gene_family_knn = False,
                    n.ran_bioactive_peptide_knn = False,
                    n.ran_ko_knn = False,
            ON MATCH
                SET n.date = date(),
                    n.is_enzyme = row.is_enzyme,
                    n.ec1_label = row.ec1_label,
                    n.ec1_score = row.ec1_score,
                    n.ran_ec4_knn = False,
                    n.ran_gene_family_knn = False,
                    n.ran_bioactive_peptide_knn = False,
                    n.ran_ko_knn = False,
        """
        )
    if embedding_uploaded:
        for batch in tqdm(
            batches,
            desc="Adding relationships between Orf embedding and annotation",
        ):
            batch_str = stringfy_dicts(batch, keys=["hash_id"])
            run_cypher(
                f"""
                    UNWIND {batch_str} as row
                    MATCH (n: OrfEmbedding {{hash_id: row.hash_id}}),
                          (m: OrfAnnotation {{hash_id: row.hash_id}})
                    MERGE (n)-[r: orf_embedding_to_annotation]->(m)
            """
            )
    return True
