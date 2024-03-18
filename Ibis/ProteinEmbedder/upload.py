from Ibis.Utilities.knowledge_graph import upload_embeddings
import numpy as np
from typing import List, TypedDict


class EmbeddingDict(TypedDict):
    contig_id: int
    contig_start: int
    contig_stop: int
    protein_id: int
    embedding: np.array


def upload_protein_embeddings(orfs: List[OrfDict]):
    # upload embeddings
    unique = {}
    for o in orfs:
        hash_id = o["protein_id"]
        embedding = o["embedding"].tolist()
        unique[hash_id] = {"hash_id": hash_id, "embedding": embedding}
    upload_embeddings(node_type="OrfEmbedding", data=list(unique.values()))
    # connect embeddings to orfs
    rels = []
    for o in orfs:
        contig_id = o["contig_id"]
        contig_start = o["contig_start"]
        contig_stop = o["contig_stop"]
