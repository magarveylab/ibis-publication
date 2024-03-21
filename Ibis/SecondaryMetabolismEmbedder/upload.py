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
    contig_id: int
    hash_id: int
    contig_start: int
    contig_stop: int
    embedding: np.array


def upload_bgc_embeddings(
    bgcs: List[EmbeddingDict], bgcs_uploaded: bool, bs: int = 1000
) -> bool:
    if len(bgcs) == 0:
        return False
    # add region ids
    for c in bgcs:
        contig_id = c["contig_id"]
        contig_start = c["contig_start"]
        contig_stop = c["contig_stop"]
        c["region_id"] = f"{contig_id}_{contig_start}_{contig_stop}"
        c["embedding"] = c["embedding"].tolist()
    # upload embeddings
    upload_embeddings(
        node_type="MetabolomicRegionEmbedding",
        data=bgcs,
    )
    # connect embeddings to regions
    if bgcs_uploaded:
        batches = batchify(bgcs, bs=bs)
        for batch in tqdm(
            batches, desc="Adding relationships between bgcs and embeddings"
        ):
            batch_str = stringfy_dicts(batch, keys=["hash_id", "region_id"])
            run_cypher(
                f"""UNWIND {batch_str} as row
                        MATCH (n: MetabolomicRegion {{region_id: row.region_id}}),
                            (m: MetabolomicRegionEmbedding {{hash_id: row.hash_id}})
                        MERGE (n)-[:metab_to_embedding]->(m)
            """
            )
    return True
