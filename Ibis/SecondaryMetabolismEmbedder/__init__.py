from Ibis.SecondaryMetabolismEmbedder.datastructs import (
    ClusterInput,
    ClusterEmbeddingOutput,
)
from Ibis.SecondaryMetabolismEmbedder.pipeline import (
    MetabolismEmbedderPipeline,
)
from tqdm import tqdm
from typing import List, Optional


def embed_clusters(
    clusters: List[ClusterInput],
    gpu_id: Optional[int] = None,
    pipeline: Optional[MetabolismEmbedderPipeline] = None,
) -> List[ClusterEmbeddingOutput]:
    # load pipeline
    if pipeline == None:
        pipeline = BGCEmbedderPipeline(gpu_id=gpu_id)
    # embed
    out = [pipeline(c) for c in tqdm(clusters, desc="Embedding bgcs")]
    return out
