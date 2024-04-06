from Ibis.Utilities.Qdrant.base import QdrantBase


class IbisKnownCluster(QdrantBase):

    def __init__(self):

        super().__init__(
            collection_name="ibis_known_cluster",
            memory_strategy="disk",
            label_alias="metabolomic_region_id",
            embedding_dim=256,
            memmap_threshold=20000,
        )
