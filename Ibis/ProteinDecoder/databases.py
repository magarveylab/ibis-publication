from Ibis.Utilities.Qdrant.base import QdrantBase


class IbisKO(QdrantBase):

    def __init__(self):

        super().__init__(
            collection_name="ibis_ko",
            memory_strategy="disk",
            label_alias="ko_id",
            embedding_dim=1024,
            memmap_threshold=20000,
        )


class IbisEC(QdrantBase):

    def __init__(self):

        super().__init__(
            collection_name="ibis_ec",
            memory_strategy="disk",
            label_alias="ec",
            embedding_dim=1024,
            memmap_threshold=20000,
        )


class IbisMolecule(QdrantBase):

    def __init__(self):

        super().__init__(
            collection_name="ibis_molecule",
            memory_strategy="disk",
            label_alias="molecule",
            embedding_dim=1024,
            memmap_threshold=20000,
        )


class IbisGeneFamily(QdrantBase):

    def __init__(self):

        super().__init__(
            collection_name="ibis_gene_family",
            memory_strategy="disk",
            label_alias="gene_family",
            embedding_dim=1024,
            memmap_threshold=20000,
        )


class IbisGene(QdrantBase):

    def __init__(self):

        super().__init__(
            collection_name="ibis_gene",
            memory_strategy="disk",
            label_alias="name",
            embedding_dim=1024,
            memmap_threshold=20000,
        )
