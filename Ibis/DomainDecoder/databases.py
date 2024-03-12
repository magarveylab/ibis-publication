from Ibis.Utilities.Qdrant.base import QdrantBase


class IbisAdenylation(QdrantBase):

    def __init__(self):

        super().__init__(
            collection_name="ibis_adenylation",
            memory_strategy="disk",
            label_alias="substrate",
            embedding_dim=1024,
            memmap_threshold=20000,
        )


class IbisAcyltransferase(QdrantBase):

    def __init__(self):

        super().__init__(
            collection_name="ibis_acyltransferase",
            memory_strategy="disk",
            label_alias="substrate",
            embedding_dim=1024,
            memmap_threshold=20000,
        )


class IbisKetosynthase(QdrantBase):

    def __init__(self):

        super().__init__(
            collection_name="ibis_ketosynthase",
            memory_strategy="disk",
            label_alias="functional",
            embedding_dim=1024,
            memmap_threshold=20000,
        )


class IbisKetoreductase(QdrantBase):

    def __init__(self):

        super().__init__(
            collection_name="ibis_ketoreductase",
            memory_strategy="disk",
            label_alias="functional",
            embedding_dim=1024,
            memmap_threshold=20000,
        )


class IbisDehydratase(QdrantBase):

    def __init__(self):

        super().__init__(
            collection_name="ibis_dehydratase",
            memory_strategy="disk",
            label_alias="functional",
            embedding_dim=1024,
            memmap_threshold=20000,
        )


class IbisEnoylreductase(QdrantBase):

    def __init__(self):

        super().__init__(
            collection_name="ibis_enoylreductase",
            memory_strategy="disk",
            label_alias="functional",
            embedding_dim=1024,
            memmap_threshold=20000,
        )


class IbisThiolation(QdrantBase):

    def __init__(self):

        super().__init__(
            collection_name="ibis_thiolation",
            memory_strategy="disk",
            label_alias="subclass",
            embedding_dim=1024,
            memmap_threshold=20000,
        )
