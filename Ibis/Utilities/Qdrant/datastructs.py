import numpy as np
from typing import TypedDict, List, Union, Literal

acceptable_label_types = Literal[
    "EC4Label",
    "GeneFamilyLabel",
    "BioactivePeptideLabel",
    "MetabolismProteinFamilyLabel",
    "KeggOrthologLabel",
    "SubstrateLabel",
    "DomainSubclassLabel",
    "DomainFunctionalLabel",
]


class DataQuery(TypedDict):
    query_id: int  # identifier
    embedding: np.array


class DistHitResponse(TypedDict):
    subject_id: int
    distance: float
    label: str
    data: dict


class SearchResponse(TypedDict):
    query_id: int  # identifier
    hits: List[DistHitResponse]


class KnnOutput(TypedDict):
    hash_id: str
    label: str
    label_type: acceptable_label_types
    similarity: float
    homology: float
    reference_id: int  # hashed protein ID of reference protein
