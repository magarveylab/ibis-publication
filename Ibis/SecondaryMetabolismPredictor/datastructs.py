import numpy as np
from typing import List, TypedDict, Literal, Optional


class DomainInput(TypedDict):
    protein_start: int
    protein_stop: int
    label: str
    embedding: np.array


class OrfInput(TypedDict):
    contig_id: int
    contig_start: int
    contig_stop: int
    embedding: np.array


class SecondaryPredictionDict(TypedDict):
    label: Literal["core", "peripheral"]
    score: float


class ChemotypePredictionDict(TypedDict):
    label: str
    score: float


class InternalAnnotatedOrfDict(TypedDict, total=False):
    orf_id: int
    secondary: SecondaryPredictionDict
    chemotype: Optional[ChemotypePredictionDict]


class InternalAnnotatedOrfDictWithMeta(TypedDict, total=False):
    orf_id: int
    secondary: SecondaryPredictionDict
    chemotypes: Optional[ChemotypePredictionDict]
    contig_id: int
    contig_start: int
    contig_stop: int


class MibigAnnotatedOrfDict(TypedDict):
    orf_id: int
    chemotypes: List[ChemotypePredictionDict]


class MibigAnnotatedOrfDictWithMeta(TypedDict):
    orf_id: int
    chemotypes: List[ChemotypePredictionDict]
    contig_id: int
    contig_start: int
    contig_stop: int


class ClusterOutput(TypedDict):
    contig_id: int
    contig_start: int
    contig_stop: int
    mibig_chemotypes: List[str]
    internal_chemotypes: List[str]
    num_annotated_orfs: int
    orfs: List[str]
