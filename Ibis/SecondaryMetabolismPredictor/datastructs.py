import numpy as np
from typing import List, TypedDict

############################################################
# Data Inputs
############################################################


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


class OrfInputWithDomains(TypedDict):
    contig_id: int
    contig_start: int
    contig_stop: int
    embedding: np.array
    domains: List[DomainInput]


class ClusterInput(TypedDict):
    cluster_id: str  # {contig_id}_{start}_{stop}
    orfs: List[OrfInput]
    mibig_chemotypes: List[str]
    internal_chemotypes: List[str]


############################################################
# Intermediate Data Outputs
############################################################


class SecondaryPredictionDict(TypedDict):
    label: Union["core", "peripheral"]
    score: float


class ChemotypePredictionDict(TypedDict):
    label: str
    score: float


class AnnotatedOrfDict(TypedDict, total=False):
    orf_id: int
    secondary: SecondaryPredictionDict
    chemotype: Optional[ChemotypePredictionDict]


class AnnotatedOrfDictWithMeta(TypedDict, total=False):
    orf_id: int
    secondary: SecondaryPredictionDict
    chemotypes: Optional[ChemotypePredictionDict]
    contig_id: Union[str, int]
    contig_start: int
    contig_stop: int


class MibigAnnotatedOrfDict(TypedDict):
    orf_id: int
    chemotypes: List[ChemotypePredictionDict]


class MibigAnnotatedOrfDictWithMeta(TypedDict):
    orf_id: int
    chemotypes: List[ChemotypePredictionDict]
    contig_id: Union[str, int]
    contig_start: int
    contig_stop: int


############################################################
# Data Outputs
############################################################


class ClusterOutput(TypedDict):
    contig_id: int
    contig_start: int
    contig_stop: int
    mibig_chemotypes: List[str]
    internal_chemotypes: List[str]
    num_annotated_orfs: int
    orfs: List[str]
