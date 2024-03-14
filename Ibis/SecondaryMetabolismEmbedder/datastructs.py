import numpy as np
from typing import List, TypedDict


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
    domains: List[DomainInput]


class ClusterInput(TypedDict):
    cluster_id: str  # {contig_id}_{start}_{stop}
    orfs: List[OrfInput]
    mibig_chemotypes: List[str]
    internal_chemotypes: List[str]
