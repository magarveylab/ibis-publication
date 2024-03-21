from typing import List, TypedDict


class TokenOutput(TypedDict):
    pos: int
    label: str
    score: float


class TokenRegionOutput(TypedDict):
    label: str
    protein_start: int
    protein_stop: int
    score: float


class PipelineIntermediateOutput(TypedDict):
    protein_id: int
    sequence: str
    residue_classification: List[TokenOutput]


class PipelineOutput(TypedDict):
    protein_id: int
    sequence: str
    regions: List[TokenRegionOutput]
