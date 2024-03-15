from typing import TypedDict, List


class TokenOutput(TypedDict):
    pos: int
    label: str
    score: float


class TokenRegionOutput(TypedDict):
    label: str
    start: int
    stop: int
    score: float


class PipelineIntermediateOutput(TypedDict):
    protein_id: int
    sequence: str
    residue_classification: List[TokenOutput]


class PipelineOutput(TypedDict):
    protein_id: int
    sequence: str
    regions: List[TokenRegionOutput]
