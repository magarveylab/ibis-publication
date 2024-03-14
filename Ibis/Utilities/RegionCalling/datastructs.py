from typing import TypedDict


class TokenOutput(TypedDict):
    pos: int
    label: str
    score: float


class TokenRegionOutput(TypedDict):
    label: str
    start: int
    end: int
    score: float


class PipelineIntermediateOutput(TypedDict):
    domain_id: int
    sequence: str
    residue_classification: List[TokenOutput]


class PipelineOutput(TypedDict):
    domain_id: int
    sequence: str
    regions: List[TokenRegionOutput]
