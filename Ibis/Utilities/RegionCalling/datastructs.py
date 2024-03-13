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
