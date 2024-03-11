from typing import TypedDict


class ProdigalOutput(TypedDict):
    protein_id: int
    nuc_id: int
    start: int
    end: int
    sequence: str
