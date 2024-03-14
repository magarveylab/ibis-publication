from typing import TypedDict


class ProdigalOutput(TypedDict):
    protein_id: int
    contig_id: int
    contig_start: int
    contig_stop: int
    sequence: str
