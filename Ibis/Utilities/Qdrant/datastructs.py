import numpy as np
from typing import TypedDict, List, Union


class DataQuery(TypedDict):
    query_id: int  # identifier
    embedding: np.array


class SimHitResponse(TypedDict):
    subject_id: int
    similarity: float
    label: str
    data: dict


class SearchResponse(TypedDict):
    query_id: int  # identifier
    hits: List[SimHitResponse]
