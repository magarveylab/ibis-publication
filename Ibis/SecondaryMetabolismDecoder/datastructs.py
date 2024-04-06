from typing import TypedDict

import numpy as np


class ComparisonInput(TypedDict):
    region_1: str
    region_2: str
    embedding_1: np.array
    embedding_2: np.array


class ComparisonOutput(TypedDict):
    region_1: str
    region_2: str
    molecular_similarity: str
    probability: float
