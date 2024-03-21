from typing import List, TypedDict

import numpy as np


class TokenizedInput:
    input_ids: np.array
    attention_mask: np.array
    token_type_ids: np.array


class ModelInput(TypedDict):
    sequence: str
    lengths: List[int]
    batch_tokenized_inputs: List[TokenizedInput]


class ModelOutput(TypedDict):
    sequence: str
    lengths: List[int]
    cls_window_embeddings: np.array


class PipelineOutput(TypedDict):
    domain_id: int
    embedding: np.array
