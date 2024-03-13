import numpy as np
from typing import TypedDict, List


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
    sequence: str
    embedding: np.array
