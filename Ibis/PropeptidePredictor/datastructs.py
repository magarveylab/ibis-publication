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
    propeptide_window_predictions: np.array


class TokenOutput(TypedDict):
    pos: int
    label: str
    score: float


class PipelineIntermediateOutput(TypedDict):
    protein_id: int
    sequence: str
    residue_classification: List[TokenOutput]


class PipelineOutput(TypedDict):
    protein_id: int
    trimmed_sequence: str
    protein_start: int
    protein_stop: int
    score: float
