from transformers import PreTrainedTokenizerFast
from Ibis.Utilities.preprocess import (
    slice_proteins,
    batchify_tokenized_inputs,
    get_indices,
)
from Ibis.Utilities.onnx import get_onnx_base_model, get_onnx_head
from Ibis.Utilities.tokenizers import get_protbert_tokenizer
from Ibis.ProteinEmbedder.datastructs import (
    ModelInput,
    ModelOutput,
    PipelineOutput,
)
from Ibis import curdir
from tqdm import tqdm
import numpy as np
from typing import List, Optional


class DomainEmbedderPipeline:

    def __init__(
        self,
        model_fp: str = f"{curdir}/Models/domain_embedder.onnx",
        protein_tokenizer: PreTrainedTokenizerFast = get_protbert_tokenizer(),
        gpu_id: Optional[int] = None,
    ):
        self.model = get_onnx_base_model(model_fp=model_fp, gpu_id=gpu_id)
        self.tokenizer = protein_tokenizer

    def __call__(self, sequence: str):
        model_inputs = self.preprocess(sequence)
        model_outputs = self._forward(model_inputs)
        return self.postprocess(model_outputs)

    def run(self, sequences: List[str]):
        return [self(s) for s in tqdm(sequences)]

    def preprocess(self, sequence: str) -> ModelInput:
        windows = slice_proteins(sequence)
        lengths = [len(x) for x in windows]
        windows = [" ".join(x) for x in windows]
        tokenized_inputs = self.tokenizer(
            windows, padding=True, return_tensors="np"
        )
        batch_tokenized_inputs = batchify_tokenized_inputs(tokenized_inputs)
        return {
            "sequence": sequence,
            "lengths": lengths,
            "batch_tokenized_inputs": batch_tokenized_inputs,
        }

    def _forward(self, model_inputs: ModelInput) -> ModelOutput:
        batch_tokenized_inputs = model_inputs["batch_tokenized_inputs"]
        # run pipeline in batches
        batch_pooler_output = []
        for inp in batch_tokenized_inputs:
            po = self.model.run(["pooler_output"], dict(inp))
            batch_pooler_output.append(po)
        pooler_output = np.concatenate(batch_pooler_output)
        # return output
        return {
            "sequence": model_inputs["sequence"],
            "lengths": model_inputs["lengths"],
            "cls_window_embeddings": pooler_output,
        }

    def postprocess(self, model_outputs: ModelOutput) -> PipelineOutput:
        # parameters
        sequence = model_outputs["sequence"]
        min_slice_size = 0.1
        lengths = model_outputs["lengths"]
        indices = get_indices(min_slice_size, sequence, lengths)
        # finalize protein embedding (across windows)
        cls_embeddings = model_outputs["cls_window_embeddings"]
        avg_cls_embedding = np.mean(
            np.take(cls_embeddings, indices, axis=0), axis=0
        )
        # return output
        return {
            "domain_id": xxhash.xxh32(sequence).intdigest(),
            "sequence": sequence,
            "embedding": avg_cls_embedding,
        }
