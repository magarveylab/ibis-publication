from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerFast,
    BertConfig,
    Pipeline,
)
from Ibis.Utilities.preprocess import (
    slice_proteins,
    batchify_tokenized_inputs,
    get_indices,
)
from Ibis.Utilities.onnx import get_onnx_base_model, get_onnx_head
from Ibis.Utilities.tokenizers import get_protbert_tokenizer
from Ibis.Utilities.configs import get_protbert_config
from Ibis.Utilities.class_dicts import get_class_dict
from Ibis.ProteinEmbedder.datastructs import (
    ModelInput,
    ModelOutput,
    PipelineOutput,
)
from Ibis import curdir
from tqdm import tqdm
import numpy as np
import os
from typing import List


class ProteinEmbedderPipeline(Pipeline):

    def __init__(
        self,
        model_fp: str = os.environ.get("PROTEIN_EMBEDDER_MODEL_FP"),
        ec1_head_fp: str = os.environ.get("EC1_HEAD_FP"),
        config: BertConfig = get_protbert_config(),
        protein_tokenizer: PreTrainedTokenizerFast = get_protbert_tokenizer(),
        ec1_cls_dict_fp: str = f"{curdir}/ProteinEmbedder/tables/ec1.csv",
        gpu_id: Optional[int] = None,
    ):
        # load base transformer
        model = get_onnx_base_model(
            model_fp=model_fp, config=config, gpu_id=gpu_id
        )
        super().__init__(model=model, tokenizer=protein_tokenizer)
        # load head
        self.ec1_head = get_onnx_head(model_fp=ec1_head_fp, gpu_id=gpu_id)
        self.ec1_cls_dict = get_class_dict(ec1_cls_dict_fp)

    def run(self, sequences: List[str]):
        return [self(s, **kwargs) for s in tqdm(sequences)]

    def _sanitize_parameters(self):
        preprocess_kwargs = {}
        forward_kwargs = {}
        postprocess_kwargs = {}
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, sequence: str, **preprocess_kwargs) -> ModelInput:
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
            "batch_tokenized_inputs": tokenized_inputs,
        }

    def _forward(self, model_inputs: ModelInput) -> ModelOutput:
        batch_tokenized_inputs = model_inputs["batch_tokenized_inputs"]
        # run pipeline in batches
        batch_last_hidden_state = []
        batch_pooler_output = []
        for inp in batch_tokenized_inputs:
            lhs, po = self.model.run(
                ["last_hidden_state", "pooler_output"], dict(inp)
            )
            batch_last_hidden_state.append(lhs)
            batch_pooler_output.append(po)
        # concatenate outputs
        last_hidden_state = np.concatenate(batch_last_hidden_state)
        pooler_output = np.concatenate(batch_pooler_output)
        # EC1 sequence classification head prediction
        ec1_predictions = np.concatenate(
            [
                self.ec1_head.run(["output"], {"input": inp})[0]
                for inp in batch_pooler_output
            ]
        )
        # return output
        return {
            "sequence": model_inputs["sequence"],
            "lengths": model_inputs["lengths"],
            "cls_window_embeddings": pooler_output,
            "ec1_window_predictions": ec1_predictions,
        }

    def postprocess(
        self, model_outputs: ModelOutput, **postprocess_kwargs
    ) -> PipelineOutput:
        # parameters
        sequence = model_outputs["sequence"]
        min_slice_size = postprocess_kwargs["min_slice_size"]
        lengths = model_outputs["lengths"]
        indices = get_indices(min_slice_size, sequence, lengths)
        # finalize protein embedding (across windows)
        cls_embeddings = model_outputs["cls_window_embedding"]
        avg_cls_embedding = np.mean(
            np.take(cls_embeddings, indices, axis=0), axis=0
        )
        # finalize protein ec1 prediction (across windows)
        logits = model_outputs["ec1_window_predictions"]
        logits = self.softmax(
            np.mean(np.take(logits, indices, axis=0), axis=0)
        )
        label_id = int(logits.argmax())
        label = self.ec1_cls_dict.get(label_id)
        score = round(float(logits[label_id]), 2)
        # output
        return {
            "sequence": sequence,
            "embedding": avg_cls_embedding,
            "ec1": label,
            "ec1_score": score,
        }
