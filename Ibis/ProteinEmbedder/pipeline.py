from typing import List, Optional

import numpy as np
import xxhash
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from Ibis import curdir
from Ibis.ProteinEmbedder.datastructs import (
    ModelInput,
    ModelOutput,
    PipelineOutput,
)
from Ibis.Utilities.class_dicts import get_class_dict
from Ibis.Utilities.onnx import get_onnx_base_model, get_onnx_head
from Ibis.Utilities.preprocess import (
    batchify_tokenized_inputs,
    get_indices,
    slice_proteins,
)
from Ibis.Utilities.tokenizers import get_protbert_tokenizer


class ProteinEmbedderPipeline:

    def __init__(
        self,
        model_fp: str = f"{curdir}/Models/protein_embedder.onnx",
        protein_tokenizer: PreTrainedTokenizerFast = get_protbert_tokenizer(),
        ec1_head_fp: str = f"{curdir}/Models/ec1_predictor.onnx",
        ec1_cls_dict_fp: str = f"{curdir}/ProteinEmbedder/tables/ec1.csv",
        ec2_head_fp: str = None,
        ec2_cls_dict_fp: str = None,
        ec3_head_fp: str = None,
        ec3_cls_dict_fp: str = None,
        ec4_head_fp: str = None,
        ec4_cls_dict_fp: str = None,
        gpu_id: Optional[int] = None,
    ):
        self.model = get_onnx_base_model(model_fp=model_fp, gpu_id=gpu_id)
        self.tokenizer = protein_tokenizer
        self.ec1_head = get_onnx_head(model_fp=ec1_head_fp, gpu_id=gpu_id)
        self.ec1_cls_dict = get_class_dict(ec1_cls_dict_fp)
        if ec2_head_fp:
            self.ec2_head = get_onnx_head(model_fp=ec2_head_fp, gpu_id=gpu_id)
            self.ec2_cls_dict = get_class_dict(ec2_cls_dict_fp)
        if ec3_head_fp:
            self.ec3_head = get_onnx_head(model_fp=ec3_head_fp, gpu_id=gpu_id)
            self.ec3_cls_dict = get_class_dict(ec3_cls_dict_fp)
        if ec4_head_fp:
            self.ec4_head = get_onnx_head(model_fp=ec4_head_fp, gpu_id=gpu_id)
            self.ec4_cls_dict = get_class_dict(ec4_cls_dict_fp)

    def __call__(self, sequence: str):
        model_inputs = self.preprocess(sequence)
        model_outputs = self._forward(model_inputs)
        return self.postprocess(model_outputs)

    def run(self, sequences: List[str]) -> PipelineOutput:
        return [self(s) for s in tqdm(sequences, leave=False)]

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
            batch_pooler_output.append(po[0])
        # concatenate outputs
        pooler_output = np.concatenate(batch_pooler_output)
        # EC1 sequence classification head prediction
        ec1_predictions = np.concatenate(
            [
                self.ec1_head.run(["output"], {"input": inp})[0]
                for inp in batch_pooler_output
            ]
        )
        # Generate output
        output = {
            "sequence": model_inputs["sequence"],
            "lengths": model_inputs["lengths"],
            "cls_window_embeddings": pooler_output,
            "ec1_window_predictions": ec1_predictions,
        }
        # Optionally annotate with additional heads if instantiated
        if hasattr(self, "ec2_head"):
            ec2_predictions = np.concatenate(
                [
                    self.ec2_head.run(["output"], {"input": inp})[0]
                    for inp in batch_pooler_output
                ]
            )
            output["ec2_window_predictions"] = ec2_predictions
        if hasattr(self, "ec3_head"):
            ec3_predictions = np.concatenate(
                [
                    self.ec3_head.run(["output"], {"input": inp})[0]
                    for inp in batch_pooler_output
                ]
            )
            output["ec3_window_predictions"] = ec3_predictions
        if hasattr(self, "ec4_head"):
            ec4_predictions = np.concatenate(
                [
                    self.ec4_head.run(["output"], {"input": inp})[0]
                    for inp in batch_pooler_output
                ]
            )
            output["ec4_window_predictions"] = ec4_predictions
        # return output
        return output

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
        # generate output
        output = {
            "protein_id": xxhash.xxh32(sequence).intdigest(),
            "embedding": avg_cls_embedding,
        }
        # finalize protein ec predictions (across windows)
        for k, v in model_outputs.items():
            if "ec" in k:
                ec_level = k.split("_")[0]
                logits = self.softmax(
                    np.mean(np.take(v, indices, axis=0), axis=0)
                )
                label_id = int(logits.argmax())
                cls_dict = getattr(self, f"{ec_level}_cls_dict")
                label = cls_dict.get(label_id)
                score = round(float(logits[label_id]), 2)
                output[ec_level] = label
                output[f"{ec_level}_score"] = score
        # return output
        return output

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.exp(x).sum(-1, keepdims=True)
