from transformers import PreTrainedTokenizerFast
from Ibis.Utilities.preprocess import slice_proteins, batchify_tokenized_inputs
from Ibis.Utilities.onnx import get_onnx_base_model, get_onnx_head
from Ibis.Utilities.tokenizers import get_protbert_tokenizer
from Ibis.Utilities.class_dicts import get_class_dict
from Ibis.Utilities.RegionCalling.postprocess import TokenRegionCaller
from Ibis.DomainPredictor.datastructs import (
    ModelInput,
    ModelOutput,
    PipelineOutput,
)
from Ibis import curdir
from tqdm import tqdm
import numpy as np
from typing import List, Optional


class DomainPredictorPipeline:

    def __init__(
        self,
        model_fp: str = f"{curdir}/Models/ibis3_base.onnx",
        domain_head_fp: str = f"{curdir}/Models/domain_residue_head.onnx",
        protein_tokenizer: PreTrainedTokenizerFast = get_protbert_tokenizer(),
        domain_cls_dict_fp: str = f"{curdir}/DomainPredictor/tables/domain_residue.csv",
        gpu_id: Optional[int] = None,
    ):
        self.model = get_onnx_base_model(model_fp=model_fp, gpu_id=gpu_id)
        self.tokenizer = protein_tokenizer
        self.domain_head = get_onnx_head(
            model_fp=domain_head_fp, gpu_id=gpu_id
        )
        self.domain_cls_dict = get_class_dict(domain_cls_dict_fp)

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
        batch_last_hidden_state = []
        for inp in batch_tokenized_inputs:
            lhs = self.model.run(["last_hidden_state"], dict(inp))
            batch_last_hidden_state.append(lhs)
        # domain residue annotation
        batch_predictions = []
        for inp in batch_last_hidden_state:
            # first and last token correspond to [CLS] and [SEP]
            batch_predictions.append(
                self.domain_head.run(["output"], {"input": inp[:, 1:-1, :]})[0]
            )
        domain_predictions = np.concatenate(batch_predictions)
        # return output
        return {
            "sequence": model_inputs["sequence"],
            "domain_window_predictions": domain_predictions,
        }

    def postprocess(self, model_outputs: ModelOutput) -> PipelineOutput:
        logits = model_outputs["domain_window_predictions"]
        # average logits
        logits = self.softmax(
            functools.reduce(self.merge_overlap_average, logits)
        )
        # remove pad tokens
        residue_classification = []
        logits = logits[: len(sequence)]
        for pos, scores in enumerate(logits):
            # take top prediction that passes threshold
            top = int(scores.argmax())
            top_score = round(float(scores[top]), 2)
            if top_score >= 0.5:
                residue_classification.append(
                    {
                        "pos": pos,
                        "label": self.domain_cls_dict.get(top),
                        "score": top_score,
                    }
                )
        # region calling
        regions = TokenRegionCaller(residue_classification)
        # return output
        return {
            "sequence": sequence,
            "domain_regions": regions,
        }

    def merge_overlap_average(self, a, b, step=256):
        overlapping_length = b.shape[0] - step
        overlapping_arr_a = a[-overlapping_length:]
        overlapping_arr_b = b[:overlapping_length]
        a[-overlapping_length:] = np.mean(
            [overlapping_arr_a, overlapping_arr_b], axis=0
        )
        out_arr = np.concatenate([a, b[overlapping_length:]], axis=0)
        return out_arr
