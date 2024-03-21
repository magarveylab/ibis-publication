import functools
from typing import List, Optional

import numpy as np
import xxhash
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from Ibis import curdir
from Ibis.PropeptidePredictor.datastructs import (
    ModelInput,
    ModelOutput,
    PipelineIntermediateOutput,
    PipelineOutput,
)
from Ibis.Utilities.class_dicts import get_class_dict
from Ibis.Utilities.onnx import get_onnx_base_model, get_onnx_head
from Ibis.Utilities.preprocess import batchify_tokenized_inputs, slice_proteins
from Ibis.Utilities.RegionCalling.postprocess import (
    parallel_pipeline_token_region_calling,
)
from Ibis.Utilities.tokenizers import get_protbert_tokenizer


class PropeptidePredictorPipeline:

    def __init__(
        self,
        model_fp: str = f"{curdir}/Models/protein_embedder.onnx",
        propeptide_head_fp: str = f"{curdir}/Models/propeptide_predictor.onnx",
        protein_tokenizer: PreTrainedTokenizerFast = get_protbert_tokenizer(),
        propeptide_cls_dict_fp: str = f"{curdir}/PropeptidePredictor/tables/propeptide_residue.csv",
        gpu_id: Optional[int] = None,
        cpu_cores: int = 1,
    ):
        self.model = get_onnx_base_model(model_fp=model_fp, gpu_id=gpu_id)
        self.tokenizer = protein_tokenizer
        self.propeptide_head = get_onnx_head(
            model_fp=propeptide_head_fp, gpu_id=gpu_id
        )
        self.propeptide_cls_dict = get_class_dict(propeptide_cls_dict_fp)
        self.cpu_cores = cpu_cores

    def __call__(self, sequence: str) -> PipelineIntermediateOutput:
        model_inputs = self.preprocess(sequence)
        model_outputs = self._forward(model_inputs)
        return self.postprocess(model_outputs)

    def run(self, sequences: List[str]) -> List[PipelineOutput]:
        out = [self(s) for s in tqdm(sequences)]
        out = parallel_pipeline_token_region_calling(
            pipeline_outputs=out, cpu_cores=self.cpu_cores
        )
        final = []
        for p in out:
            prop_regions = [r for r in p["regions"] if r["label"] == "prop"]
            if len(prop_regions) > 0:
                prop = max(prop_regions, key=lambda x: x["stop"] - x["start"])
                prop["trimmed_sequence"] = p["sequence"][
                    prop["protein_start"] : prop["protein_stop"]
                ]
                final.append(prop)
        return final

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
            batch_last_hidden_state.append(lhs[0])
        # propeptide residue annotation
        batch_predictions = []
        for inp in batch_last_hidden_state:
            # first and last token correspond to [CLS] and [SEP]
            batch_predictions.append(
                self.propeptide_head.run(
                    ["output"], {"input": inp[:, 1:-1, :]}
                )[0]
            )
        propeptide_predictions = np.concatenate(batch_predictions)
        # return output
        return {
            "sequence": model_inputs["sequence"],
            "propeptide_window_predictions": propeptide_predictions,
        }

    def postprocess(
        self, model_outputs: ModelOutput
    ) -> PipelineIntermediateOutput:
        sequence = model_outputs["sequence"]
        logits = model_outputs["propeptide_window_predictions"]
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
                        "label": self.propeptide_cls_dict.get(top),
                        "score": top_score,
                    }
                )
        # return output
        return {
            "protein_id": xxhash.xxh32(sequence).intdigest(),
            "sequence": sequence,
            "residue_classification": residue_classification,
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

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.exp(x).sum(-1, keepdims=True)
