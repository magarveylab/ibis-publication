from typing import List, Optional

import numpy as np
import torch

from Ibis import curdir
from Ibis.SecondaryMetabolismDecoder.datastructs import (
    ComparisonInput,
    ComparisonOutput,
)


def batchify(l, bs=500):  # group calls for database
    return [l[x : x + bs] for x in range(0, len(l), bs)]


class_dict = {
    "tanimoto_bin_0.3": {
        0: "0.7-1",
        1: "0.4-0.7",
        2: "0.1-0.4",
        3: "0-0.1",
    }
}


class MetabolismComparisonPipeline:

    def __init__(
        self,
        model_dir: str = f"{curdir}/Models/metabolism_embedder",
        tanimoto_bin: str = "tanimoto_bin_0.3",
        gpu_id: Optional[int] = None,
    ):
        # bins corrspond to distances (1 - tanimoto score)
        # the first bin for tanimoto_bin_0.3 corresponds 0.7-1
        model_fp = f"{model_dir}/heads/{tanimoto_bin}.pt"
        self.head = torch.jit.load(model_fp)
        self.class_dict = class_dict[tanimoto_bin]
        # move models to gpu (if device defined)
        self.gpu_id = gpu_id
        if isinstance(self.gpu_id, int):
            self.head.to(f"cuda:{self.gpu_id}")

    def __call__(self, data: List[ComparisonInput]) -> List[ComparisonOutput]:
        batches = batchify(data)
        out = []
        for b in batches:
            pooled_output_a = torch.Tensor(
                np.array([i["embedding_1"] for i in b])
            )
            pooled_output_b = torch.Tensor(
                np.array([i["embedding_2"] for i in b])
            )
            preds = self._forward(pooled_output_a, pooled_output_b)
            for i, p in zip(b, preds):
                out.append(
                    {
                        "region_1": i["region_1"],
                        "region_2": i["region_2"],
                        "similarity": p,
                    }
                )
        return out

    def _forward(
        self, pooled_output_a: torch.Tensor, pooled_output_b: torch.Tensor
    ) -> np.array:
        if isinstance(self.gpu_id, int):
            pooled_output_a = pooled_output_a.to(f"cuda:{self.gpu_id}")
            pooled_output_b = pooled_output_b.to(f"cuda:{self.gpu_id}")
        preds = self.head(pooled_output_a, pooled_output_b)
        preds = preds.cpu().detach().numpy()
        preds = self.softmax(preds)
        return [
            {
                self.class_dict[idx]: round(float(score), 2)
                for idx, score in enumerate(i)
            }
            for i in preds
        ]

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.exp(x).sum(-1, keepdims=True)
