from Ibis.SecondaryMetabolismPredictor.datastructs import (
    OrfInput,
    InternalAnnotatedOrfDict,
    InternalAnnotatedOrfDictWithMeta,
    MibigAnnotatedOrfDict,
    MibigAnnotatedOrfDictWithMeta,
)
from Ibis.SecondaryMetabolismPredictor.preprocess import (
    get_tensors_from_genome,
)
from Ibis.Utilities.class_dicts import get_class_dict
from Ibis import curdir
from torch_geometric.data import Data, Batch
from tqdm import tqdm
from glob import glob
import torch
from typing import Optional, List, Union


def batchify(l, bs=10):
    return [l[x : x + bs] for x in range(0, len(l), bs)]


def add_meta_data_to_output(
    out: List[Union[InternalAnnotatedOrfDict, MibigAnnotatedOrfDict]],
    orfs: List[OrfInput],
) -> List[
    Union[InternalAnnotatedOrfDictWithMeta, MibigAnnotatedOrfDictWithMeta]
]:
    orf_meta = {
        o["orf_id"]: {
            "contig_start": o["contig_start"],
            "contig_stop": o["contig_stop"],
            "contig_id": o["contig_id"],
        }
        for o in orfs
    }
    [o.update(orf_meta[o["orf_id"]]) for o in out]
    return out


class MibigMetabolismPredictorPipeline:

    def __init__(
        self,
        model_dir: str = f"{curdir}/Models/mibig_metabolism_predictor",
        gpu_id: Optional[int] = None,
    ):
        # load models (torchscript format)
        self.node_encoder = torch.jit.load(f"{model_dir}/node_encoder.pt")
        self.gnn = torch.jit.load(f"{model_dir}/gnn.pt")
        self.transformer = torch.jit.load(f"{model_dir}/transformer.pt")
        self.heads = {}
        for head_fp in glob(f"{model_dir}/heads/*.pt"):
            head_name = head_fp.split("/")[-1].split(".")[0]
            self.heads[head_name] = torch.jit.load(head_fp)
        self.all_chemotypes = sorted(self.heads.keys())
        # mode models to gpu
        self.gpu_id = gpu_id
        if isinstance(self.gpu_id, int):
            self.node_encoder.to(f"cuda:{self.gpu_id}")
            self.gnn.to(f"cuda:{self.gpu_id}")
            self.transformer.to(f"cuda:{self.gpu_id}")
            for head_name, head in self.heads.items():
                head.to(f"cuda:{self.gpu_id}")

    def __call__(
        self,
        orfs: List[OrfInput] = [],
        batched_data: Optional[List[Data]] = None,
    ) -> List[MibigAnnotatedOrfDictWithMeta]:
        if batched_data == None:
            batched_data = self.preprocess(orfs)
        preds = self._forward(batched_data)
        out = self.postprocess(preds)
        out = add_meta_data_to_output(out, orfs)
        return out

    def preprocess(self, orfs: List[OrfInput]) -> List[Data]:
        return get_tensors_from_genome(orfs)

    def _forward(self, data_list: List[Data]) -> Batch:
        out = []
        batches = batchify(data_list)
        for data in tqdm(batches, desc="Running model on data batches"):
            data = Batch.from_data_list(data)
            if isinstance(self.gpu_id, int):
                data = data.to(f"cuda:{self.gpu_id}")
            # preprocess node and edge encoding
            data.x = self.node_encoder(data.x)
            # message passing
            data.x = self.gnn(data.x, data.edge_index, data.edge_attr)
            # transformer (global attention accross nodes)
            data.x = self.transformer(data.x, data.batch)
            # heads (single label node classification)
            for head_name, head in self.heads.items():
                setattr(data, head_name, torch.softmax(head(data.x), dim=1))
            # detach from gpu memory
            if isinstance(self.gpu_id, int):
                data = data.detach()
            out.append(data)
        # batch output data object
        return Batch.from_data_list(out)

    def postprocess(self, data: Batch) -> List[MibigAnnotatedOrfDict]:
        orf_to_preds = {}
        length = data.x.shape[0]
        # parse through all the predictions
        for idx in tqdm(
            range(length), total=length, desc="Reorganizing Predictions"
        ):
            orf_id = int(data.ids[idx][0])
            if orf_id == -1:
                continue
            if orf_id not in orf_to_preds:
                orf_to_preds[orf_id] = {"orf_id": orf_id}
                for c in self.all_chemotypes:
                    orf_to_preds[orf_id][c] = []
            # add chemotype predictions
            for c in self.all_chemotypes:
                score = round(float(getattr(data, c)[idx][1]), 2)
                if score >= 0.5:
                    orf_to_preds[orf_id][c].append(score)
        # collapse predictions (by maximum)
        output = []
        for orf_id, pred in orf_to_preds.items():
            row = {"orf_id": orf_id, "chemotypes": []}
            for c in self.all_chemotypes:
                if len(pred[c]) == 0:
                    continue
                row["chemotypes"].append({"label": c, "score": max(pred[c])})
            output.append(row)
        return output


class InternalMetabolismPredictorPipeline:

    def __init__(
        self,
        model_dir: str = f"{curdir}/Models/internal_metabolism_predictor",
        class_dict_fp: str = f"{curdir}/SecondaryMetabolismPredictor/tables/chemotypes.csv",
    ):
        # class dicts
        self.chemotype_class_dict = get_class_dict(class_dict_fp)
        # load models (torchscript format)
        self.node_encoder = torch.jit.load(f"{model_dir}/node_encoder.pt")
        self.gnn = torch.jit.load(f"{model_dir}/gnn.pt")
        self.transformer = torch.jit.load(f"{model_dir}/transformer.pt")
        self.secondary_head = torch.jit.load(f"{model_dir}/secondary_head.pt")
        self.chemotype_head = torch.jit.load(f"{model_dir}/chemotype_head.pt")
        # move models to gpu
        self.gpu_id = gpu_id
        if isinstance(self.gpu_id, int):
            self.node_encoder.to(f"cuda:{self.gpu_id}")
            self.gnn.to(f"cuda:{self.gpu_id}")
            self.transformer.to(f"cuda:{self.gpu_id}")
            self.secondary_head.to(f"cuda:{self.gpu_id}")
            self.chemotype_head.to(f"cuda:{self.gpu_id}")

    def __call__(
        self,
        orfs: List[OrfInput] = [],
        batched_data: Optional[List[Data]] = None,
    ) -> List[InternalAnnotatedOrfDictWithMeta]:
        if batched_data == None:
            batched_data = self.preprocess(orfs)
        preds = self._forward(batched_data)
        out = self.postprocess(preds)
        out = add_meta_data_to_output(out, orfs)
        return out

    def preprocess(self, orfs: List[OrfInput]) -> List[Data]:
        return get_tensors_from_genome(orfs)

    def _forward(self, data_list: List[Data]) -> Batch:
        out = []
        batches = batchify(data_list)
        for data in tqdm(batches, desc="Running model on data batches"):
            data = Batch.from_data_list(data)
            if isinstance(self.gpu_id, int):
                data = data.to(f"cuda:{self.gpu_id}")
            # preprocess node and edge encoding
            data.x = self.node_encoder(data.x)
            # message passing
            data.x = self.gnn(data.x, data.edge_index, data.edge_attr)
            # transformer (global attention accross nodes)
            data.x = self.transformer(data.x, data.batch)
            # heads
            # secondary - multi label node classification
            data.secondary = torch.sigmoid(self.secondary_head(data.x))
            # chemotype - single label node classification
            data.chemotype = torch.softmax(self.chemotype_head(data.x), dim=1)
            # detach from gpu memory
            if isinstance(self.gpu_id, int):
                data = data.detach()
            out.append(data)
        # batch output data object
        return Batch.from_data_list(out)

    def postprocess(self, data: Batch) -> List[InternalAnnotatedOrfDict]:
        orf_to_preds = {}
        length = data.x.shape[0]
        # parse through all the predictions
        for idx in tqdm(
            range(length), total=length, desc="Reorganizing Predictions"
        ):
            orf_id = int(data.ids[idx][0])
            if orf_id == -1:
                continue
            if orf_id not in orf_to_preds:
                orf_to_preds[orf_id] = {
                    "orf_id": orf_id,
                    "secondary": [],
                    "chemotype": [],
                }
            # add secondary label (note this training task was setup as binary classification)
            score = round(float(data.secondary[idx][0]), 2)
            label = "core" if score >= 0.5 else "peripheral"
            # score needs to be adjusted to reflect probability for predicted label
            score = 1 - score if label == "peripheral" else score
            orf_to_preds[orf_id]["secondary"].append(
                {"label": label, "score": round(score, 2)}
            )
            # add chemotype label
            top_idx = int(torch.argmax(data.chemotype[idx]))
            orf_to_preds[orf_id]["chemotype"].append(
                {
                    "label": self.chemotype_class_dict[top_idx],
                    "score": round(float(data.chemotype[idx][top_idx]), 2),
                }
            )
        # collapse predictions (by maximum)
        # if predicted as peripheral, ignore chemotype prediction
        for orf_id, pred in orf_to_preds.items():
            orf_to_preds[orf_id]["secondary"] = max(
                orf_to_preds[orf_id]["secondary"], key=lambda x: x["score"]
            )
            if orf_to_preds[orf_id]["secondary"]["label"] == "peripheral":
                orf_to_preds[orf_id]["chemotype"] = None
            else:
                orf_to_preds[orf_id]["chemotype"] = max(
                    orf_to_preds[orf_id]["chemotype"], key=lambda x: x["score"]
                )
        return list(orf_to_preds.values())
