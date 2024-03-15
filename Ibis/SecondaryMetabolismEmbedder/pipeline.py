from Ibis.SecondaryMetabolismEmbedder.datastructs import (
    ClusterInput,
    ClusterEmbeddingOutput,
)
from Ibis.SecondaryMetabolismEmbedder.preprocess import BGCGraph
from Ibis.Utilities.GraphStructs.HeteroGraph import (
    get_lookup_from_hetero,
    batch_to_homogeneous,
)
from Ibis import curdir
from torch_geometric.data import Data, Batch
from glob import glob
import numpy as np
import torch
import json
from typing import Optional, Dict

vocab_dir = f"{curdir}/SecondaryMetabolismEmbedder/vocab"
node_vocab = json.load(open(f"{vocab_dir}/node_vocab.json"))
edge_vocab = json.load(open(f"{vocab_dir}/edge_vocab.json"))


class MetabolismEmbedderPipeline:

    def __init__(
        self,
        model_dir: str = f"{curdir}/Models/metabolism_embedder",
        node_vocab: Dict[str, int] = node_vocab,
        edge_vocab: Dict[str, int] = edge_vocab,
        gpu_id: Optional[int] = None,
    ):
        # vocab
        self.node_vocab = node_vocab
        self.edge_vocab = edge_vocab
        self.node_types_with_embedding = ["domain_embedding", "orf"]
        self.edge_type_lookup = {
            "orf_to_orf": ("orf", "orf_to_orf", "orf"),
            "orf_to_domain": ("orf", "orf_to_domain", "domain"),
            "domain_to_domain": ("domain", "domain_to_domain", "domain"),
            "domain_to_domain_embedding": (
                "domain",
                "domain_to_domain_embedding",
                "domain_embedding",
            ),
        }
        # load node encoders
        self.node_encoders = {}
        for model_fp in glob(f"{model_dir}/node_encoders/*.pt"):
            node_type = model_fp.split("/")[-1].split("_node_encoder")[0]
            self.node_encoders[node_type] = torch.jit.load(model_fp)
        # load edge encoders
        self.edge_encoders = {}
        for model_fp in glob(f"{model_dir}/edge_encoders/*.pt"):
            edge_type = model_fp.split("/")[-1].split("_edge_encoder")[0]
            self.edge_encoders[edge_type] = torch.jit.load(model_fp)
        # load edge type encoder
        self.edge_type_encoder = torch.jit.load(
            f"{model_dir}/edge_type_encoder.pt"
        )
        # load message passing nn
        self.gnn = torch.jit.load(f"{model_dir}/gnn.pt")
        # load transformer
        self.transformer = torch.jit.load(f"{model_dir}/transformer.pt")
        # load graph pooler
        self.graph_pooler = torch.jit.load(f"{model_dir}/graph_pooler.pt")
        # move models to gpu (if device defined)
        self.gpu_id = gpu_id
        if isinstance(self.gpu_id, int):
            for node_type in self.node_encoders:
                self.node_encoders[node_type].to(f"cuda:{self.gpu_id}")
            for edge_type in self.edge_encoders:
                self.edge_encoders[edge_type].to(f"cuda:{self.gpu_id}")
            self.edge_type_encoder.to(f"cuda:{self.gpu_id}")
            self.gnn.to(f"cuda:{self.gpu_id}")
            self.transformer.to(f"cuda:{self.gpu_id}")
            self.graph_pooler.to(f"cuda:{self.gpu_id}")

    def __call__(self, report: ClusterInput) -> ClusterEmbeddingOutput:
        contig_id = report["contig_id"]
        contig_start = report["contig_start"]
        contig_stop = report["contig_stop"]
        data = self.preprocess(report)
        embedding = self._forward(data)
        return {
            "contig_id": contig_id,
            "contig_start": contig_start,
            "contig_stop": contig_stop,
            "embedding": embedding,
        }

    def preprocess(self, report: ClusterInput) -> Batch:
        G = BGCGraph.build_from(data=report)
        data = G.get_tensor_data(
            node_vocab=self.node_vocab, edge_vocab=self.edge_vocab
        )
        data = Batch.from_data_list([data])
        return data

    def _forward(self, data: Batch) -> np.array:
        if isinstance(self.gpu_id, int):
            data = data.to(f"cuda:{self.gpu_id}")
        # preprocess node encoding (all types should be converted to same dimensionality)
        for node_type, node_encoder in self.node_encoders.items():
            if node_type in self.node_types_with_embedding:
                data[node_type]["x"] = node_encoder(data[node_type]["x"])
            else:  # label
                data[node_type]["x"] = node_encoder(
                    data[node_type]["x"], data[node_type].get("extra_x", None)
                )
        # preprocess edge encoding
        for edge_name, edge_encoder in self.edge_encoders.items():
            edge_type = self.edge_type_lookup[edge_name]
            data[edge_type]["edge_attr"] = edge_encoder(
                data[edge_type]["edge_attr"],
                data[edge_type].get("extra_edge_attr", None),
            )
        # convert heterogenous to homogenous
        lookup = get_lookup_from_hetero(data)
        data = batch_to_homogeneous(data)
        # edge encode by edge type
        data.edge_attr = self.edge_type_encoder(
            data.edge_type, getattr(data, "edge_attr", None)
        )
        # message passing
        data.x = self.gnn(data.x, data.edge_index, data.edge_attr)
        # transformer (global attention accross nodes)
        data.x = self.transformer(data.x, data.batch)
        # convert homogenous to heterogenous
        data = data.to_heterogeneous()
        # get pooled output
        pooled_output = self.graph_pooler(
            data[lookup[self.graph_pooler.node_type]]["x"],
            data[lookup[self.graph_pooler.node_type]]["batch"],
        )
        # move pooled output to cpu memory
        pooled_output = pooled_output.cpu().detach().numpy()[0]
        return pooled_output
