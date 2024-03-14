from torch_geometric.data import HeteroData, Batch
import networkx as nx
import numpy as np
import torch
from typing import Set, List, Optional, Tuple, Dict

NodeLabel = str
NodeType = str
EdgeLabel = str
EdgeName = str
EdgeType = Tuple[NodeType, EdgeName, NodeType]
NodeVocab = Dict[NodeType, Dict[NodeLabel, int]]
EdgeVocab = Dict[EdgeType, Dict[EdgeLabel, int]]


class HeteroGraphSchema:
    # the order of node_type and edge_types should not be changed - as it determines indexes
    node_types: List[NodeType]
    node_embedding_dim: Dict[
        NodeType, int
    ]  # any nodes missing from this is assumed to be label
    edge_types: List[EdgeType]
    edge_embedding_dim: Dict[
        EdgeType, int
    ]  # any edges missing from this is assumed to be described with label


class HeteroGraph:

    def __init__(self, schema: HeteroGraphSchema):
        self.G = nx.DiGraph()
        self.node_types = schema["node_types"]
        self.node_embedding_dim = schema["node_embedding_dim"]
        self.edge_types = schema["edge_types"]
        self.edge_embedding_dim = schema["edge_embedding_dim"]
        self.edge_type_lookup = {
            et: idx for idx, et in enumerate(self.edge_types)
        }
        self.node_id = -1
        self.add_blank_graph()

    def __getitem__(self, key) -> dict:
        return (
            self.G.nodes[key]
            if isinstance(key, int)
            else self.G[key[0]][key[1]]
        )

    @property
    def nodes(self):
        return self.G.nodes

    def add_blank_graph(self):
        # necessary to create a consistent HeteroData objects for batching
        for nt in self.node_types:
            blank_embedding = (
                np.zeros(self.node_embedding_dim[nt])
                if nt in self.node_embedding_dim
                else None
            )
            self.add_node(
                node_type=nt,
                label="[BLANK]",
                embedding=blank_embedding,
                blank=True,
            )
        for et in self.edge_types:
            n1 = list(self.get_nodes_from(node_type=et[0], blank=True))[0]
            n2 = list(self.get_nodes_from(node_type=et[2], blank=True))[0]
            blank_embedding = (
                np.zeros(self.edge_embedding_dim[et])
                if et in self.edge_embedding_dim
                else None
            )
            self.add_edge(
                n1=n1,
                n2=n2,
                edge_type=et,
                label="[BLANK]",
                embedding=blank_embedding,
                blank=True,
            )

    def get_nodes_from(
        self, node_type: NodeType, blank: Optional[bool] = False
    ) -> Set[int]:
        if blank == None:  # returns all nodes
            return {n for n in self.nodes if self[n]["node_type"] == node_type}
        else:  # selective in real vs placeholder nodes
            return {
                n
                for n in self.nodes
                if self[n]["node_type"] == node_type
                and self[n]["blank"] == blank
            }

    def get_edges_from(self, edge_type: EdgeType) -> Set[Tuple[int, int]]:
        return {
            (n1, n2)
            for n1, n2, e in self.G.edges(data=True)
            if e["edge_type"] == edge_type
        }

    def add_node(
        self,
        node_type: NodeType,
        label: str = "[BLANK]",
        embedding: Optional[np.array] = None,
        meta: dict = {},
        blank: bool = False,
    ):
        if node_type in self.node_types:
            self.node_id += 1
            self.G.add_node(
                self.node_id,
                node_type=node_type,
                label=label,
                embedding=embedding,
                meta=meta,
                blank=blank,
            )
            return self.node_id

    def add_edge(
        self,
        n1: int,
        n2: int,
        edge_type: EdgeType,
        label: str = "[BLANK]",
        embedding: Optional[np.array] = None,
        blank: bool = False,
    ):
        if edge_type in self.edge_types:
            self.G.add_edge(
                n1,
                n2,
                edge_type=edge_type,
                label=label,
                embedding=embedding,
                bank=blank,
            )

    def print_summary(self):
        print("Node Types")
        for nt in self.node_types:
            print(
                f"{nt}: {len(self.get_nodes_from(node_type=nt, blank=None))}"
            )
        print("\nEdge Types")
        for et in self.edge_types:
            print(f"{et[1]}: {len(self.get_edges_from(edge_type=et))}")

    def get_tensor_data(
        self,
        node_vocab: NodeVocab = {},
        edge_vocab: EdgeVocab = {},
        apply_multigraph_wrapper: bool = True,
        node_types_to_consider: Optional[List[NodeType]] = None,
        edge_types_to_consider: Optional[List[EdgeType]] = None,
    ) -> HeteroData:
        if node_types_to_consider == None:
            node_types_to_consider = self.node_types
        if edge_types_to_consider == None:
            edge_types_to_consider = self.edge_types
        # create data tensor object
        data = HeteroData()
        node_index_map = {}
        # parse nodes
        for nt in node_types_to_consider:
            x = []
            node_index_map[nt] = {}
            nodes = sorted(self.get_nodes_from(node_type=nt, blank=None))
            if nt in node_vocab:
                for idx, n in enumerate(nodes):
                    label = self[n]["label"]
                    node_index_map[nt][n] = idx
                    x.append(
                        [node_vocab[nt].get(label, node_vocab[nt]["[UNK]"])]
                    )
                data[nt].x = torch.LongTensor(x)
            else:
                for idx, n in enumerate(nodes):
                    node_index_map[nt][n] = idx
                    x.append(self[n]["embedding"])
                data[nt].x = torch.Tensor(np.array(x))
        # parse edges
        for et in edge_types_to_consider:
            n1_name, edge_name, n2_name = et
            # prepare edge index, edge attr and additional features
            edge_index = []
            edge_attr = []
            extra_edge_attr = []
            if edge_name in edge_vocab:
                for n1, n2 in self.get_edges_from(edge_type=et):
                    edge_index.append(
                        [
                            node_index_map[n1_name][n1],
                            node_index_map[n2_name][n2],
                        ]
                    )
                    label = self[(n1, n2)]["label"]
                    edge_attr.append(
                        [
                            edge_vocab[edge_name].get(
                                label, edge_vocab[edge_name]["[UNK]"]
                            )
                        ]
                    )
                    if et in self.edge_embedding_dim:
                        extra_edge_attr.append(self[(n1, n2)]["embedding"])

                data[n1_name, edge_name, n2_name].edge_attr = torch.LongTensor(
                    edge_attr
                )
                if len(extra_edge_attr) > 0:
                    data[n1_name, edge_name, n2_name].extra_edge_attr = (
                        torch.Tensor(extra_edge_attr)
                    )
            else:
                for n1, n2 in self.get_edges_from(edge_type=et):
                    edge_index.append(
                        [
                            node_index_map[n1_name][n1],
                            node_index_map[n2_name][n2],
                        ]
                    )
                    edge_attr.append(self[(n1, n2)]["embedding"])
                data[n1_name, edge_name, n2_name].edge_attr = torch.Tensor(
                    np.array(edge_attr)
                )
            # prepare edge index
            edge_count = len(edge_index)
            edge_index = torch.LongTensor(edge_index)
            edge_index = torch.transpose(edge_index, 0, 1)
            data[n1_name, edge_name, n2_name].edge_index = edge_index
            # prepare edge type
            et_idx = self.edge_type_lookup[et]
            data[n1_name, edge_name, n2_name].edge_type = torch.LongTensor(
                [[et_idx]] * edge_count
            )
        return data


############################################################
# Helper Functions
############################################################


def get_lookup_from_hetero(h: HeteroData):
    # needed for looking up nodes and edges after converting homogenous to heterogenous
    lookup = {
        node_type: str(idx) for idx, node_type in enumerate(h.node_types)
    }
    for idx, edge_type in enumerate(h.edge_types):
        n1 = lookup[edge_type[0]]
        n2 = lookup[edge_type[2]]
        lookup[edge_type] = (n1, str(idx), n2)
    return lookup


def batch_to_homogeneous(
    batch: Batch, replace_nan_with: Optional[int] = -100, **kwargs
) -> Batch:
    data_list = batch.to_data_list()
    data_list = [d.to_homogeneous(**kwargs) for d in data_list]
    batch = Batch.from_data_list(data_list=data_list)
    for k, v in batch.to_dict().items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if torch.is_tensor(v2):
                    batch[k][k2] = torch.nan_to_num(v2, nan=replace_nan_with)
        elif torch.is_tensor(v):
            batch[k] = torch.nan_to_num(v, nan=replace_nan_with)
    return batch
