from typing import Dict, List

import more_itertools as mit
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from Ibis.SecondaryMetabolismPredictor.datastructs import OrfInput


def sort_orfs_by_contigs(orfs: List[OrfInput]) -> Dict[int, List[OrfInput]]:
    contigs_to_orfs = {o["contig_id"]: [] for o in orfs}
    for o in orfs:
        contig_id = o["contig_id"]
        contigs_to_orfs[contig_id].append(o)
    for contig_id in contigs_to_orfs:
        contigs_to_orfs[contig_id] = sorted(
            contigs_to_orfs[contig_id], key=lambda x: x["contig_start"]
        )
    return contigs_to_orfs


def get_orf_graphs_from_genome(
    orfs: List[OrfInput], tolerance: int = 10000
) -> Dict[int, nx.Graph]:
    contig_to_G = {}
    contig_to_orfs = sort_orfs_by_contigs(orfs)
    # draw edges between orfs that meet distance threshold
    for contig_id, contig_orfs in tqdm(
        contig_to_orfs.items(), leave=False, desc="Preparing Orf Graphs"
    ):
        # build graph per graph
        G = nx.Graph()
        contig_to_G[contig_id] = G
        for o in contig_orfs:
            G.add_node(
                o["orf_id"],
                contig_start=o["contig_start"],
                embedding=o["embedding"],
            )
        # add edges to capture spatial orientation
        for idx, o1 in enumerate(contig_orfs):
            for o2 in contig_orfs[idx + 1 :]:
                dist = o2["contig_start"] - o1["contig_stop"]
                if dist > tolerance:
                    break
                weight = round((tolerance - dist) / tolerance, 2)
                contig_to_G[contig_id].add_edge(
                    o1["orf_id"], o2["orf_id"], weight=weight
                )
    return contig_to_G


def get_tensor_from_graph(G: nx.Graph):
    num_nodes = len(G.nodes)
    num_edges = len(G.edges)
    # lookup dictionaries to form matrixes
    orf_id_to_node_idx = {}
    node_idx_to_orf_id = {}
    # note that the first node corresponds to class token
    for node_idx, orf_id in enumerate(G.nodes):
        orf_id_to_node_idx[orf_id] = node_idx
        node_idx_to_orf_id[node_idx] = orf_id
    # create empty numpy matrixes
    x = np.empty((num_nodes, 1024))
    ids = np.zeros((num_nodes, 1))
    edge_index = np.zeros((2, num_edges), dtype=int)
    edge_attr = np.empty((num_edges, 1))
    # populate x and ids
    for idx in range(num_nodes):
        orf_id = node_idx_to_orf_id[idx]
        x[idx] = G.nodes[orf_id]["embedding"]
        ids[idx][0] = orf_id
    # populate edge_index
    for edge_idx, (o1, o2, prop) in enumerate(G.edges(data=True)):
        n1 = orf_id_to_node_idx[o1]
        n2 = orf_id_to_node_idx[o2]
        edge_index[0][edge_idx] = n1
        edge_index[1][edge_idx] = n2
        edge_attr[edge_idx][0] = prop["weight"]  # spatial distance
    # pytorch datapoint
    datapoint = Data(
        x=torch.Tensor(x),
        ids=torch.LongTensor(ids),
        edge_index=torch.LongTensor(edge_index),
        edge_attr=torch.Tensor(edge_attr),
    )
    return datapoint


def windowify(
    l: List[int], size: int = 200, step: int = 75
) -> List[List[int]]:
    return [
        [n for n in w if n != None] for w in mit.windowed(l, n=size, step=step)
    ]


def get_tensors_from_genome(
    orfs: List[OrfInput],
    tolerance: int = 10000,
    window_size: int = 200,
    window_step: int = 75,
) -> List[Data]:
    out = []
    graphs = get_orf_graphs_from_genome(orfs, tolerance=tolerance)
    for contig_id, G in tqdm(
        graphs.items(), leave=False, desc="Preparing Tensors"
    ):
        # sort orfs by nucleotide position
        nodes = sorted(G.nodes, key=lambda n: G.nodes[n]["contig_start"])
        # window graph
        windows = windowify(nodes, size=window_size, step=window_step)
        for w_nodes in windows:
            # render subgraph
            subG = G.subgraph(w_nodes)
            # render tensor
            t = get_tensor_from_graph(subG)
            out.append(t)
    return out
