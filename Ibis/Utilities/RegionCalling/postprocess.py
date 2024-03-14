from Ibis.Utilities.RegionCalling.datastructs import (
    TokenOutput,
    TokenRegionOutput,
)
from multiprocessing import Pool
import networkx as nx
import numpy as np
from collections import Counter
from typing import List


class TokenGraph:

    def __init__(self):
        self.G = nx.Graph()

    def __getitem__(self, key) -> dict:
        return self.G.nodes[key]

    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def nodes(self) -> Set[int]:
        return self.G.nodes

    @property
    def nodes_with_labels(self) -> List[int]:
        return sorted([n for n in self.nodes if self[n]["label"] != None])

    def add_node(
        self,
        pos: int,
        label: Optional[str] = None,
        score: Optional[float] = None,
    ):
        self.G.add_node(pos, label=label, score=score)

    def add_edge(self, n1: int, n2: int, weight: float):
        # weight = 1 for adjacent residues, = min(probability) for matching label
        self.G.add_edge(n1, n2, weight=weight)

    @property
    def louvain_communities(self):
        """Uses Louvain Communities for Label Correction
        1. Draw edges between neighbouring residues with a weight of 1
        2. Draw edges between residues with the same label
        with a weight of the minimum of the two confidence scores
        (Only if the residues are within n residues of each other)
        3. Run Louvain Communities on the graph
        """
        # It is CRITICAL to set the seed for louvain communities or you WILL get inconsistencies.
        # Please, please do NOT forget this.
        try:
            return nx.algorithms.community.louvain.louvain_communities(
                self.G, seed=42
            )
        except ZeroDivisionError:
            print(self.G.nodes, self.G.edges)
            return []

    def get_label_from_nodes(self, nodes: List[int]):
        labels = [self[n]["label"] for n in nodes if self[n]["label"] != None]
        if len(labels) == 0:
            return None
        else:
            labels_counted = Counter(labels)
            return max(set(labels), key=lambda x: labels_counted[x])


def token_region_calling(
    token_results: List[TokenOutput], min_nodes: int = 10, max_dist: int = 10
) -> TokenRegionOutput:
    out = []
    # sort residue results by position
    token_results = sorted(token_results, key=lambda x: x["pos"])
    # create graph
    token_graph = TokenGraph()
    min_pos = token_results[0]["pos"]
    max_pos = token_results[-1]["pos"]
    all_pos = list(range(min_pos, max_pos + 1))
    # add nodes
    [token_graph.add_node(n) for n in all_pos]
    # add edges between adjacent residues
    [
        token_graph.add_edge(n1, n2, 1)
        for n1, n2 in list(zip(all_pos, all_pos[1:]))
    ]
    # capture annotations that meet minimum probability cutoff
    for r in token_results:
        if len(r["labels"]) == 0:
            continue
        best_label = max(r["labels"], key=lambda x: x["score"])
        token_graph.add_node(
            r["pos"], label=best_label["label"], score=best_label["score"]
        )
    # draw connections between neraby positions sharing label
    nodes_to_examine = token_graph.nodes_with_labels
    for idx, n1 in enumerate(nodes_to_examine):
        for n2 in nodes_to_examine[idx + 1 :]:
            # early break if distance not met
            if n2 - n1 > max_dist:
                break
            # check if labels match
            if token_graph[n2]["label"] == token_graph[n1]["label"]:
                min_prob = min(
                    [token_graph[n2]["score"], token_graph[n1]["score"]]
                )
                token_graph.add_edge(n1, n2, weight=min_prob)
    # calculate communities
    regions = []
    communities = token_graph.louvain_communities
    for community in communities:
        community = list(community)
        community.sort()
        community_label = token_graph.get_label_from_nodes(community)
        if community_label == None:
            continue
        support = len(
            [
                n
                for n in community
                if token_graph[n]["label"] == community_label
            ]
        )
        if support < min_nodes:
            continue
        # cache
        start = min(community)
        end = max(community)
        regions.append((community_label, start, end))
    # sometimes there is adjavent duplication of regions - combine these regions
    region_graph = nx.Graph()
    regions = sorted(regions, key=lambda x: x[1])
    # add nodes
    region_graph.add_nodes_from(regions)
    # add edges
    for idx, r in enumerate(regions[:-1]):
        next_r = regions[idx + 1]
        if r[0] == next_r[0]:
            region_graph.add_edge(r, next_r)
    # combine regions
    for combined_r in nx.connected_components(region_graph):
        combined_r = sorted(combined_r, key=lambda x: x[1])
        label = combined_r[0][0]
        start = combined_r[0][1]
        end = combined_r[-1][2]
        nodes = list(range(start, end + 1))
        score_list = [
            token_graph[n]["score"]
            for n in nodes
            if token_graph[n]["label"] == label
        ]
        support = len(score_list)
        score = round(float(np.mean(score_list)), 2)
        length = end - start
        # cache
        out.append(
            {"label": label, "start": start, "end": end, "score": score}
        )
    return sorted(out, key=lambda x: x["start"])


def pipeline_token_region_calling(
    pipeline_output: PipelineIntermediateOutput,
) -> PipelineOutput:
    return {
        "domain_id": pipeline_output["domain_id"],
        "sequence": pipeline_output["sequence"],
        "regions": token_region_calling(
            pipeline_output["residue_classification"]
        ),
    }


def parallel_pipeline_token_region_calling(
    pipeline_outputs: List[PipelineIntermediateOutput],
    cpu_cores: int = 1,
) -> List[PipelineOutput]:
    pool = Pool(cpu_cores)
    process = pool.imap_unordered(
        pipeline_token_region_calling, pipeline_outputs
    )
    out = [p for p in tqdm(process, total=len(pipeline_outputs))]
    pool.close()
    return out
