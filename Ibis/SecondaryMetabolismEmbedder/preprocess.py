from Ibis.SecondaryMetabolismEmbedder.datastructs import ClusterInput
from Ibis.Utilities.GraphStructs.HeteroGraph import HeteroGraph


class BGCGraph(HeteroGraph):

    def __init__(self):
        # define schema
        schema = {}
        schema["node_types"] = [
            "metabolite",
            "mibig_chemotype",
            "internal_chemotype",
            "orf",
            "domain",
            "domain_embedding",
        ]
        schema["edge_types"] = [
            ("orf", "orf_to_orf", "orf"),
            ("orf", "orf_to_domain", "domain"),
            ("domain", "domain_to_domain", "domain"),
            ("domain", "domain_to_domain_embedding", "domain_embedding"),
        ]
        schema["node_embedding_dim"] = {"domain_embedding": 1024, "orf": 1024}
        schema["edge_embedding_dim"] = {}
        super().__init__(schema=schema)

    def add_orf_to_orf_edges(self):
        orf_nodes = self.get_nodes_from(node_type="orf")
        orf_nodes = sorted(orf_nodes, key=lambda x: self[x]["meta"]["start"])
        for n1, n2 in zip(orf_nodes, orf_nodes[1:]):
            self.add_edge(n1=n1, n2=n2, edge_type=("orf", "orf_to_orf", "orf"))

    def add_domain_to_domain_edges(self):
        domain_nodes = self.get_nodes_from(node_type="domain")
        domain_nodes = sorted(
            domain_nodes,
            key=lambda x: (
                self[x]["meta"]["orf_node_id"],
                self[x]["meta"]["start"],
            ),
        )
        for n1, n2 in zip(domain_nodes, domain_nodes[1:]):
            if (
                self[n1]["meta"]["orf_node_id"]
                == self[n2]["meta"]["orf_node_id"]
            ):
                self.add_edge(
                    n1=n1,
                    n2=n2,
                    edge_type=("domain", "domain_to_domain", "domain"),
                )

    @classmethod
    def build_from(cls, data: ClusterInput):
        G = cls()
        # add cluster nodes (this can be either cluster or molecule)
        # treat this as virtual node
        cluster_node_id = G.add_node(node_type="metabolite", label="bgc")
        # add chemotype nodes
        for chemotype in data["mibig_chemotypes"]:
            mibig_chemotype_node_id = G.add_node(
                node_type="mibig_chemotype", label=chemotype
            )
        for chemotype in data["internal_chemotypes"]:
            internal_chemotype_node_id = G.add_node(
                node_type="internal_chemotype", label=chemotype
            )
        # add orf nodes
        for orf in data["orfs"]:
            orf_meta = {"start": orf["contig_start"]}
            orf_embedding = orf["embedding"]
            orf_node_id = G.add_node(
                node_type="orf", embedding=orf_embedding, meta=orf_meta
            )
            # add domain nodes
            for domain in orf["domains"]:
                domain_meta = {
                    "orf_node_id": orf_node_id,
                    "start": domain["protein_start"],
                }
                domain_node_id = G.add_node(
                    node_type="domain", label=domain["label"], meta=domain_meta
                )
                domain_embedding = domain["embedding"]
                G.add_edge(
                    n1=orf_node_id,
                    n2=domain_node_id,
                    edge_type=("orf", "orf_to_domain", "domain"),
                )
                if domain_embedding is not None:
                    domain_embedding_id = G.add_node(
                        node_type="domain_embedding",
                        embedding=domain_embedding,
                    )
                    G.add_edge(
                        n1=domain_node_id,
                        n2=domain_embedding_id,
                        edge_type=(
                            "domain",
                            "domain_to_domain_embedding",
                            "domain_embedding",
                        ),
                    )
        G.add_orf_to_orf_edges()
        G.add_domain_to_domain_edges()
        return G
