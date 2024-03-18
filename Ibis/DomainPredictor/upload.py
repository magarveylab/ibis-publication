from typing import List, TypedDict

from tqdm import tqdm

from Ibis.Utilities.knowledge_graph import batchify, run_cypher, stringfy_dicts


class DomainDict(TypedDict):
    protein_start: int
    protein_stop: int
    score: float
    label: str


class OrfDict(TypedDict):
    contig_id: int
    protein_id: int
    contig_start: int
    contig_stop: int
    domains: List[DomainDict]


target_domains = ["A", "AT", "KS", "KR", "DH", "ER", "T"]


def upload_domains(orfs: List[OrfDict], orfs_uploaded: bool, bs: int = 1000):
    orf_to_domain_rels = []
    domain_to_label_rels = []
    domains_to_submit = []
    domain_annos_to_submit = []
    for orf in orfs:
        contig_id = orf["contig_id"]
        contig_start = orf["contig_start"]
        contig_stop = orf["contig_stop"]
        orf_id = f"{contig_id}_{contig_start}_{contig_stop}"
        protein_id = orf["protein_id"]
        for domain in orf["domains"]:
            protein_start = domain["protein_start"]
            protein_stop = domain["protein_stop"]
            domain_id = f"{protein_id}_{protein_start}_{protein_stop}"
            domain_label = domain["label"]
            hash_id = domain["domain_id"]
            domains_to_submit.append(
                {
                    "domain_id": domain_id,
                    "protein_start": protein_start,
                    "protein_stop": protein_stop,
                    "score": domain["score"],
                }
            )
            domain_to_label_rels.append(
                {"domain_id": domain_id, "label": domain_label}
            )
            orf_to_domain_rels.append(
                {"orf_id": orf_id, "domain_id": domain_id}
            )
            if domain_label in target_domains:
                domain_annos_to_submit.append(
                    {"domain_id": domain_id, "hash_id": hash_id}
                )
    # add domains
    batches = batchify(domains_to_submit, bs=bs)
    for batch in tqdm(batches, desc="Uploading domains"):
        batch_str = stringfy_dicts(
            batch, keys=["domain_id", "protein_start", "protein_stop", "score"]
        )
        run_cypher(
            f"""
            UNWIND {batch_str} as row
            MERGE (n: Domain {{domain_id: row.domain_id}})
            ON CREATE
                SET n.protein_start = row.protein_start,
                    n.protein_stop = row.protein_stop,
                    n.score = row.score
            ON MATCH
                SET n.protein_start = row.protein_start,
                    n.protein_stop = row.protein_stop,
                    n.score = row.score
        """
        )
    # add orf to domain rels
    if orfs_uploaded:
        batches = batchify(orf_to_domain_rels, bs=bs)
        for batch in tqdm(batches, desc="Uploading orf to domain rels"):
            batch_str = stringfy_dicts(batch, keys=["orf_id", "domain_id"])
            run_cypher(
                f"""
                UNWIND {batch_str} as row
                MATCH (n: Orf {{orf_id: row.orf_id}}),
                      (m: Domain {{domain_id: row.domain_id}})
                MERGE (n)-[r: orf_to_domain]->(m)
            """
            )
    # add domain to label rels
    batches = batchify(domain_to_label_rels, bs=bs)
    for batch in tqdm(batches, desc="Uploading domain to label rels"):
        batch_str = stringfy_dicts(batch, keys=["domain_id", "label"])
        run_cypher(
            f"""
            UNWIND {batch_str} as row
            MATCH (n: Domain {{domain_id: row.domain_id}}),
                  (m: DomainLabel {{label: row.label}})
            MERGE (n)-[r: domain_to_label]->(m)
        """
        )
    # add domain annotations
    batches = batchify(domain_annos_to_submit, bs=bs)
    for batch in tqdm(batches, desc="Uploading domain annotations"):
        batch_str = stringfy_dicts(batch, keys=["hash_id"])
        run_cypher(
            f"""
            UNWIND {batch_str} as row
            MERGE (n: DomainAnnotation {{hash_id: row.hash_id}})
            ON CREATE
                SET n.date = date(),
                    n.ran_substrate_knn = False,
                    n.ran_subclass_knn = False,
                    n.ran_functional_knn = False,
            ON MATCH
                SET n.date = date(),
                    n.ran_substrate_knn = False,
                    n.ran_subclass_knn = False,
                    n.ran_functional_knn = False,
        """
        )

    # link domain annotations to domains
