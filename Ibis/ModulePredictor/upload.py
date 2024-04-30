from typing import List, TypedDict

import pandas as pd
from tqdm import tqdm

from Ibis import curdir
from Ibis.Utilities.knowledge_graph import batchify, run_cypher, stringfy_dicts


class TagDict(TypedDict):
    tag_id: str
    rank: int


class ModuleDict(TypedDict):
    module_id: str
    protein_start: int
    protein_stop: int
    orfs: List[str]
    domains: List[str]
    tags: List[TagDict]
    adjacency_modules: List[str]


def load_tag_lookup():
    # load tag lookup
    df = pd.read_csv(f"{curdir}/ModulePredictor/dat/module_tags.csv")
    return dict(zip(df.name, df.tag_id))


tag_lookup = load_tag_lookup()


def upload_modules(modules: List[ModuleDict]):
    if len(modules) == 0:
        return True
    # upload modules
    batches = batchify(modules, bs=1000)
    for batch in tqdm(
        batches,
        desc="Uploading module predictions",
        leave=False,
    ):
        batch_str = stringfy_dicts(
            batch,
            keys=[
                "module_id",
                "protein_start",
                "protein_stop",
            ],
        )
        run_cypher(
            f"""
            UNWIND {batch_str} as row
            MERGE (m: Module {{module_id: row.module_id}})
            ON CREATE
                SET
                    m.protein_start = row.protein_start,
                    m.protein_stop = row.protein_stop
            """
        )
    # prepare relationships for upload
    module_orf_rels = []
    module_domain_rels = []
    modules_adjacency_rels = []
    module_tag_rels = []
    for m in modules:
        module_id = m["module_id"]
        module_orf_rels.extend(
            [{"module_id": module_id, "orf_id": o} for o in m["orfs"]]
        )
        module_domain_rels.extend(
            [{"module_id": module_id, "domain_id": d} for d in m["domains"]]
        )
        modules_adjacency_rels.extend(
            [
                {"module_id": module_id, "adjacency_module_id": a}
                for a in m["adjacency_modules"]
            ]
        )
        module_tag_rels.extend(
            [
                {
                    "module_id": module_id,
                    "tag_id": t["tag_id"],
                    "rank": t["rank"],
                }
                for t in m["tags"]
            ]
        )
    # upload relationships between modules and orfs
    batches = batchify(module_orf_rels, bs=1000)
    for batch in tqdm(
        batches,
        desc="Uploading module-orf relationships",
        leave=False,
    ):
        batch_str = stringfy_dicts(
            batch,
            keys=[
                "module_id",
                "orf_id",
            ],
        )
        run_cypher(
            f"""
            UNWIND {batch_str} as row
            MATCH (o: Orf {{orf_id: row.orf_id}}),
                  (m: Module {{module_id: row.module_id}})
            MERGE (o)-[:orf_to_module]->(m)
            """
        )
    # upload relationships between modules and domains
    batches = batchify(module_domain_rels, bs=1000)
    for batch in tqdm(
        batches,
        desc="Uploading module-domain relationships",
        leave=False,
    ):
        batch_str = stringfy_dicts(
            batch,
            keys=[
                "module_id",
                "domain_id",
            ],
        )
        run_cypher(
            f"""
            UNWIND {batch_str} as row
            MATCH (m: Module {{module_id: row.module_id}}),
                  (d: Domain {{domain_id: row.domain_id}})
            MERGE (m)-[:module_to_domain]->(d)
            """
        )
    # upload relationships between modules and adjacent modules
    batches = batchify(modules_adjacency_rels, bs=1000)
    for batch in tqdm(
        batches,
        desc="Uploading module-adjacency relationships",
        leave=False,
    ):
        batch_str = stringfy_dicts(
            batch,
            keys=[
                "module_id",
                "adjacency_module_id",
            ],
        )
        run_cypher(
            f"""
            UNWIND {batch_str} as row
            MATCH (m: Module {{module_id: row.module_id}}),
                  (a: Module {{module_id: row.adjacency_module_id}})
            MERGE (m)-[:module_to_adjacency]->(a)
            """
        )
    # upload relationships between modules and tags
    batches = batchify(module_tag_rels, bs=1000)
    for batch in tqdm(
        batches,
        desc="Uploading module-tag relationships",
        leave=False,
    ):
        batch_str = stringfy_dicts(
            batch,
            keys=[
                "module_id",
                "tag_id",
                "rank",
            ],
        )
        run_cypher(
            f"""
            UNWIND {batch_str} as row
            MATCH (m: Module {{module_id: row.module_id}}),
                  (t: ModuleTag {{tag_id: row.tag_id}})
            MERGE (m)-[r:module_to_tag]->(t)
            ON CREATE
                SET r.rank = row.rank,
                    r.date = date()
            ON MATCH
                SET r.rank = row.rank,
                    r.date = date()
            """
        )
    return True
