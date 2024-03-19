from typing import List, Optional, TypedDict

from tqdm import tqdm

from Ibis.Utilities.knowledge_graph import batchify, run_cypher, stringfy_dicts


class BGCDict(TypedDict):
    contig_id: int
    hash_id: int
    contig_start: int
    contig_stop: int
    orfs: List[str]  # {contig_id}_{contig_start}_{contig_stop}
    internal_chemotypes: List[str]
    mibig_chemotypes: List[str]


def upload_bgcs(
    bgcs: List[BGCDict],
    genome_id: Optional[int],
    orfs_uploaded: bool,
    contigs_uploaded: bool,
    genome_uploaded: bool,
    bs: int = 1000,
    source: str = "internal",
) -> bool:
    if len(bgcs) == 0:
        return False
    # add region_ids
    for c in bgcs:
        contig_id = c["contig_id"]
        contig_start = c["contig_start"]
        contig_stop = c["contig_stop"]
        c["region_id"] = f"{contig_id}_{contig_start}_{contig_stop}"
        c["source"] = source
    # add bgcs
    batches = batchify(bgcs, bs=bs)
    for batch in tqdm(batches, desc="Uploading BGCs"):
        batch_str = stringfy_dicts(
            batch,
            keys=[
                "region_id",
                "hash_id",
                "contig_start",
                "contig_stop",
                "source",
            ],
        )
        run_cypher(
            f"""
            UNWIND {batch_str} as row
            MERGE (n: MetabolomicRegion {{region_id: row.region_id}})
            ON CREATE
                SET n.hash_id = row.hash_id,
                    n.contig_start = row.contig_start,
                    n.contig_stop = row.contig_stop,
                    n.source = row.source
        """
        )
    # connect contigs to bgcs
    if contigs_uploaded:
        rels = [
            {"region_id": c["region_id"], "contig_id": c["contig_id"]}
            for c in bgcs
        ]
        if len(rels) > 0:
            batches = batchify(rels, bs=bs)
            for batch in tqdm(
                batches, desc="Adding relationships between contigs and BGCs"
            ):
                batch_str = stringfy_dicts(
                    batch, keys=["region_id", "contig_id"]
                )
                run_cypher(
                    f"""
                    UNWIND {batch_str} as row
                    MATCH (n: MetabolomicRegion {{region_id: row.region_id}}),
                        (m: Contig {{contig_id: row.contig_id}})
                    MERGE (n)-[:metab_to_contig]->(m)
                """
                )
    # connect bgc to internal chemotypes
    rels = [
        {"region_id": c["region_id"], "label": ch}
        for c in bgcs
        for ch in c["internal_chemotypes"]
    ]
    if len(rels) > 0:
        batches = batchify(rels, bs=bs)
        for batch in tqdm(
            batches,
            desc="Adding relationships between BGCs and internal chemotypes",
        ):
            batch_str = stringfy_dicts(batch, keys=["region_id", "label"])
            run_cypher(
                f"""
                UNWIND {batch_str} as row
                MATCH (n: MetabolomicRegion {{region_id: row.region_id}}),
                      (m: InternalChemotype {{label: row.label}})
                MERGE (n)-[:metab_to_internal_chemotype]->(m)
            """
            )
    # connect bgc to mibig chemotypes
    rels = [
        {"region_id": c["region_id"], "label": ch}
        for c in bgcs
        for ch in c["mibig_chemotypes"]
    ]
    if len(rels) > 0:
        batches = batchify(rels, bs=bs)
        for batch in tqdm(
            batches,
            desc="Adding relationships between BGCs and MIBiG Chemotypes",
        ):
            batch_str = stringfy_dicts(batch, keys=["region_id", "label"])
            run_cypher(
                f"""
                UNWIND {batch_str} as row
                MATCH (n: MetabolomicRegion {{region_id: row.region_id}}),
                      (m: MibigChemotype {{label: row.label}})
                MERGE (n)-[:metab_to_mibig_chemotype]->(m)
            """
            )
    # connect bgcs to orfs
    if orfs_uploaded:
        rels = [
            {"region_id": c["region_id"], "orf_id": o}
            for c in bgcs
            for o in c["orfs"]
        ]
        if len(rels) > 0:
            batches = batchify(rels, bs=bs)
            for batch in tqdm(
                batches, desc="Adding relationships between BGCs and orfs"
            ):
                batch_str = stringfy_dicts(batch, keys=["region_id", "orf_id"])
                run_cypher(
                    f"""
                    UNWIND {batch_str} as row
                    MATCH (n: MetabolomicRegion {{region_id: row.region_id}}),
                          (m: Orf {{orf_id: row.orf_id}})
                    MERGE (n)-[:metab_to_orfs]->(m)
                """
                )
    # connect bgcs to genomes
    if genome_uploaded and isinstance(genome_id, int):
        rels = [
            {"region_id": c["region_id"], "genome_id": genome_id} for c in bgcs
        ]
        if len(rels) > 0:
            batches = batchify(rels, bs=bs)
            for batch in tqdm(
                batches, desc="Adding relationships between genomes and BGCs"
            ):
                batch_str = stringfy_dicts(
                    batch, keys=["region_id", "genome_id"]
                )
                run_cypher(
                    f"""
                    UNWIND {batch_str} as row
                    MATCH (n: Genome {{genome_id: row.genome_id}}),
                          (m: MetabolomicRegion {{region_id: row.region_id}})
                    MERGE (n)-[:genome_to_metab]->(m)
                """
                )
    return True
