from Ibis.Utilities.knowledge_graph import run_cypher, batchify, stringfy_dicts
from typing import TypedDict


class GenomeDict(TypedDict):
    genome_id: int
    filepath: str
    contig_ids


class OrfDict(TypedDict):
    protein_id: int
    contig_id: int
    contig_start: int
    contig_stop: int


def upload_contigs(contig_ids: List[int], bs: int = 1000) -> bool:
    batches = batchify(contig_ids, bs=bs)
    for batch in tqdm(batches, desc="Uploading contigs"):
        run_cypher(
            f"UNWIND {contig_ids} as row MERGE (n: Contig {{hash_id: row}})"
        )
    return True


def upload_genomes(genomes: List[GenomeDict], contigs_uploaded: bool) -> bool:
    # upload genomes
    batches = batchify(genomes, bs=bs)
    for batch in batches:
        batch_str = stringfy_dicts(batch, keys=["genome_id", "filepath"])
        run_cypher(
            f"""UNWIND {batch_str} as row
                MERGE (n: Genome {{genome_id: row.genome_id}})
                ON CREATE
                    SET n.filepath = row.filepath
        """
        )
        if contigs_uploaded:
            rels = [
                {"genome_id": g, "contig_id": c}
                for g in batch
                for c in g["contig_ids"]
            ]
            batch_str = stringfy_dicts(rels, keys=["genome_id", "contig_id"])
            run_cypher(
                f"""
                UNWIND {batch_str} as row
                MATCH (n: Genome {{genome_id: row.genome_id}}),
                      (m: Contig {{hash_id: row.contig_id}})
                MERGE (n)-[r: genome_to_contig]->(m)
            """
            )
    return True


def upload_orfs(
    orfs: List[OrfDict], contigs_uploaded: bool = False, bs: int = 1000
) -> bool:
    source = "pyrodigal-2.0.4"
    batches = batchify(orfs, bs=bs)
    for batch in tqdm(batches, desc="Uploading orfs"):
        # add orf ids
        for orf in batch:
            contig_id = orf["contig_id"]
            contig_start = orf["contig_start"]
            contig_stop = orf["contig_stop"]
            orf_id = f"{contig_id}_{contig_start}_{contig_stop}"
            batch["orf_id"] = orf_id
            batch["source"] = source
        batch_str = stringfy_dicts(
            batch,
            keys=[
                "orf_id",
                "protein_id",
                "contig_start",
                "contig_stop",
                "source",
            ],
        )
        run_cypher(
            f"""
            UNWIND {batch_str} as row
            MERGE (n: Orf {{orf_id: row.orf_id}})
            ON CREATE
                SET n.hash_id = row.protein_id,
                    n.contig_start = row.contig_start,
                    n.contig_stop = row.contig_stop,
                    n.source = row.source
        """
        )
        if contigs_uploaded:
            batch_str = stringfy_dicts(batch, keys=["orf_id", "contig_id"])
            run_cypher(
                f"""
                UNWIND {batch_str} as row
                MATCH (n: Contig {{hash_id: row.contig_id}}),
                      (m: Orf {{orf_id: row.orf_id}})
                MERGE (n)-[r: contig_to_orf]->(m)
            """
            )
    return True
