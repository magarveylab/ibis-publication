import json
import os
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Union

import pyrodigal
import xxhash
from Bio import SeqIO
from tqdm import tqdm

from Ibis.Prodigal.datastructs import ProdigalOutput

########################################################################
# General functions
########################################################################


def run_prodigal(nuc_fasta_fp: str) -> List[ProdigalOutput]:
    proteins = []
    for record in SeqIO.parse(nuc_fasta_fp, "fasta"):
        seq_length = len(record.seq)
        seq_input = bytes(record.seq)
        contig_id = xxhash.xxh32(str(record.seq)).intdigest()
        # always run meta procedure
        orf_finder = pyrodigal.OrfFinder(meta=True)
        genes = orf_finder.find_genes(seq_input)
        for gene in genes:
            prot_seq = gene.translate().replace("*", "")
            prot_id = xxhash.xxh32(str(prot_seq)).intdigest()
            contig_start = gene.begin
            contig_stop = gene.end
            proteins.append(
                {
                    "protein_id": prot_id,
                    "contig_id": contig_id,
                    "contig_start": contig_start,
                    "contig_stop": contig_stop,
                    "sequence": prot_seq,
                }
            )
    return proteins


########################################################################
# Airflow inference functions
########################################################################


def run_on_single_file(nuc_fasta_fp: str, output_dir: str = None) -> bool:
    basename = os.path.basename(nuc_fasta_fp)
    output_fp = f"{output_dir}/{basename}/prodigal.json"
    if os.path.exists(output_fp) == False:
        proteins = run_prodigal(nuc_fasta_fp)
        with open(output_fp, "w") as f:
            json.dump(proteins, f)
    return True


def parallel_run_on_files(
    filenames: List[str], output_dir: str, cpu_cores: int = 1
) -> bool:
    funct = partial(run_on_single_file, output_dir=output_dir)
    pool = Pool(cpu_cores)
    process = pool.imap_unordered(funct, filenames)
    out = [
        p
        for p in tqdm(
            process, total=len(filenames), leave=False, desc="Running Prodigal"
        )
    ]
    pool.close()
    return True


########################################################################
# Airflow upload functions
########################################################################


def upload_contigs_from_files(prodigal_fp: str, log_dir: str) -> bool:
    from Ibis.Prodigal.upload import upload_contigs

    log_fp = f"{log_dir}/contigs_uploaded.json"
    if os.path.exists(log_fp) == False:
        contig_ids = set(
            p["contig_id"] for p in json.load(open(prodigal_fp, "r"))
        )
        upload_contigs(contig_ids=list(contig_ids))
        json.dump({"upload": True}, open(log_fp, "w"))
    return True


def upload_genome_from_files(
    nuc_fasta_fp: str,
    prodigal_fp: str,
    log_dir: str,
    genome_id: int,
    contigs_uploaded: bool,
) -> bool:
    from Ibis.Prodigal.upload import upload_genomes

    if isinstance(genome_id, int):
        log_fp = f"{log_dir}/genomes_uploaded.json"
        if os.path.exists(log_fp) == False:
            contig_ids = set(
                p["contig_id"] for p in json.load(open(prodigal_fp, "r"))
            )
            genome = {
                "genome_id": genome_id,
                "filepath": nuc_fasta_fp,
                "contig_ids": list(contig_ids),
            }
            upload_genomes(genomes=[genome], contigs_uploaded=contigs_uploaded)
            json.dump({"upload": True}, open(log_fp, "w"))
        return True
    else:
        return False


def upload_orfs_from_files(
    prodigal_fp: str, log_dir: str, contigs_uploaded: bool = False
):
    from Ibis.Prodigal.upload import upload_orfs

    log_fp = f"{log_dir}/orfs_uploaded.json"
    if os.path.exists(log_fp) == False:
        orfs = []
        for p in json.load(open(prodigal_fp, "r")):
            orfs.append(
                {
                    "protein_id": p["protein_id"],
                    "contig_id": p["contig_id"],
                    "contig_start": p["contig_start"],
                    "contig_stop": p["contig_stop"],
                }
            )
        upload_orfs(orfs=orfs, contigs_uploaded=contigs_uploaded)
        json.dump({"upload": True}, open(log_fp, "w"))
    return True
