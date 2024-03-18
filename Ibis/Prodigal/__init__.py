from Ibis.Prodigal.datastructs import ProdigalOutput
from Ibis.Prodigal.upload import upload_contigs, upload_genomes, upload_contigs
from multiprocessing import Pool
from functools import partial
from Bio import SeqIO
from tqdm import tqdm
import pyrodigal
import xxhash
import json
import os
from typing import List, Dict, Union


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


def run_on_nuc_fasta_fp(nuc_fasta_fp: str, output_dir: str = None):
    basename = os.path.basename(nuc_fasta_fp)
    output_fp = f"{output_dir}/{basename}/prodigal.json"
    if os.path.exists(output_fp) == False:
        proteins = run_prodigal(nuc_fasta_fp)
        with open(output_fp, "w") as f:
            json.dump(proteins, f)
    return output_fp


def parallel_run_on_nuc_fasta_fps(
    filenames: List[str], output_dir: str, cpu_cores: int = 1
):
    funct = partial(run_on_nuc_fasta_fp, output_dir=output_dir)
    pool = Pool(cpu_cores)
    process = pool.imap_unordered(funct, filenames)
    out = [p for p in tqdm(process, total=len(filenames))]
    pool.close()
    return out


def upload_contigs_from_prodigal_fps(filenames: List[str]) -> bool:
    contig_ids = set()
    for prodigal_fp in tqdm(filenames):
        contig_ids.updae(
            [p["contig_id"] for p in json.load(open(prodigal_fp, "r"))]
        )
    return upload_contigs(contig_ids=list(contig_ids))


def upload_genomes_from_prodigal_fps(
    filenames: List[str], genome_lookup: Dict[str, int], contigs_uploaded: bool
) -> bool:
    genomes = []
    for prodigal_fp in tqdm(filenames):
        name = prodigal_fp.split("/")[-2]
        if name not in genome_lookup:
            continue
        contig_ids = set(
            p["contig_id"] for p in json.load(open(prodigal_fp, "r"))
        )
        genomes.append(
            {
                "genome_id": genome_lookup[name],
                "filepath": name,
                "contig_ids": list(contig_ids),
            }
        )
    if len(genomes) > 0:
        return upload_genome(
            genomes=genomes, contigs_uploaded=contigs_uploaded
        )
    else:
        return False


def upload_orfs_from_prodigal_fps(
    filenames: List[str], contigs_uploaded: bool = False
):
    orfs = []
    for prodigal_fp in tqdm(filenames):
        for p in json.load(open(prodigal_fp, "r")):
            orfs.append(
                {
                    "protein_id": p["protein_id"],
                    "contig_id": p["contig_id"],
                    "contig_start": p["contig_start"],
                    "contig_stop": p["contig_stop"],
                }
            )
    return upload_orfs(orfs=orfs, contigs_uploaded=contigs_uploaded)
