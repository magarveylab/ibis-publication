from Ibis.Prodigal.datastructs import ProdigalOutput
from multiprocessing import Pool
from functools import partial
from Bio import SeqIO
import pyrodigal
import xxhash
import json
import os
from typing import List, Union


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
    return output


def parallel_run_on_nuc_fasta_fps(
    filenames: List[str], output_dir: str, cpu_cores: int = 1
):
    funct = partial(run_on_single_nuc_fasta_fp, output_dir=output_dir)
    pool = Pool(cpu_cores)
    process = pool.imap_unordered(funct, filenames)
    out = [p for p in tqdm(process, total=len(filenames))]
    pool.close()
    return out
