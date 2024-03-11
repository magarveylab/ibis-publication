from Ibis.Prodigal.datastructs import ProdigalOutput
from Bio import SeqIO
import pyrodigal
import xxhash
from typing import List


def run_prodigal(nuc_fasta_fp: str) -> List[ProdigalOutput]:
    proteins = []
    for record in SeqIO.parse(nuc_fasta_fp, "fasta"):
        seq_length = len(record.seq)
        seq_input = bytes(record.seq)
        nuc_id = xxhash.xxh32(str(record.seq)).intdigest()
        # always run meta procedure
        orf_finder = pyrodigal.OrfFinder(meta=True)
        genes = orf_finder.find_genes(seq_input)
        for gene in genes:
            prot_seq = gene.translate().replace("*", "")
            prot_id = xxhash.xxh32(str(prot_seq)).intdigest()
            start = gene.begin
            end = gene.end
            proteins.append(
                {
                    "protein_id": prot_id,
                    "nuc_id": nuc_id,
                    "start": start,
                    "end": end,
                    "sequence": prot_seq,
                }
            )
    return proteins
