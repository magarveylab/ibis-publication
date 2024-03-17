from Ibis.PrimaryMetabolismPredictor.datastructs import EnzymeKOData
import json
from typing import List


def merge_protein_annotations(
    prodigal_fp: str,
    ko_pred_fp: str,
    ec_pred_fp: str,
) -> List[EnzymeKOData]:
    prots = json.load(open(prodigal_fp, "r"))
    ko_lookup = {x["hash_id"]: x for x in json.load(open(ko_pred_fp, "r"))}
    ec_lookup = {x["hash_id"]: x for x in json.load(open(ec_pred_fp, "r"))}
    merged = []
    for annot in prots:
        # set defaults
        ec_num = None
        ec_homol = None
        ko_num = None
        ko_homol = None
        ko_sim = None
        # protein properties
        hash_id = annot["protein_id"]
        contig_id = annot["contig_id"]
        contig_start = annot["contig_start"]
        contig_stop = annot["contig_stop"]
        orf_id = f"{contig_id}_{contig_start}_{contig_stop}"
        # pull decoded labels
        ec = ec_lookup.get(hash_id)
        if ec is not None:
            if ec["label"] is not None:
                ec_num = ec["label"]
                ec_homol = ec["homology"]
        ko = ko_lookup[hash_id]
        if ko["label"] is not None:
            ko_num = ko["label"]
            ko_homol = ko["homology"]
            ko_sim = ko["similarity"]
        merged.append(
            {
                "orf_id": orf_id,
                "ec_number": ec_num,
                "homology_score": ec_homol,
                "ko_ortholog": ko_num,
                "ko_homology_score": ko_homol,
                "ko_similarity_score": ko_sim,
            }
        )
    return merged
