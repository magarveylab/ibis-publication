import json
import pickle

import xxhash


def merge_protein_annotations(
    pyrodigal_fp: str,
    ko_fp: str,
    ec_fp: str,
    embed_fp: str,
):
    prots = json.load(open(pyrodigal_fp, "r"))
    embed_dat = pickle.load(open(embed_fp, "rb"))
    # get enzymatic hash IDs
    enzyme_hashes = {
        xxhash.xxh32(x["sequence"].replace("*", "")).intdigest()
        for x in embed_dat
        if x["ec1"] != "EC:-"
    }
    ko_preds = json.load(open(ko_fp, "r"))
    ko_lookup = {x["hash_id"]: x for x in ko_preds}
    ec_preds = json.load(open(ec_fp, "r"))
    # eliminate non-enzyme EC annotations. Theoretically, this
    # should be unnecessary, but it's a good sanity check.
    ec_lookup = {
        x["hash_id"]: x for x in ec_preds if x["hash_id"] in enzyme_hashes
    }
    merged = []
    for annot in prots:
        # set defaults
        ec_num, ec_homol, ko_num, ko_homol, ko_sim = (
            None,
            None,
            None,
            None,
            None,
        )
        hash_id = annot["protein_id"]
        # use .get() due to elmination of non-enzyme annotations (if any).
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
        orf_id = f"{annot['contig_id']}_{annot['contig_start']}_{annot['contig_stop']}"
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
