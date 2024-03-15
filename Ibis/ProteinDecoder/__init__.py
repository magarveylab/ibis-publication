from Ibis.Utilities.Qdrant.classification import (
    ontology_neighborhood_classification,
    neighborhood_classification,
    KNNClassification,
)
from Ibis.ProteinDecoder.databases import (
    IbisEC,
    IbisKO,
    IbisGeneFamily,
    IbisGene,
    IbisMolecule,
)
from functools import partial
from tqdm import tqdm
import pickle
import json
import os
from typing import List, Callable

decode_ec = partial(
    KNNClassification,
    qdrant_db=IbisEC,
    classification_method=ontology_neighborhood_classification,
    top_n=5,
    dist_cutoff=25.36,
    apply_cutoff_before_homology=True,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)
decode_ko = partial(
    KNNClassification,
    qdrant_db=IbisKO,
    classification_method=neighborhood_classification,
    top_n=5,
    dist_cutoff=26.06,
    apply_cutoff_before_homology=True,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)

decode_molecule = partial(
    KNNClassification,
    qdrant_db=IbisMolecule,
    classification_method=neighborhood_classification,
    top_n=5,
    dist_cutoff=31.69,
    apply_cutoff_before_homology=False,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)

decode_gene_family = partial(
    KNNClassification,
    qdrant_db=IbisGeneFamily,
    classification_method=neighborhood_classification,
    partition_names=None,
    top_n=5,
    dist_cutoff=29.58,
    apply_cutoff_before_homology=True,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)

decode_gene = partial(
    KNNClassification,
    qdrant_db=IbisGene,
    classification_method=neighborhood_classification,
    top_n=1,
    dist_cutoff=29.58,
    apply_cutoff_before_homology=True,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)


def decode_from_embedding_fps(
    filenames: List[str],
    output_dir: str,
    decode_fn: Callable,
    decode_name: str,
) -> List[str]:
    # run on all proteins
    decode_pred_filenames = []
    for embedding_fp in tqdm(filenames):
        name = embedding_fp.split("/")[-2]
        export_fp = f"{output_dir}/{name}/{decode_name}_predictions.json"
        if os.path.exists(export_fp) == False:
            data_queries = [
                {"query_id": p["protein_id"], "embedding": p["embedding"]}
                for p in pickle.load(open(embedding_fp, "rb"))
            ]
            out = decode_fn(data_queries)
            with open(export_fp, "w") as f:
                json.dump(out, f)
        decode_pred_filenames.append(export_fp)
    return decode_pred_filenames


def decode_from_bgc_filenames(
    filenames: List[str],
    output_dir: str,
    decode_fn: Callable,
    decode_name: str,
) -> List[str]:
    # run on only proteins in bgc
    decode_pred_filenames = []
    for bgc_fp in tqdm(filenames):
        name = bgc_fp.split("/")[-2]
        export_fp = f"{output_dir}/{name}/{decode_name}_predictions.json"
        if os.path.exists(export_fp) == False:
            embedding_fp = f"{output_dir}/{name}/protein_embeddings.pkl"
            embedding_lookup = {}
            for p in pickle.load(open(embedding_fp, "rb")):
                orf_id = (
                    f'{p["contig_id"]}_{p["contig_start"]}_{p["contig_stop"]}'
                )
                embedding_lookup[orf_id] = {
                    "query_id": p["protein_id"],
                    "embedding": p["embedding"],
                }
            data_queries = []
            if decode_name == "molecule":
                for cluster in pickle.load(open(bgc_fp, "rb")):
                    internal_chemotypes = cluster["internal_chemotypes"]
                    if (
                        "Bacteriocin" in internal_chemotypes
                        and "Ripp" in internal_chemotypes
                    ):
                        for orf_id in cluster["orfs"]:
                            data_queries.append(embedding_lookup[orf_id])
            else:
                for cluster in pickle.load(open(bgc_fp, "rb")):
                    for orf_id in cluster["orfs"]:
                        data_queries.append(embedding_lookup[orf_id])
            out = decode_fn(data_queries)
            with open(export_fp, "w") as f:
                json.dump(out, f)
        decode_pred_filenames.append(export_fp)
    return decode_pred_filenames
