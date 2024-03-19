import json
import os
import pickle
from functools import partial
from typing import Callable, List

from tqdm import tqdm

from Ibis.ProteinDecoder.databases import (
    IbisEC,
    IbisGene,
    IbisGeneFamily,
    IbisKO,
    IbisMolecule,
)
from Ibis.ProteinDecoder.upload import upload_knn
from Ibis.Utilities.Qdrant.classification import (
    KNNClassification,
    neighborhood_classification,
    ontology_neighborhood_classification,
)

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


def decode_from_files(
    filenames: List[str],
    output_dir: str,
    protein_embs_created: bool,
    decode_fn: Callable,
    decode_name: str,
) -> bool:
    if protein_embs_created == False:
        raise ValueError("Protein embeddings not created")
    # run on all proteins
    for name in tqdm(filenames):
        export_fp = f"{output_dir}/{name}/{decode_name}_predictions.json"
        if os.path.exists(export_fp) == False:
            embedding_fp = f"{output_dir}/{name}/protein_embedding.pkl"
            data_queries = []
            if decode_name == "ec":
                for p in pickle.load(open(embedding_fp, "rb")):
                    # only consider enzymes for ec predictions
                    if p["ec1"] != "EC:-":
                        data_queries.append(
                            {
                                "query_id": p["protein_id"],
                                "embedding": p["embedding"],
                            }
                        )
            else:
                for p in pickle.load(open(embedding_fp, "rb")):
                    data_queries.append(
                        {
                            "query_id": p["protein_id"],
                            "embedding": p["embedding"],
                        }
                    )
            out = decode_fn(data_queries)
            with open(export_fp, "w") as f:
                json.dump(out, f)
    return True


def decode_from_bgc_filenames(
    filenames: List[str],
    output_dir: str,
    decode_fn: Callable,
    decode_name: str,
) -> List[str]:
    # run on only proteins in bgc
    decode_pred_filenames = []
    for bgc_fp in tqdm(filenames):
        name = os.path.basename(os.path.dirname(bgc_fp))
        export_fp = f"{output_dir}/{name}/{decode_name}_predictions.json"
        if os.path.exists(export_fp) == False:
            # load embeddings
            hash_embedding_lookup = {}
            embedding_fp = f"{output_dir}/{name}/protein_embedding.pkl"
            for p in pickle.load(open(embedding_fp, "rb")):
                protein_id = p["protein_id"]
                embedding = p["embedding"]
                hash_embedding_lookup[protein_id] = embedding
            # connect embeddings to orfs
            orf_embedding_lookup = {}
            prodigal_fp = f"{output_dir}/{name}/prodigal.json"
            for p in json.load(open(prodigal_fp)):
                contig_id = p["contig_id"]
                contig_start = p["contig_start"]
                contig_stop = p["contig_stop"]
                protein_id = p["protein_id"]
                orf_id = f"{contig_id}_{contig_start}_{contig_stop}"
                embedding = hash_embedding_lookup[protein_id]
                orf_embedding_lookup[orf_id] = {
                    "query_id": protein_id,
                    "embedding": embedding,
                }
            # build data queries
            data_queries = []
            if decode_name == "molecule":
                for cluster in json.load(open(bgc_fp)):
                    internal_chemotypes = cluster["internal_chemotypes"]
                    if (
                        "Bacteriocin" in internal_chemotypes
                        and "Ripp" in internal_chemotypes
                    ):
                        for orf_id in cluster["orfs"]:
                            data_queries.append(orf_embedding_lookup[orf_id])
            else:
                for cluster in json.load(open(bgc_fp)):
                    for orf_id in cluster["orfs"]:
                        data_queries.append(orf_embedding_lookup[orf_id])
            # analysis
            out = decode_fn(data_queries)
            with open(export_fp, "w") as f:
                json.dump(out, f)
        decode_pred_filenames.append(export_fp)
    return decode_pred_filenames


def upload_protein_decoding_from_fp(
    knn_fp: str, label_type: str, protein_embs_uploaded: bool
) -> bool:
    if protein_embs_uploaded:
        return upload_knn(
            annotations=json.load(open(knn_fp)), label_type=label_type
        )
    else:
        return False
