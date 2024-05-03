import json
import os
import pickle
from functools import partial
from typing import Callable, List

import xxhash
from tqdm import tqdm

from Ibis.DomainDecoder.databases import (
    IbisAcyltransferase,
    IbisAdenylation,
    IbisDehydratase,
    IbisEnoylreductase,
    IbisKetoreductase,
    IbisKetosynthase,
    IbisThiolation,
)
from Ibis.DomainDecoder.upload import upload_knn
from Ibis.Utilities.Qdrant.classification import (
    KNNClassification,
    neighborhood_classification,
    ontology_neighborhood_classification,
)

########################################################################
# General functions
########################################################################


decode_adenylation = partial(
    KNNClassification,
    qdrant_db=IbisAdenylation,
    classification_method=ontology_neighborhood_classification,
    top_n=3,
    apply_cutoff_before_homology=False,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)

decode_acyltransferase = partial(
    KNNClassification,
    qdrant_db=IbisAcyltransferase,
    classification_method=ontology_neighborhood_classification,
    top_n=5,
    apply_cutoff_before_homology=False,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)

decode_ketosynthase = partial(
    KNNClassification,
    qdrant_db=IbisKetosynthase,
    classification_method=neighborhood_classification,
    top_n=5,
    apply_cutoff_before_homology=False,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)

decode_ketoreductase = partial(
    KNNClassification,
    qdrant_db=IbisKetoreductase,
    classification_method=neighborhood_classification,
    top_n=5,
    apply_cutoff_before_homology=False,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)

decode_dehydratase = partial(
    KNNClassification,
    qdrant_db=IbisDehydratase,
    classification_method=neighborhood_classification,
    top_n=5,
    apply_cutoff_before_homology=False,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)

decode_enoylreductase = partial(
    KNNClassification,
    qdrant_db=IbisEnoylreductase,
    classification_method=neighborhood_classification,
    top_n=5,
    apply_cutoff_before_homology=False,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)

decode_thiolation = partial(
    KNNClassification,
    qdrant_db=IbisThiolation,
    classification_method=neighborhood_classification,
    top_n=5,
    dist_cutoff=6.32,
    apply_cutoff_before_homology=False,
    homology_cutoff=1.0,
    apply_homology_cutoff=True,
    apply_cutoff_after_homology=True,
)

########################################################################
# Airflow inference functions
########################################################################


def run_on_files(
    filenames: List[str],
    output_dir: str,
    domain_embs_created: bool,
    decode_fn: Callable,
    target_domain: str,
) -> bool:
    if domain_embs_created == False:
        raise ValueError("Domain embeddings not created")
    for name in tqdm(filenames, leave=False, desc="Running DomainDecoder"):
        embedding_fp = f"{output_dir}/{name}/domain_embedding.pkl"
        export_fp = f"{output_dir}/{name}/{target_domain}_predictions.json"
        if os.path.exists(export_fp) == False:
            # find domains to analyze
            domains_to_run = set()
            domain_pred_fp = f"{output_dir}/{name}/domain_predictions.json"
            for prot in json.load(open(domain_pred_fp)):
                for region in prot["regions"]:
                    if region["label"] == target_domain:
                        domains_to_run.add(region["domain_id"])
            # analysis
            data_queries = [
                {"query_id": p["domain_id"], "embedding": p["embedding"]}
                for p in pickle.load(open(embedding_fp, "rb"))
                if p["domain_id"] in domains_to_run
            ]
            if len(data_queries) == 0:
                out = []
            else:
                out = decode_fn(data_queries)
            with open(export_fp, "w") as f:
                json.dump(out, f)
    return True


########################################################################
# Airflow upload functions
########################################################################


def upload_domain_decoding_from_files(
    knn_fp: str, log_dir: str, label_type: str, domain_embs_uploaded: bool
) -> bool:
    if domain_embs_uploaded:
        root = knn_fp.split("/")[-1].split(".")[0]
        domain_label = root.split("_")[1]
        log_fp = f"{log_dir}/{root}_uploaded.json"
        if os.path.exists(log_fp) == False:
            annotations = json.load(open(knn_fp))
            # fix the labels for functional
            if domain_label in ["KS", "KR", "DH", "ER"]:
                for q in annotations:
                    for d in q["predictions"]:
                        if d["label"] == "inactive":
                            d["label"] = f"{domain_label}0"
                        else:
                            d["label"] = domain_label
            upload_knn(annotations=annotations, label_type=label_type)
            json.dump({"uploaded": True}, open(log_fp, "w"))
        return True
    else:
        return False
