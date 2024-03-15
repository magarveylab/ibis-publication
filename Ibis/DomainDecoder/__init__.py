from Ibis.Utilities.Qdrant.classification import (
    ontology_neighborhood_classification,
    neighborhood_classification,
    KNNClassification,
)
from Ibis.DomainDecoder.databases import (
    IbisAdenylation,
    IbisAcyltransferase,
    IbisKetosynthase,
    IbisKetoreductase,
    IbisDehydratase,
    IbisEnoylreductase,
    IbisThiolation,
)
from functools import partial
from tqdm import tqdm
import xxhash
from typing import List

decode_adenylation = partial(
    KNNClassification,
    label_type="SubstrateLabel",
    qdrant_db=IbisAdenylation,
    classification_method=ontology_neighborhood_classification,
    top_n=3,
    apply_cutoff_before_homology=False,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)

decode_acyltransferase = partial(
    KNNClassification,
    label_type="SubstrateLabel",
    qdrant_db=IbisAcyltransferase,
    classification_method=ontology_neighborhood_classification,
    top_n=5,
    apply_cutoff_before_homology=False,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)

decode_ketosynthase = partial(
    KNNClassification,
    label_type="DomainFunctionalLabel",
    qdrant_db=IbisKetosynthase,
    classification_method=neighborhood_classification,
    top_n=5,
    apply_cutoff_before_homology=False,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)

decode_ketoreductase = partial(
    KNNClassification,
    label_type="DomainFunctionalLabel",
    qdrant_db=IbisKetoreductase,
    classification_method=neighborhood_classification,
    top_n=5,
    apply_cutoff_before_homology=False,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)

decode_dehydratase = partial(
    KNNClassification,
    label_type="DomainFunctionalLabel",
    qdrant_db=IbisDehydratase,
    classification_method=neighborhood_classification,
    top_n=5,
    apply_cutoff_before_homology=False,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)

decode_enoylreductase = partial(
    KNNClassification,
    label_type="DomainFunctionalLabel",
    qdrant_db=IbisDehydratase,
    classification_method=neighborhood_classification,
    top_n=5,
    apply_cutoff_before_homology=False,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)

decode_thiolation = partial(
    KNNClassification,
    label_type="DomainSubclassLabel",
    qdrant_db=IbisThiolation,
    classification_method=neighborhood_classification,
    top_n=5,
    dist_cutoff=6.32,
    apply_cutoff_before_homology=False,
    homology_cutoff=1.0,
    apply_homology_cutoff=True,
    apply_cutoff_after_homology=True,
)


def decode_from_embedding_fps(
    filenames: List[str],
    output_dir: str,
    decode_fn: Callable,
    target_domain: str,
):
    decode_pred_filenames = []
    for embedding_fp in tqdm(filenames):
        name = embedding_fp.split("/")[-2]
        export_fp = f"{output_dir}/{name}/{target_domain}_predictions.json"
        if os.path.exists(export_fp) == False:
            # find domains to analyze
            domains_to_run = set()
            domain_pred_fp = f"{output_dir}/{name}/domain_predictions.json"
            for prot in json.load(open(domain_pred_fp)):
                prot_seq = prot["sequence"]
                for region in prot["regions"]:
                    if region["label"] == target_domain:
                        start, end = region["start"], region["end"]
                        domain_seq = prot_seq[start:end]
                        domain_id = xxhash.xxh32(domain_seq).intdigest()
                        domains_to_run.add(domain_id)
            # analysis
            data_queries = [
                {"query_id": p["protein_id"], "embedding": p["embedding"]}
                for p in pickle.load(open(embedding_fp, "rb"))
                if p["protein_id"] in domains_to_run
            ]
            if len(data_queries) == 0:
                out = []
            else:
                out = decode_fn(data_queries)
            with open(export_fp, "w") as f:
                json.dump(out, f)
        decode_pred_filenames.append(export_fp)
    return decode_pred_filenames
