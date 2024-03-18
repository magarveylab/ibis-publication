import json
import os
import pickle
from typing import List

from tqdm import tqdm

from Ibis.DomainEmbedder.datastructs import PipelineOutput
from Ibis.DomainEmbedder.pipeline import DomainEmbedderPipeline
from Ibis.DomainEmbedder.upload import (
    initialize_domain_annotations,
    upload_domain_embeddings,
)


def run_on_protein_sequences(
    sequences: List[str], gpu_id: int = 0
) -> List[PipelineOutput]:
    # load pipeline
    pipeline = DomainEmbedderPipeline(gpu_id=gpu_id)
    return pipeline.run(sequences)


def run_on_domain_pred_fps(
    filenames: List[str], output_dir: str, gpu_id: int = 0
):
    target_domains = ["A", "AT", "KS", "KR", "DH", "ER", "T"]
    domain_embedding_filenames = []
    # load pipeline
    pipeline = DomainEmbedderPipeline(gpu_id=gpu_id)
    # analysis
    for domain_pred_fp in tqdm(filenames):
        name = os.path.basename(os.path.dirname(domain_pred_fp))
        export_filename = f"{output_dir}/{name}/domain_embedding.pkl"
        if os.path.exists(export_filename) == False:
            sequences = set()
            for protein in json.load(open(domain_pred_fp)):
                prot_sequence = protein["sequence"]
                for domain in protein["regions"]:
                    domain_label = domain["label"]
                    if domain_label in target_domains:
                        start, stop = domain["start"], domain["stop"]
                        domain_sequence = prot_sequence[start:stop]
                        sequences.add(domain_sequence)
            out = pipeline.run(list(sequences))
            with open(export_filename, "wb") as f:
                pickle.dump(out, f)
        domain_embedding_filenames.append(export_filename)
    # delete pipeline
    del pipeline
    return domain_embedding_filenames


def upload_domain_embeddings_from_fp(
    domain_pred_fp: str,
    domain_embedding_fp: str,
    domains_uploaded: bool,
):
    # domain embedding lookup
    embedding_lookup = {
        d["domain_id"]: d["embedding"]
        for d in pickle.load(open(domain_embedding_fp, "rb"))
    }
    # upload domains
    domains = []
    for p in json.load(open(domain_pred_fp)):
        protein_id = p["protein_id"]
        for d in p["regions"]:
            hash_id = d["domain_id"]
            if hash_id in embedding_lookup:
                embedding = embedding_lookup[hash_id]
                domains.append(
                    {
                        "protein_id": protein_id,
                        "protein_start": d["start"],
                        "protein_stop": d["stop"],
                        "hash_id": hash_id,
                        "embedding": embedding,
                    }
                )
    if len(domains) > 0:
        embedding_uploaded = upload_domain_embeddings(
            domains=domains, domains_uploaded=domains_uploaded
        )
        initialize_domain_annotations(
            hash_ids=list(set(d["hash_id"] for d in domains)),
            embedding_uploaded=embedding_uploaded,
        )
    return True
