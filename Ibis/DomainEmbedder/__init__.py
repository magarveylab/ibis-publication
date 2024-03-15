from Ibis.DomainEmbedder.pipeline import DomainEmbedderPipeline
from Ibis.DomainEmbedder.datastructs import PipelineOutput
from tqdm import tqdm
import pickle
import json
import os
from typing import List


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
        name = domain_pred_fp.split("/")[-2]
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
