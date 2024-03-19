import json
import os
import pickle
from typing import List

from tqdm import tqdm

from Ibis.ProteinEmbedder.datastructs import PipelineOutput
from Ibis.ProteinEmbedder.pipeline import ProteinEmbedderPipeline
from Ibis.ProteinEmbedder.upload import (
    upload_ec1_annotations,
    upload_protein_embeddings,
)


def run_on_protein_sequences(
    sequences: List[str], gpu_id: int = 0
) -> List[PipelineOutput]:
    pipeline = ProteinEmbedderPipeline(gpu_id=gpu_id)
    return pipeline.run(sequences)


def run_on_files(
    filenames: List[str],
    output_dir: str,
    prodigal_preds_created: bool,
    gpu_id: int = 0,
) -> bool:
    if prodigal_preds_created == False:
        raise ValueError("Prodigal predictions not created")
    # load pipeline
    pipeline = ProteinEmbedderPipeline(gpu_id=gpu_id)
    # analysis
    for name in tqdm(filenames):
        export_filename = f"{output_dir}/{name}/protein_embedding.pkl"
        if os.path.exists(export_filename) == False:
            prodigal_fp = f"{output_dir}/{name}/prodigal.json"
            sequences = [p["sequence"] for p in json.load(open(prodigal_fp))]
            out = pipeline.run(sequences)
            with open(export_filename, "wb") as f:
                pickle.dump(out, f)
    # delete pipeline
    del pipeline
    return True


def upload_protein_embeddings_from_files(
    prodigal_fp: str,
    protein_embedding_fp: str,
    bgc_pred_fp: str,
    primary_metabolism_pred_fp: str,
    orfs_uploaded: bool,
) -> bool:
    # only upload protein embeddings for ones involved in metabolism
    embedding_lookup = {}
    for p in pickle.load(open(protein_embedding_fp, "rb")):
        protein_id = p["protein_id"]
        embedding = p["embedding"]
        ec1_label = p["ec1"]
        ec1_score = p["ec1_score"]
        is_enzyme = False if ec1_label == "EC:-" else True
        embedding_lookup[protein_id] = {
            "protein_id": protein_id,
            "embedding": embedding,
            "ec1_label": ec1_label,
            "ec1_score": ec1_score,
            "is_enzyme": is_enzyme,
        }
    # caputre orfs involved in metabolism
    relevant_orfs = set()
    primary_data = json.load(open(primary_metabolism_pred_fp))
    for module in ["ko_results", "ec_results"]:
        for pathway in primary_data[module]:
            for orfs in pathway["candidate_orfs"].values():
                relevant_orfs.update(orfs)
    secondary_data = json.load(open(bgc_pred_fp))
    for bgc in secondary_data:
        relevant_orfs.update(bgc["orfs"])
    # reformat upload data
    to_upload = []
    for p in json.load(open(prodigal_fp)):
        contig_id = p["contig_id"]
        contig_start = p["contig_start"]
        contig_stop = p["contig_stop"]
        orf_id = f"{contig_id}_{contig_start}_{contig_stop}"
        if orf_id in relevant_orfs:
            protein_id = p["protein_id"]
            to_upload.append(embedding_lookup[protein_id])
    embedding_uploaded = upload_protein_embeddings(
        orfs=to_upload, orfs_uploaded=orfs_uploaded
    )
    ec1_uploaded = upload_ec1_annotations(
        orfs=to_upload, embedding_uploaded=embedding_uploaded
    )
    return True
