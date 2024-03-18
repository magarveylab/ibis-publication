from Ibis.ProteinEmbedder.pipeline import ProteinEmbedderPipeline
from Ibis.ProteinEmbedder.datastructs import PipelineOutput
from Ibis.ProteinEmbedder.upload import upload_protein_embeddings
from tqdm import tqdm
import pickle
import json
import os
from typing import List


def run_on_protein_sequences(
    sequences: List[str], gpu_id: int = 0
) -> List[PipelineOutput]:
    pipeline = ProteinEmbedderPipeline(gpu_id=gpu_id)
    return pipeline.run(sequences)


def run_on_prodigal_fps(
    filenames: List[str], output_dir: str, gpu_id: int = 0
) -> List[str]:
    protein_embedding_filenames = []
    # load pipeline
    pipeline = ProteinEmbedderPipeline(gpu_id=gpu_id)
    # analysis
    for prodigal_fp in tqdm(filenames):
        name = os.path.basename(os.path.dirname(prodigal_fp))
        export_filename = f"{output_dir}/{name}/protein_embedding.pkl"
        if os.path.exists(export_filename) == False:
            sequences = [p["sequence"] for p in json.load(open(prodigal_fp))]
            out = pipeline.run(sequences)
            with open(export_filename, "wb") as f:
                pickle.dump(out, f)
        protein_embedding_filenames.append(export_filename)
    # delete pipeline
    del pipeline
    return protein_embedding_filenames


def upload_protein_embeddings_from_fp(
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
        embedding_lookup[protein_id] = embedding
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
            embedding = embedding_lookup[protein_id]
            to_upload.append(
                {
                    "contig_id": contig_id,
                    "contig_start": contig_start,
                    "contig_stop": contig_stop,
                    "protein_id": protein_id,
                    "embedding": embedding,
                }
            )
    return upload_protein_embeddings(
        orfs=to_upload, orfs_uploaded=orfs_uploaded
    )
