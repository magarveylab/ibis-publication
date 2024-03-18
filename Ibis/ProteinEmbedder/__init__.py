from Ibis.ProteinEmbedder.pipeline import ProteinEmbedderPipeline
from Ibis.ProteinEmbedder.datastructs import PipelineOutput
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


def upload_protein_embeddings_from_fps(
    filenames: List[str], output_dir: str, uploaded_orfs: bool, bs: int = 1000
) -> bool:
    to_upload = []
    for emb_fp in filenames:
        name
