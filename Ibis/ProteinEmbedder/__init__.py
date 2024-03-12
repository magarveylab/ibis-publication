from Ibis.ProteinEmbedder.pipeline import ProteinEmbedderPipeline
import pickle
import os
from typing import List, Union


def run_on_protein_sequences(sequences: List[str], gpu_id: int = 0):
    pipeline = ProteinEmbedderPipeline(gpu_id=gpu_id)
    return pipeline.run(sequences)


def run_on_prodigal_fps(
    filenames: Union[str, List[str]], output_dir: str, gpu_id: int = 0
):
    # generates embedding fps in output directory
    if isinstance(filenames, str):
        filenames = [filenames]
    pipeline = ProteinEmbedderPipeline(gpu_id=gpu_id)
    for filename in filenames:
        basename = os.path.basename(filename)
        export_filename = f"{output_dir}/{basename}.pkl"
        if os.path.exists(export_filename):
            continue
        sequences = [p["sequence"] for p in json.load(open(filename))]
        out = pipeline.run(sequences)
        with open(export_filename, "wb") as f:
            pickle.dump(out, f)
