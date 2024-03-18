import json
import os
from typing import List, Optional

from Ibis.PropeptidePredictor.pipeline import PropeptidePredictorPipeline


def run_propeptide_predictor_on_proteins(
    protein_sequences: List[str], gpu_id: Optional[int] = None
):
    propeptide_predictor = PropeptidePredictorPipeline(gpu_id=gpu_id)
    return propeptide_predictor.run(protein_sequences)


def run_on_mol_pred_fps(
    filenames: List[str],
    output_dir: str,
    gpu_id: Optional[int] = None,
    cpu_cores: int = 1,
):
    propeptide_pred_filenames = []
    # load pipeline
    pipeline = PropeptidePredictorPipeline(gpu_id=gpu_id, cpu_cores=cpu_cores)
    # analysis
    for mol_pred_fp in filenames:
        name = os.path.basename(os.path.dirname(mol_pred_fp))
        export_fp = f"{output_dir}/{name}/propeptide_predictions.json"
        if os.path.exists(export_fp) == False:
            proteins_to_run = set()
            for p in json.load(open(mol_pred_fp)):
                if (
                    p["homology"] is not None
                    and p["homology"] >= 0.6
                    and p["label"] != "Bacteriocin"
                ):
                    proteins_to_run.add(p["hash_id"])
            if len(proteins_to_run) > 0:
                sequences = set()
                prodigal_fp = f"{output_dir}/{name}/prodigal.json"
                for p in json.load(open(prodigal_fp)):
                    if p["hash_id"] in proteins_to_run:
                        sequences.add(p["sequence"])
                out = pipeline.run(list(sequences))
            else:
                out = []
            with open(export_fp, "w") as f:
                json.dump(out, f)
        propeptide_pred_filenames.append(export_fp)
    del pipeline
    return propeptide_pred_filenames
