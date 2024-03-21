import json
import os
from typing import List, Optional

from tqdm import tqdm

from Ibis.PropeptidePredictor.pipeline import PropeptidePredictorPipeline
from Ibis.PropeptidePredictor.upload import upload_propeptides


def run_propeptide_predictor_on_proteins(
    protein_sequences: List[str], gpu_id: Optional[int] = None
):
    propeptide_predictor = PropeptidePredictorPipeline(gpu_id=gpu_id)
    return propeptide_predictor.run(protein_sequences)


def run_on_files(
    filenames: List[str],
    output_dir: str,
    prodigal_preds_created: bool,
    mol_preds_created: bool,
    gpu_id: Optional[int] = None,
    cpu_cores: int = 1,
) -> bool:
    if prodigal_preds_created == False:
        raise ValueError("Prodigal predictions not created")
    if mol_preds_created == False:
        raise ValueError("Molecule predictions not created")
    # load pipeline
    pipeline = PropeptidePredictorPipeline(gpu_id=gpu_id, cpu_cores=cpu_cores)
    # analysis
    for name in tqdm(filenames):
        export_fp = f"{output_dir}/{name}/propeptide_predictions.json"
        if os.path.exists(export_fp) == False:
            proteins_to_run = set()
            mol_pred_fp = f"{output_dir}/{name}/molecule_predictions.json"
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
                    if p["protein_id"] in proteins_to_run:
                        sequences.add(p["sequence"])
                out = pipeline.run(list(sequences))
            else:
                out = []
            with open(export_fp, "w") as f:
                json.dump(out, f)
    del pipeline
    return True


def upload_propetides_from_files(
    propeptide_pred_fp: str, log_dir: str, orfs_uploaded: bool
) -> bool:
    log_fp = f"{log_dir}/propeptide_uploaded.json"
    if os.path.exists(log_fp) == False:
        propeptides = json.load(open(propeptide_pred_fp))
        upload_propeptides(
            propeptides=propeptides, orfs_uploaded=orfs_uploaded
        )
        json.dump({"uploaded": True}, open(log_fp, "w"))
    return True
