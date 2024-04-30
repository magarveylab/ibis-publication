import json
import os
from multiprocessing import Pool
from typing import List

from tqdm import tqdm

from Ibis.ModulePredictor.Domain import Domain
from Ibis.ModulePredictor.Module import Module


def predict_modules_from_ibis_dir(
    ibis_dir: str,
    min_domain_score: float = 0.5,
    min_functional_score: float = 0.6,
    min_subclass_score: float = 0.6,
):
    # mandatory filenames
    filenames = {
        "domain": f"{ibis_dir}/domain_predictions.json",
        "A": f"{ibis_dir}/A_predictions.json",
        "AT": f"{ibis_dir}/AT_predictions.json",
        "KR": f"{ibis_dir}/KR_predictions.json",
        "DH": f"{ibis_dir}/DH_predictions.json",
        "ER": f"{ibis_dir}/ER_predictions.json",
        "T": f"{ibis_dir}/T_predictions.json",
    }
    # load domain hyperannotations
    domain_knn_lookup = {}
    for knn_type in ["A", "AT", "KR", "DH", "ER", "T"]:
        domain_knn_lookup[knn_type] = {}
        data = json.load(open(filenames[knn_type]))
        for query in data:
            hash_id = query["query_id"]
            predictions = query["predictions"]
            if len(predictions) > 0:
                if knn_type in ["A", "AT"]:
                    substrates = [
                        {"label": p["label"], "rank": p["rank"]}
                        for p in predictions
                    ]
                    domain_knn_lookup[knn_type][hash_id] = substrates
                elif knn_type in ["KR", "DH", "ER"]:
                    best_p = predictions[0]
                    if (
                        best_p["label"] == "inactive"
                        and best_p["homology"] >= min_functional_score
                    ):
                        functional = False
                    else:
                        functional = True
                    domain_knn_lookup[knn_type][hash_id] = functional
                elif knn_type == "T":
                    best_p = predictions[0]
                    if (
                        best_p["label"] == "B"
                        and best_p["homology"] >= min_subclass_score
                    ):
                        subclass = "B"
                    else:
                        subclass = None
                    domain_knn_lookup[knn_type][hash_id] = subclass
    # load domain regions
    data = json.load(open(filenames["domain"]))
    protein_to_modules = []
    for protein in data:
        domains = protein["regions"]
        if len(domains) == 0:
            continue
        # filter domains
        protein_id = protein["protein_id"]
        filtered_domains = []
        region_size = len(domains)
        last_idx = region_size - 1
        for idx, d in enumerate(domains):
            d_hash_id = d["domain_id"]
            d_label = d["label"]
            d_score = d["score"]
            d_start = d["protein_start"]
            d_stop = d["protein_stop"]
            # filter uncertain hits
            keep = False
            # usually determined by high domain scores
            if d_score >= min_domain_score:
                keep = True
            # flexible condition to keep T
            if d_label == "T":
                if idx > 0 and domains[idx - 1]["label"] in [
                    "KS",
                    "AT",
                    "KR",
                ]:
                    keep = True
            # flexible condition to keep TE as long as its last domain
            if d_label == "TE" and idx == last_idx:
                keep = True
            if d_label == "TE" and idx != last_idx:
                keep = False
            # for A domains, consider context relationships
            if d_label == "A" and keep == False:
                # starter A domain
                if idx == 0 and region_size > 1:
                    if domains[idx + 1]["label"] in ["T", "C", "KR"]:
                        keep = True
                # A domain following condensation
                else:
                    if domains[idx - 1]["label"] in ["C", "KR"]:
                        keep = True
            # skip if domain is determined to be unconfident
            if keep == False:
                continue
            # store domain
            if d_label in ["A", "AT"]:
                substrates = domain_knn_lookup[d_label].get(d_hash_id, [])
                filtered_domains.append(
                    Domain(
                        label=d_label,
                        protein_id=protein_id,
                        start=d_start,
                        stop=d_stop,
                        substrates=substrates,
                    )
                )
            elif d_label in ["KR", "DH", "ER"]:
                functional = domain_knn_lookup[d_label].get(d_hash_id, True)
                filtered_domains.append(
                    Domain(
                        label=d_label,
                        protein_id=protein_id,
                        start=d_start,
                        stop=d_stop,
                        functional=functional,
                    )
                )
            elif d_label == "T":
                subclass = domain_knn_lookup[d_label].get(d_hash_id, None)
                filtered_domains.append(
                    Domain(
                        label=d_label,
                        protein_id=protein_id,
                        start=d_start,
                        stop=d_stop,
                        subclass=subclass,
                    )
                )
            else:
                filtered_domains.append(
                    Domain(
                        label=d_label,
                        protein_id=protein_id,
                        start=d_start,
                        stop=d_stop,
                    )
                )
        # calculate modules
        if len(filtered_domains) > 0:
            modules = Module.load_from_domains(filtered_domains)
            if len(modules) > 0:
                protein_to_modules.append(
                    {
                        "protein_id": protein_id,
                        "modules": [m.report for m in modules],
                    }
                )
    export_fp = f"{ibis_dir}/module_predictions.json"
    json.dump(protein_to_modules, open(export_fp, "w"))
    return True


def run_on_files(
    filenames: List[str],
    output_dir: str,
    domain_preds_created: bool,
    adenylation_preds_created: bool,
    acyltransferase_preds_created: bool,
    ketosynthase_preds_created: bool,
    ketoreductase_preds_created: bool,
    dehydratase_preds_created: bool,
    enoylreductase_preds_created: bool,
    thiolation_preds_created: bool,
    cpu_cores: int,
) -> bool:
    if domain_preds_created == False:
        raise ValueError("Domain predictions not created")
    if adenylation_preds_created == False:
        raise ValueError("Adenylation predictions not created")
    if acyltransferase_preds_created == False:
        raise ValueError("Acyltransferase predictions not created")
    if ketosynthase_preds_created == False:
        raise ValueError("Ketosynthase predictions not created")
    if ketoreductase_preds_created == False:
        raise ValueError("Ketoreductase predictions not created")
    if dehydratase_preds_created == False:
        raise ValueError("Dehydratase predictions not created")
    if enoylreductase_preds_created == False:
        raise ValueError("Enoylreductase predictions not created")
    if thiolation_preds_created == False:
        raise ValueError("Thiolation predictions not created")
    ibis_dirs = [
        f"{output_dir}/{name}"
        for name in filenames
        if os.path.exists(f"{output_dir}/{name}/module_predictions.json")
        == False
    ]
    if len(ibis_dirs) == 0:
        return True
    # parallel prediction
    pool = Pool(cpu_cores)
    process = pool.imap_unordered(predict_modules_from_ibis_dir, ibis_dirs)
    [p for p in tqdm(process, total=len(ibis_dirs), desc="Predicting modules")]
    pool.close()
    return True
