import json
import os
from functools import partial
from multiprocessing import Pool
from typing import List

from tqdm import tqdm

from Ibis.PrimaryMetabolismPredictor.annotation import (
    KOAnnotator,
    annotate_enzyme_orfs_with_pathways,
)
from Ibis.PrimaryMetabolismPredictor.preprocess import (
    merge_protein_annotations,
)


def run_on_pyrodigal_fp(
    pyrodigal_fp: str,
    output_dir: str,
    ec_homology_cutoff: float = 0.6,
    ko_homology_cutoff: float = 0.2,
    module_score: float = 0.7,
    allow_inf_ec: bool = True,
):
    filename = os.path.basename(os.path.dirname(pyrodigal_fp))
    export_fp = os.path.join(
        output_dir, filename, "primary_metabolism_predictions.json"
    )
    ko_fp = os.path.join(output_dir, filename, "ko_predictions.json")
    ec_fp = os.path.join(output_dir, filename, "ec_predictions.json")
    embed_fp = os.path.join(output_dir, filename, "protein_embedding.pkl")
    if os.path.exists(export_fp) == False:
        # do things.
        annots = merge_protein_annotations(
            pyrodigal_fp=pyrodigal_fp,
            ko_fp=ko_fp,
            ec_fp=ec_fp,
            embed_fp=embed_fp,
        )
        ec_results = annotate_enzyme_orfs_with_pathways(
            orfs=annots,
            homology_score_threshold=ec_homology_cutoff,
            module_completeness_threshold=module_score,
            annotate_kegg=True,
        )
        ko_annotator = KOAnnotator(
            allow_inferred_kegg_ecs=allow_inf_ec,
            ec_homology_cutoff=ec_homology_cutoff,
            ko_homology_cutoff=ko_homology_cutoff,
            module_completeness_threshold=module_score,
        )
        ko_results = ko_annotator.run_annotation(genome_orfs=annots)
        json.dump(
            {"ko_results": ko_results, "ec_results": ec_results},
            open(export_fp, "w"),
        )
    return export_fp


def parallel_run_on_pyrodigal_fps(
    filenames: List[str],
    output_dir: str,
    cpu_cores: int = 1,
):
    funct = partial(run_on_pyrodigal_fp, output_dir=output_dir)
    pool = Pool(cpu_cores)
    process = pool.imap_unordered(funct, filenames)
    out = [p for p in tqdm(process, total=len(filenames))]
    pool.close()
    return out
