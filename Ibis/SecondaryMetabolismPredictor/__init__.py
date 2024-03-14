from Ibis.SecondaryMetabolismPredictor.datastructs import OrfInput
from Ibis.SecondaryMetabolismPredictor.preprocess import (
    get_tensors_from_genome,
)
from Ibis.SecondaryMetabolismPredictor.pipeline import (
    MibigMetabolismPredictorPipeline,
    InternalMetabolismPredictorPipeline,
)
from Ibis.SecondaryMetabolismPredictor.postprocess import (
    call_bgcs_by_proximity,
    call_bgcs_by_chemotype,
    add_orfs_to_bgcs,
)
import os
from typing import List, Optional


def run_on_orfs(
    orfs: List[OrfInput],
    gpu_id: Optional[int] = None,
    internal_pipeline: Optional[InternalMetabolismPredictorPipeline] = None,
    mibig_pipeline: Optional[MibigMetabolismPredictorPipeline] = None,
    min_threshold: int = 10000,
    ignore_orfs_wo_embedding: bool = False,
) -> List[ClusterOutput]:
    # load pipeline
    if internal_pipeline == None:
        internal_pipeline = InternalMetabolismPredictorPipeline(gpu_id=gpu_id)
    if mibig_pipeline == None:
        mibig_pipeline = MibigMetabolismPredictorPipeline(gpu_id=gpu_id)
    # enumerate orf ids (need it as ints for tensor stacks)
    orf_traceback = {}
    for idx, o in enumerate(orfs):
        o["orf_id"] = idx
        orf_traceback[idx] = (
            f"{o['contig_id']}_{o['contig_start']}_{o['contig_stop']}"
        )
    # boundary predictions with secondary metabolism
    orf_meta = {o["orf_id"]: o for o in orfs}
    internal_annotated_orfs = internal_pipeline(orfs=orfs)
    # call bgcs
    proximity_based_bgcs = call_bgcs_by_proximity(
        all_orfs=internal_annotated_orfs, min_threshold=min_threshold
    )
    # get batched data - each batch should correspond to called bgc
    batched_data = []
    for bgc in proximity_based_bgcs:
        bgc = [orf_meta[o] for o in bgc]
        batched_data.extend(get_tensors_from_genome(orfs=bgc, window_size=500))
    # chemotype predictions
    if len(batched_data) > 0:
        mibig_annotated_orfs = mibig_pipeline(
            orfs=orfs, batched_data=batched_data
        )
        mibig_lookup = {o["orf_id"]: o for o in mibig_annotated_orfs}
    else:
        mibig_lookup = {}
    chemotype_based_bgcs = call_bgcs_by_chemotype(
        all_orfs=secondary_annotated_orfs,
        mibig_lookup=mibig_lookup,
        min_threshold=min_threshold,
    )
    # assign orfs to regions
    chemotype_based_bgcs = add_orfs_to_bgcs(
        regions=chemotype_based_bgcs, orfs=orfs, orf_traceback=orf_traceback
    )
    return chemotype_based_bgcs


def run_on_embedding_fps(
    filenames: List[str], output_dir: str, gpu_id: int = 0
):
    bgc_filenames = []
    # load pipeline
    internal_pipeline = InternalMetabolismPredictorPipeline(gpu_id=gpu_id)
    mibig_pipeline = MibigMetabolismPredictorPipeline(gpu_id=gpu_id)
    # analysis
    for embedding_fp in filenames:
        name = embedding_fp.split("/")[-2]
        prodigal_fp = f"{output_dir}/{name}/prodigal.json"
        export_fp = f"{output_dir}/{name}/bgc_predictions.json"
        if os.path.exists(export_fp) == False:
            # create embedding lookup
            embedding_looup = {}
            for protein in pickle.load(open(embedding_fp, "rb")):
                embedding_lookup[protein["protein_id"]] = protein["embedding"]
            # create input data
            orfs = []
            for orf in json.load(open(prodigal_fp)):
                protein_id = orf["protein_id"]
                if protein_id not in embedding_lookup:
                    continue
                embedding = embedding_lookup[protein_id]
                orfs.append(
                    {
                        "contig_id": orf["contig_id"],
                        "contig_start": orf["contig_start"],
                        "contig_stop": orf["contig_stop"],
                        "embedding": embedding,
                    }
                )
            # run pipeline
            out = run_on_orfs(
                orfs=orfs,
                internal_pipeline=internal_pipeline,
                mibig_pipeline=mibig_pipeline,
            )
            with open(export_fp, "w") as f:
                json.dump(out, f)
        bgc_filenames.append(export_fp)
    # delete pipeline
    del internal_pipeline
    del mibig_pipeline
    return bgc_filenames
