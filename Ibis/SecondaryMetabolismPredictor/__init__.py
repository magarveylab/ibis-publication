import json
import os
import pickle
from typing import List, Optional

import xxhash
from Bio import SeqIO
from tqdm import tqdm

from Ibis.SecondaryMetabolismPredictor.datastructs import (
    ClusterOutput,
    OrfInput,
)
from Ibis.SecondaryMetabolismPredictor.pipeline import (
    InternalMetabolismPredictorPipeline,
    MibigMetabolismPredictorPipeline,
)
from Ibis.SecondaryMetabolismPredictor.postprocess import (
    add_orfs_to_bgcs,
    call_bgcs_by_chemotype,
    call_bgcs_by_proximity,
)
from Ibis.SecondaryMetabolismPredictor.preprocess import (
    get_tensors_from_genome,
)
from Ibis.SecondaryMetabolismPredictor.upload import upload_bgcs


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
        all_orfs=internal_annotated_orfs,
        mibig_lookup=mibig_lookup,
        min_threshold=min_threshold,
    )
    # assign orfs to regions
    chemotype_based_bgcs = add_orfs_to_bgcs(
        regions=chemotype_based_bgcs, orfs=orfs, orf_traceback=orf_traceback
    )
    return chemotype_based_bgcs


def run_on_files(
    filenames: List[str],
    output_dir: str,
    prodigal_preds_created: bool,
    protein_embs_created: bool,
    gpu_id: int = 0,
) -> bool:
    if prodigal_preds_created == False:
        raise ValueError("Prodigal predictions not created")
    if protein_embs_created == False:
        raise ValueError("Protein embeddings not created")
    # load pipeline
    internal_pipeline = InternalMetabolismPredictorPipeline(gpu_id=gpu_id)
    mibig_pipeline = MibigMetabolismPredictorPipeline(gpu_id=gpu_id)
    # analysis
    for name in tqdm(filenames):
        export_fp = f"{output_dir}/{name}/bgc_predictions.json"
        if os.path.exists(export_fp) == False:
            prodigal_fp = f"{output_dir}/{name}/prodigal.json"
            embedding_fp = f"{output_dir}/{name}/protein_embedding.pkl"
            # create embedding lookup
            embedding_lookup = {}
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
    # delete pipeline
    del internal_pipeline
    del mibig_pipeline
    return True


def upload_bgcs_from_files(
    nuc_fasta_fp: str,
    bgc_pred_fp: str,
    log_dir: str,
    genome_id: Optional[int],
    orfs_uploaded: bool,
    contigs_uploaded: bool,
    genome_uploaded: bool,
) -> bool:
    log_fp = f"{log_dir}/bgcs_uploaded.json"
    if os.path.exists(log_fp) == False:
        # nucleotide sequence lookup
        seq_lookup = {}
        for seq in SeqIO.parse(nuc_fasta_fp, "fasta"):
            seq = str(seq.seq)
            contig_id = xxhash.xxh32(seq).intdigest()
            seq_lookup[contig_id] = seq
        # add hash_id to bgcs
        bgcs = json.load(open(bgc_pred_fp))
        for c in bgcs:
            contig_id = c["contig_id"]
            contig_start = c["contig_start"]
            contig_stop = c["contig_stop"]
            bgc_seq = seq_lookup[contig_id][contig_start:contig_stop]
            c["hash_id"] = xxhash.xxh32(bgc_seq).intdigest()
        upload_bgcs(
            bgcs=bgcs,
            genome_id=genome_id,
            orfs_uploaded=orfs_uploaded,
            contigs_uploaded=contigs_uploaded,
            genome_uploaded=genome_uploaded,
        )
        json.dump({"upload": True}, open(log_fp, "w"))
    return True
