import json
import os
import pickle
import shutil
from functools import partial
from multiprocessing import Pool
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

########################################################################
# General functions
########################################################################


def run_on_orfs(
    orfs: List[OrfInput],
    gpu_id: Optional[int] = None,
    internal_pipeline: Optional[InternalMetabolismPredictorPipeline] = None,
    mibig_pipeline: Optional[MibigMetabolismPredictorPipeline] = None,
    min_threshold: int = 10000,
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


########################################################################
# Airflow inference functions
########################################################################


def prepare_orfs_for_pipeline_from_single_file(
    name: str,
    output_dir: str,
) -> bool:
    final_fp = f"{output_dir}/{name}/bgc_predictions.json"
    if os.path.exists(final_fp):
        return True
    export_dir = f"{output_dir}/{name}/bgc_predictions_tmp"
    os.makedirs(export_dir, exist_ok=True)
    export_fp = f"{export_dir}/input.pkl"
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
        # add orf enumerated ids
        for idx, o in enumerate(orfs):
            o["orf_id"] = idx
        # temp deposit
        with open(export_fp, "wb") as f:
            pickle.dump(orfs, f)
    return True


def parallel_prepare_orfs_for_pipeline_from_files(
    filenames: List[str],
    output_dir: str,
    prodigal_preds_created: bool,
    protein_embs_created: bool,
    cpu_cores: int = 1,
) -> bool:
    if prodigal_preds_created == False:
        raise ValueError("Prodigal predictions not created")
    if protein_embs_created == False:
        raise ValueError("Protein embeddings not created")
    funct = partial(
        prepare_orfs_for_pipeline_from_single_file, output_dir=output_dir
    )
    pool = Pool(cpu_cores)
    process = pool.imap_unordered(funct, filenames)
    out = [
        p
        for p in tqdm(
            process,
            total=len(filenames),
            leave=False,
            desc="Preparing orfs for secondary metabolism detection",
        )
    ]
    pool.close()
    return True


def run_internal_metabolism_pipeline_on_files(
    filenames: List[str], output_dir: str, orfs_prepared: bool, gpu_id: int = 0
) -> bool:
    if orfs_prepared == False:
        raise ValueError("Orfs not prepared for cluster caller")
    internal_pipeline = InternalMetabolismPredictorPipeline(gpu_id=gpu_id)
    for name in tqdm(
        filenames, leave=False, desc="Annotate orfs with internal metabolism"
    ):
        final_fp = f"{output_dir}/{name}/bgc_predictions.json"
        if os.path.exists(final_fp):
            continue
        export_dir = f"{output_dir}/{name}/bgc_predictions_tmp"
        export_fp = f"{export_dir}/internal_annotated_orfs.pkl"
        if os.path.exists(export_fp) == False:
            prepared_orfs_fp = f"{export_dir}/input.pkl"
            orfs = pickle.load(open(prepared_orfs_fp, "rb"))
            internal_annotated_orfs = internal_pipeline(orfs=orfs)
            with open(export_fp, "wb") as f:
                pickle.dump(internal_annotated_orfs, f)
    del internal_pipeline
    return True


def call_bgcs_by_proximity_from_single_file(
    name: str, output_dir: str, min_threshold: int = 10000
) -> bool:
    final_fp = f"{output_dir}/{name}/bgc_predictions.json"
    if os.path.exists(final_fp):
        return True
    export_dir = f"{output_dir}/{name}/bgc_predictions_tmp"
    export_fp = f"{export_dir}/proximity_based_bgcs.pkl"
    if os.path.exists(export_fp) == False:
        internal_annotated_orfs = pickle.load(
            open(f"{export_dir}/internal_annotated_orfs.pkl", "rb")
        )
        proximity_based_bgcs = call_bgcs_by_proximity(
            all_orfs=internal_annotated_orfs, min_threshold=min_threshold
        )
        with open(export_fp, "wb") as f:
            pickle.dump(proximity_based_bgcs, f)
    return True


def parallel_call_bgcs_by_proximity_from_files(
    filenames: List[str],
    output_dir: str,
    internal_orf_annos_prepared: bool,
    cpu_cores: int = 1,
) -> bool:
    if internal_orf_annos_prepared == False:
        raise ValueError("Internal orf annotations not prepared")
    funct = partial(
        call_bgcs_by_proximity_from_single_file, output_dir=output_dir
    )
    pool = Pool(cpu_cores)
    process = pool.imap_unordered(funct, filenames)
    out = [
        p
        for p in tqdm(
            process,
            total=len(filenames),
            leave=False,
            desc="Calling bgcs by proximity",
        )
    ]
    pool.close()
    return True


def run_mibig_metabolism_pipeline_on_files(
    filenames: List[str],
    output_dir: str,
    orfs_prepared: bool,
    proximity_based_bgcs_prepared: bool,
    gpu_id: int = 0,
) -> bool:
    if orfs_prepared == False:
        raise ValueError("Orfs not prepared for cluster caller")
    if proximity_based_bgcs_prepared == False:
        raise ValueError("Proximity based bgcs not prepared")
    mibig_pipeline = MibigMetabolismPredictorPipeline(gpu_id=gpu_id)
    for name in tqdm(
        filenames, leave=False, desc="Annotate orfs with mibig metabolism"
    ):
        final_fp = f"{output_dir}/{name}/bgc_predictions.json"
        if os.path.exists(final_fp):
            continue
        export_dir = f"{output_dir}/{name}/bgc_predictions_tmp"
        export_fp = f"{export_dir}/mibig_annotated_orfs.pkl"
        if os.path.exists(export_fp) == False:
            proximity_based_bgcs = pickle.load(
                open(f"{export_dir}/proximity_based_bgcs.pkl", "rb")
            )
            orfs = pickle.load(open(f"{export_dir}/input.pkl", "rb"))
            orf_meta = {o["orf_id"]: o for o in orfs}
            batched_data = []
            for bgc in proximity_based_bgcs:
                bgc = [orf_meta[o] for o in bgc]
                batched_data.extend(
                    get_tensors_from_genome(orfs=bgc, window_size=500)
                )
            # chemotype predictions
            if len(batched_data) > 0:
                mibig_annotated_orfs = mibig_pipeline(
                    orfs=orfs, batched_data=batched_data
                )
                mibig_lookup = {o["orf_id"]: o for o in mibig_annotated_orfs}
            else:
                mibig_lookup = {}
            with open(export_fp, "wb") as f:
                pickle.dump(mibig_lookup, f)
    del mibig_pipeline
    return True


def call_bgcs_by_chemotype_from_single_file(
    name: str, output_dir: str, min_threshold: int = 10000
) -> bool:
    final_fp = f"{output_dir}/{name}/bgc_predictions.json"
    export_dir = f"{output_dir}/{name}/bgc_predictions_tmp"
    if os.path.exists(final_fp) == False:
        # data inputs
        internal_annotated_orfs = pickle.load(
            open(f"{export_dir}/internal_annotated_orfs.pkl", "rb")
        )
        mibig_lookup = pickle.load(
            open(f"{export_dir}/mibig_annotated_orfs.pkl", "rb")
        )
        orfs = pickle.load(open(f"{export_dir}/input.pkl", "rb"))
        orf_traceback = {}
        for o in orfs:
            orf_id = o["orf_id"]
            contig_id = o["contig_id"]
            contig_start = o["contig_start"]
            contig_stop = o["contig_stop"]
            orf_traceback[orf_id] = f"{contig_id}_{contig_start}_{contig_stop}"
        # analysis
        chemotype_based_bgcs = call_bgcs_by_chemotype(
            all_orfs=internal_annotated_orfs,
            mibig_lookup=mibig_lookup,
            min_threshold=min_threshold,
        )
        # assign orfs to regions
        chemotype_based_bgcs = add_orfs_to_bgcs(
            regions=chemotype_based_bgcs,
            orfs=orfs,
            orf_traceback=orf_traceback,
        )
        with open(final_fp, "w") as json_data:
            json.dump(chemotype_based_bgcs, json_data)
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    return True


def parallel_call_bgcs_by_chemotype_from_files(
    filenames: List[str],
    output_dir: str,
    orfs_prepared: bool,
    internal_orf_annos_prepared: bool,
    mibig_orf_annos_prepared: bool,
    cpu_cores: int = 1,
) -> bool:
    if orfs_prepared == False:
        raise ValueError("Orfs not prepared for cluster caller")
    if internal_orf_annos_prepared == False:
        raise ValueError("Internal orf annotations not prepared")
    if mibig_orf_annos_prepared == False:
        raise ValueError("Mibig orf annotations not prepared")
    funct = partial(
        call_bgcs_by_chemotype_from_single_file, output_dir=output_dir
    )
    pool = Pool(cpu_cores)
    process = pool.imap_unordered(funct, filenames)
    out = [
        p
        for p in tqdm(
            process,
            total=len(filenames),
            leave=False,
            desc="Calling bgcs by chemotype",
        )
    ]
    pool.close()
    return True


########################################################################
# Airflow upload functions
########################################################################


def upload_bgcs_from_files(
    nuc_fasta_fp: str,
    bgc_pred_fp: str,
    prodigal_fp: str,
    domain_pred_fp: str,
    log_dir: str,
    genome_id: Optional[int],
    orfs_uploaded: bool,
    contigs_uploaded: bool,
    genome_uploaded: bool,
) -> bool:
    from Ibis.SecondaryMetabolismPredictor.upload import upload_bgcs

    log_fp = f"{log_dir}/bgcs_uploaded.json"
    if os.path.exists(log_fp) == False:
        # nucleotide sequence lookup
        seq_lookup = {}
        for seq in SeqIO.parse(nuc_fasta_fp, "fasta"):
            seq = str(seq.seq)
            contig_id = xxhash.xxh32(seq).intdigest()
            seq_lookup[contig_id] = seq
        # protein_hash_lookup
        protein_hash_lookup = {}
        for p in json.load(open(prodigal_fp)):
            protein_id = p["protein_id"]
            contig_id = p["contig_id"]
            contig_start = p["contig_start"]
            contig_stop = p["contig_stop"]
            orf_id = f"{contig_id}_{contig_start}_{contig_stop}"
            protein_hash_lookup[orf_id] = protein_id
        # domain meta lookup
        orf_module_count = {}
        for o in json.load(open(domain_pred_fp)):
            protein_id = o["protein_id"]
            labels = [d["label"] for d in o["regions"]]
            pks_modules = max([labels.count("KS"), labels.count("AT")])
            nrps_modules = labels.count("A")
            orf_module_count[protein_id] = pks_modules + nrps_modules
        # add hash_id to bgcs
        bgcs = json.load(open(bgc_pred_fp))
        for c in bgcs:
            contig_id = c["contig_id"]
            contig_start = c["contig_start"]
            contig_stop = c["contig_stop"]
            bgc_seq = seq_lookup[contig_id][contig_start:contig_stop]
            c["hash_id"] = xxhash.xxh32(bgc_seq).intdigest()
            c["orf_count"] = len(c["orfs"])
            c["module_count"] = sum(
                [
                    orf_module_count.get(protein_hash_lookup[o], 0)
                    for o in c["orfs"]
                ]
            )
        upload_bgcs(
            bgcs=bgcs,
            genome_id=genome_id,
            orfs_uploaded=orfs_uploaded,
            contigs_uploaded=contigs_uploaded,
            genome_uploaded=genome_uploaded,
        )
        json.dump({"upload": True}, open(log_fp, "w"))
    return True
