import json
import os
import pickle
from typing import List, Optional

from tqdm import tqdm

from Ibis.SecondaryMetabolismEmbedder.datastructs import (
    ClusterEmbeddingOutput,
    ClusterInput,
)
from Ibis.SecondaryMetabolismEmbedder.pipeline import (
    MetabolismEmbedderPipeline,
)


def embed_clusters(
    clusters: List[ClusterInput],
    gpu_id: Optional[int] = None,
    pipeline: Optional[MetabolismEmbedderPipeline] = None,
) -> List[ClusterEmbeddingOutput]:
    # load pipeline
    if pipeline == None:
        pipeline = MetabolismEmbedderPipeline(gpu_id=gpu_id)
    # embed
    out = [pipeline(c) for c in tqdm(clusters, desc="Embedding bgcs")]
    return out


def run_on_files(
    filenames: List[str], output_dir: str, gpu_id: Optional[int] = None
) -> bool:
    # load pipeline
    pipeline = MetabolismEmbedderPipeline(gpu_id=gpu_id)
    # analysis
    for name in tqdm(filenames):
        export_fp = f"{output_dir}/{name}/bgc_embeddings.pkl"
        if os.path.exists(export_fp) == False:
            dom_emb_fp = f"{output_dir}/{name}/domain_embedding.pkl"
            # load domain embeddings
            dom_emb_lookup = {}
            for d in pickle.load(open(dom_emb_fp, "rb")):
                dom_emb_lookup[d["domain_id"]] = d["embedding"]
            # load protein embeddings
            prot_emb_fp = f"{output_dir}/{name}/protein_embedding.pkl"
            prot_emb_lookup = {}
            for p in pickle.load(open(prot_emb_fp, "rb")):
                prot_emb_lookup[p["protein_id"]] = p["embedding"]
            # load domains
            dom_pred_fp = f"{output_dir}/{name}/domain_predictions.json"
            domain_lookup = {}
            for p in json.load(open(dom_pred_fp)):
                protein_id = p["protein_id"]
                domain_lookup[protein_id] = []
                for r in p["regions"]:
                    domain_id = r["domain_id"]
                    domain_lookup[protein_id].append(
                        {
                            "protein_start": r["start"],
                            "protein_stop": r["stop"],
                            "label": r["label"],
                            "embedding": dom_emb_lookup.get(domain_id),
                        }
                    )
            # load orf data
            prodigal_fp = f"{output_dir}/{name}/prodigal.json"
            orf_lookup = {}
            for o in json.load(open(prodigal_fp)):
                contig_id = o["contig_id"]
                contig_start = o["contig_start"]
                contig_stop = o["contig_stop"]
                orf_id = f"{contig_id}_{contig_start}_{contig_stop}"
                protein_id = o["protein_id"]
                orf_lookup[orf_id] = {
                    "contig_id": contig_id,
                    "contig_start": contig_start,
                    "contig_stop": contig_stop,
                    "embedding": prot_emb_lookup.get(protein_id),
                    "domains": domain_lookup.get(protein_id, []),
                }
            # load cluster data
            bgc_pred_fp = f"{output_dir}/{name}/bgc_predictions.json"
            cluster_inputs = []
            for c in json.load(open(bgc_pred_fp)):
                contig_id = c["contig_id"]
                contig_start = c["contig_start"]
                contig_stop = c["contig_stop"]
                cluster_id = f"{contig_id}_{contig_start}_{contig_stop}"
                mibig_chemotypes = c["mibig_chemotypes"]
                internal_chemotypes = c["internal_chemotypes"]
                orfs = [orf_lookup[o] for o in c["orfs"]]
                cluster_inputs.append(
                    {
                        "cluster_id": cluster_id,
                        "mibig_chemotypes": mibig_chemotypes,
                        "internal_chemotypes": internal_chemotypes,
                        "orfs": orfs,
                    }
                )
            # analysis
            out = [pipeline(c) for c in tqdm(cluster_inputs)]
            with open(export_fp, "wb") as f:
                pickle.dump(out, f)
    del pipeline
    return True
