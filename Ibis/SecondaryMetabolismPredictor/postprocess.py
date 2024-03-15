from Ibis.SecondaryMetabolismPredictor.datastructs import (
    InternalAnnotatedOrfDictWithMeta,
    MibigAnnotatedOrfDictWithMeta,
    ClusterOutput,
    OrfInput,
)
from Ibis.SecondaryMetabolismPredictor.standardization import (
    get_internal_chemotype_lookup,
    get_mibig_chemotype_standardization,
)
from tqdm import tqdm
import networkx as nx
from collections import Counter
from typing import List, Dict

# load standardizations
internal_chemotype_lookup = get_internal_chemotype_lookup()
mibig_chemotype_standardization = get_mibig_chemotype_standardization()


def call_bgcs_by_proximity(
    all_orfs: List[InternalAnnotatedOrfDictWithMeta],
    min_threshold: int = 10000,
) -> List[List[int]]:
    # sort orfs
    all_orfs = sorted(
        all_orfs, key=lambda x: (x["contig_id"], x["contig_start"])
    )
    # filter orfs annotated with secondary metabolism
    secondary_metabolism_orfs = []
    secondary_metabolism_orf_ids = []
    unknown_metabolism_orf_ids = []
    for o in all_orfs:
        if o["secondary"]["label"] == "core":
            secondary_metabolism_orfs.append(o)
            secondary_metabolism_orf_ids.append(o["orf_id"])
        else:
            unknown_metabolism_orf_ids.append(o["orf_id"])
    # orf graph - add edges between adjacent orfs based on predicted metabolism
    G = nx.Graph()
    for idx, o1 in enumerate(secondary_metabolism_orfs):
        o1_id = o1["orf_id"]
        G.add_node(o1_id)
        for o2 in secondary_metabolism_orfs[idx + 1 :]:
            o2_id = o2["orf_id"]
            if o2["contig_id"] != o1["contig_id"]:
                break
            distance = o2["contig_start"] - o1["contig_stop"]
            if distance > min_threshold:
                break
            G.add_edge(o1_id, o2_id)
    # add unassigned orfs to the closest metabolism orf (if within threshold)
    for idx, o1 in tqdm(
        enumerate(all_orfs),
        total=len(all_orfs),
        desc="Assign unannotated orfs to closest metabolism",
    ):
        o1_id = o1["orf_id"]
        if o1_id not in unknown_metabolism_orf_ids:
            continue
        # profile slices
        right_closest = {"orf_id": None, "distance": None}
        left_closest = {"orf_id": None, "distance": None}
        # create slices
        right_slice = all_orfs[idx + 1 :]
        left_slice = all_orfs[:idx]
        left_slice.reverse()
        # find closest metabolism match (right slice)
        for o2 in right_slice:
            o2_id = o2["orf_id"]
            if o2_id not in secondary_metabolism_orfs:
                continue
            if o2["contig_id"] != o1["contig_id"]:
                break
            distance = o2["contig_start"] - o1["contig_stop"]
            if distance > min_threshold:
                break
            right_closest["orf_id"] = o2_id
            right_closest["distance"] = distance
            break
        # find closest metabolism match (left slice)
        for o2 in left_slice:
            o2_id = o2["orf_id"]
            if o2_id not in secondary_metabolism_orfs:
                continue
            if o2["contig_id"] != o1["contig_id"]:
                break
            distance = o1["contig_start"] - o2["contig_stop"]
            if distance > min_threshold:
                break
            left_closest["orf_id"] = o2_id
            left_closest["distance"] = distance
            break
        # create edge
        if (
            right_closest["distance"] == None
            and left_closest["distance"] == None
        ):
            continue
        elif (
            right_closest["distance"] == None
            and left_closest["distance"] != None
        ):
            G.add_edge(o1_id, left_closest["orf_id"])
        elif (
            left_closest["distance"] == None
            and right_closest["distance"] != None
        ):
            G.add_edge(o1_id, right_closest["orf_id"])
        elif left_closest["distance"] < right_closest["distance"]:
            G.add_edge(o1_id, left_closest["orf_id"])
        elif right_closest["distance"] <= left_closest["distance"]:
            G.add_edge(o1_id, right_closest["orf_id"])
    return list(nx.connected_components(G))


def call_bgcs_by_chemotype(
    all_orfs: List[InternalAnnotatedOrfDictWithMeta],
    mibig_lookup: Dict[int, MibigAnnotatedOrfDictWithMeta],
    min_threshold: int = 10000,
    min_frequency: float = 0.1,
) -> ClusterOutput:
    # sort orfs
    all_orfs = sorted(
        all_orfs, key=lambda x: (x["contig_id"], x["contig_start"])
    )
    # filter orfs annotated with secondary metabolism
    G = nx.Graph()
    annotated_orfs = []
    for o in tqdm(all_orfs):
        if o["secondary"]["label"] == "core":
            orf_id = o["orf_id"]
            # mibig chemotypes
            mibig_chemotypes = set(
                mibig_chemotype_standardization[c["label"]]
                for c in mibig_lookup[orf_id]["chemotypes"]
                if c["score"] >= 0.5
            )
            if len(mibig_chemotypes) == 0:
                continue
            # standardized chemotypes
            standardized_chemotypes = set(
                i
                for c in mibig_chemotypes
                for i in internal_chemotype_lookup[c]
            )
            # internal chemotypes
            if o["chemotype"] == None:
                internal_chemotypes = set()
            elif o["chemotype"]["score"] >= 0.5:
                if o["chemotype"]["label"] == "PKS-NRPS":
                    internal_chemotypes = {
                        "NonRibosomalPeptide",
                        "TypeIPolyketide",
                    }
                else:
                    internal_chemotypes = {o["chemotype"]["label"]}
            else:
                internal_chemotypes = set()
            internal_chemotypes = internal_chemotypes & standardized_chemotypes
            # cache
            annotated_orfs.append(orf_id)
            G.add_node(
                orf_id,
                contig_id=o["contig_id"],
                contig_start=o["contig_start"],
                contig_stop=o["contig_stop"],
                mibig_chemotypes=mibig_chemotypes,
                internal_chemotypes=internal_chemotypes,
            )
    # add edges
    for idx, o1_id in tqdm(
        enumerate(annotated_orfs),
        total=len(annotated_orfs),
        desc="Draw connections between same metabolism and spatially close orfs",
    ):
        o1 = G.nodes[o1_id]
        for o2_id in annotated_orfs[idx + 1 :]:
            o2 = G.nodes[o2_id]
            if o2["contig_id"] != o1["contig_id"]:
                break
            distance = o2["contig_start"] - o1["contig_stop"]
            if distance > min_threshold:
                break
            mibig_overlap = o1["mibig_chemotypes"] & o2["mibig_chemotypes"]
            if len(mibig_overlap) == 0:
                continue
            elif mibig_overlap == {"Other"}:
                internal_overlap = (
                    o1["internal_chemotypes"] & o2["internal_chemotypes"]
                )
                if len(internal_overlap) > 0:
                    G.add_edge(o1_id, o2_id)
                else:
                    continue
            elif len(mibig_overlap) > 0:
                G.add_edge(o1_id, o2_id)
    # call communities with networkx
    communities = list(nx.connected_components(G))
    out = []
    for bgc in communities:
        # start - stop
        sorted_orfs = sorted(bgc, key=lambda x: G.nodes[x]["contig_start"])
        orf_start, orf_stop = sorted_orfs[0], sorted_orfs[-1]
        start = G.nodes[orf_start]["contig_start"]
        stop = G.nodes[orf_stop]["contig_stop"]
        # chemotypes
        frequency = Counter(
            [c for o in sorted_orfs for c in G.nodes[o]["mibig_chemotypes"]]
        )
        total = sum(frequency.values())
        passed = {
            x for x, y in frequency.items() if y / total >= min_frequency
        }
        mibig_chemotypes = passed if len(passed) > 0 else set(frequency.keys())
        standardized_chemotypes = set(
            i for c in mibig_chemotypes for i in internal_chemotype_lookup[c]
        )
        internal_chemotypes = set(
            c for o in sorted_orfs for c in G.nodes[o]["internal_chemotypes"]
        )
        internal_chemotypes = internal_chemotypes & standardized_chemotypes
        # properties
        contig_id = G.nodes[orf_start]["contig_id"]
        num_annotated_orfs = len(sorted_orfs)
        # cache
        out.append(
            {
                "contig_id": contig_id,
                "contig_start": start,
                "contig_stop": stop,
                "mibig_chemotypes": sorted(mibig_chemotypes),
                "internal_chemotypes": sorted(internal_chemotypes),
                "num_annotated_orfs": num_annotated_orfs,
            }
        )
    return out


def add_orfs_to_bgcs(
    regions: List[ClusterOutput],
    orfs: List[OrfInput],
    orf_traceback: Dict[int, str],
):
    # reorganize data
    all_contig_ids = set(o["contig_id"] for o in orfs)
    contig_to_regions = {c: [] for c in all_contig_ids}
    contig_to_orfs = {c: [] for c in all_contig_ids}
    for r in regions:
        r["orfs"] = []
        contig_to_regions[r["contig_id"]].append(r)
    for o in orfs:
        contig_to_orfs[o["contig_id"]].append(o)
    # sort contigs and orfs
    for c in all_contig_ids:
        contig_to_regions[c] = sorted(
            contig_to_regions[c],
            key=lambda x: (x["contig_start"], x["contig_stop"]),
        )
        contig_to_orfs[c] = sorted(
            contig_to_orfs[c],
            key=lambda x: (x["contig_start"], x["contig_stop"]),
        )
    # assign orfs to regions
    out = []
    for c in all_contig_ids:
        for r in contig_to_regions[c]:
            r_start = r["contig_start"]
            r_stop = r["contig_stop"]
            r["orfs"] = []
            for o in contig_to_orfs[c]:
                if o["contig_start"] >= r_start:
                    if o["contig_stop"] > r_stop:
                        break
                    else:
                        r["orfs"].append(orf_traceback[o["orf_id"]])
                else:
                    continue
            out.append(r)
    return out
