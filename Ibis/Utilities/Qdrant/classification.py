from collections import Counter
from typing import Callable, Dict, List

from Ibis.Utilities.Qdrant.base import QdrantBase
from Ibis.Utilities.Qdrant.datastructs import (
    DataQuery,
    DistHitResponse,
    KnnOutput,
)


def dist2sim(d: float) -> float:
    return round(1 / (1 + d), 3)


def neighborhood_classification(
    hits: List[DistHitResponse],
    top_n: int = 1,  # top n hits to consider for vector database
    dist_cutoff: float = 0.0,
    apply_cutoff_before_homology: bool = True,
    homology_cutoff: float = 1.0,
    apply_homology_cutoff: bool = False,
    apply_cutoff_after_homology: bool = False,
    return_distance: bool = False,
    return_n: int = 5,  # return n top hits
):
    # only consider top n hits
    hits = sorted(hits, key=lambda x: x["distance"])[:top_n]
    all_labels = []
    lookup = {}
    distance_lookup = {}
    for h in hits:
        # filters based on distance
        if (
            apply_cutoff_before_homology == True
            and h["distance"] > dist_cutoff
        ):
            continue
        label = h["label"]
        distance = h["distance"]
        all_labels.append(label)
        if label not in lookup:
            lookup[label] = []
            distance_lookup[label] = {}
        lookup[label].append(h["distance"])
        distance_lookup[label][distance] = h["subject_id"]
    if len(lookup) == 0:
        return []
    # return most frequent, if tied return the closest in terms of distance
    return_labels = sorted(
        lookup, key=lambda x: (len(lookup[x]), -min(lookup[x])), reverse=True
    )[:return_n]
    response = []
    rank = 1
    for label in return_labels:
        distance = min(lookup[label])
        c = all_labels.count(label)
        homology_score = round(c / top_n, 2)
        # filters based on homology score and final distance
        if apply_homology_cutoff == True:
            if homology_score < homology_cutoff:
                continue
            if apply_cutoff_after_homology == True and distance > dist_cutoff:
                continue
        # output
        output = {
            "label": label,
            "reference_id": distance_lookup[label][distance],
            "homology": homology_score,
            "rank": rank,
        }
        if return_distance == False:
            output["similarity"] = dist2sim(distance)
        else:
            output["distance"] = distance
        response.append(output)
        rank += 1
    return response


def ontology_neighborhood_classification(
    hits: List[DistHitResponse],
    top_n: int = 1,  # top n hits to consider for vector database
    dist_cutoff: float = 0.0,
    apply_cutoff_before_homology: bool = True,
    homology_cutoff: float = 1.0,
    apply_homology_cutoff: bool = False,
    apply_cutoff_after_homology: bool = False,
    return_distance: bool = False,
    return_n: int = 5,  # return n top hits
):
    # only consider top n hits
    hits = sorted(hits, key=lambda x: x["distance"])[:top_n]
    # some reference might have multiple labels - split these cases
    all_labels = []
    all_observed = []
    neighborhood = []
    if dist_cutoff == None:
        apply_cutoff_before_homology = False
    for h in hits:
        if (
            apply_cutoff_before_homology == True
            and h["distance"] > dist_cutoff
        ):
            continue
        observed = set()
        labels = [h["label"]] if isinstance(h["label"], str) else h["label"]
        all_labels.extend(labels)
        for label in labels:
            toks = label.split(".")
            top_level = len(toks)
            breakdown = [".".join(toks[:l]) for l in range(1, top_level + 1)]
            neighborhood.append(
                {
                    "reference_id": h["subject_id"],
                    "label": label,
                    "distance": h["distance"],
                    "breakdown": breakdown,
                }
            )
            observed.update(breakdown)
        all_observed.extend(breakdown)
    # if no hits present at distance cutoff
    if len(neighborhood) == 0:
        return []
    # calculate frequency of observed label
    label_freq = Counter(all_observed)
    # annotate each hit by observed frequency in neighborhood
    for n in neighborhood:
        n["scores"] = [label_freq[e] for e in n["breakdown"]]
    # reorganize into labels
    lookup = {}
    distance_lookup = {}
    for n in neighborhood:
        label = n["label"]
        distance = n["distance"]
        if label not in lookup:
            lookup[label] = {"scores": n["scores"], "distance": []}
            distance_lookup[label] = {}
        lookup[label]["distance"].append(distance)
        distance_lookup[label][distance] = n["reference_id"]
    # choose best observed by frequency and then by distance
    return_labels = sorted(
        lookup,
        key=lambda x: (lookup[x]["scores"], -min(lookup[x]["distance"])),
        reverse=True,
    )[:return_n]
    response = []
    rank = 1
    for label in return_labels:
        c = all_labels.count(label)
        homology_score = round(c / top_n, 2)
        distance = min(lookup[label]["distance"])
        # filters based on homology score and final distance
        if apply_homology_cutoff == True:
            if homology_score < homology_cutoff:
                continue
            if apply_cutoff_after_homology == True and distance > dist_cutoff:
                continue
        # output
        reference_id = distance_lookup[label][distance]
        output = {
            "label": label,
            "reference_id": reference_id,
            "homology": homology_score,
            "rank": rank,
        }
        if return_distance == False:
            output["similarity"] = dist2sim(distance)
        else:
            output["distance"] = distance
        response.append(output)
        rank += 1
    return response


def KNNClassification(
    query_list: List[DataQuery],
    qdrant_db: QdrantBase = None,
    classification_method: Callable = None,
    top_n: int = 1,
    dist_cutoff: float = 500.0,  # arbitrarily large.
    apply_cutoff_before_homology: bool = True,
    homology_cutoff: float = 1.0,
    apply_homology_cutoff: bool = False,
    apply_cutoff_after_homology: bool = False,
    batch_size: int = 100,
    return_distance: bool = False,
    return_n: int = 5,
) -> List[KnnOutput]:
    # Initialize Qdrant Database
    db = qdrant_db()
    # run KNN
    predictions = db.batch_search(
        queries=query_list,
        batch_size=batch_size,
        max_results=top_n,
        return_embeds=False,
        return_data=True,
        distance_cutoff=dist_cutoff,
    )
    # classification
    response = []
    for p in predictions:
        query, hits = p["query_id"], p["hits"]
        if len(hits) > 0:
            cls_result = classification_method(
                hits,
                top_n=top_n,
                dist_cutoff=dist_cutoff,
                apply_cutoff_before_homology=apply_cutoff_before_homology,
                homology_cutoff=homology_cutoff,
                apply_homology_cutoff=apply_homology_cutoff,
                apply_cutoff_after_homology=apply_cutoff_after_homology,
                return_distance=return_distance,
                return_n=return_n,
            )
        else:
            cls_result = []
        response.append({"query_id": query, "predictions": cls_result})
    # terminate connection
    del db
    return response
