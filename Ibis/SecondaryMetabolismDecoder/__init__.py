from functools import partial

from Ibis.SecondaryMetabolismDecoder.database import IbisKnownCluster
from Ibis.Utilities.Qdrant.classification import (
    KNNClassification,
    neighborhood_classification,
)

decode_known_bgc = partial(
    KNNClassification,
    qdrant_db=IbisKnownCluster,
    classification_method=neighborhood_classification,
    top_n=1,
    dist_cutoff=14.223,
    apply_cutoff_before_homology=True,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
    return_distance=True,
)
