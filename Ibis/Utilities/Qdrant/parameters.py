from qdrant_client.http import models

default_dist_metric = models.Distance.EUCLID

default_search_params = {
    "exact": False,
    "hnsw_ef": None,
    "indexed_only": False,
    "quantization": models.QuantizationSearchParams(
        ignore=False, rescore=True, oversampling=None
    ),
}
