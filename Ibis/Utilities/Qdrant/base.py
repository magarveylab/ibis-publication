import os
import time
from typing import Any, Dict, List, Literal, Union

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import CollectionStatus, SearchRequest
from tqdm import tqdm

from Ibis.Utilities.Qdrant.datastructs import DataQuery, SearchResponse
from Ibis.Utilities.Qdrant.parameters import (
    default_dist_metric,
    default_search_params,
)


# helper functions
def batchify(l: list, bs: int = 1000):
    return [l[x : x + bs] for x in range(0, len(l), bs)]


# connection to client
client = QdrantClient(
    host=os.environ.get("QDRANT_HOST"),
    port=os.environ.get("QDRANT_PORT"),
    timeout=180,
)


class QdrantBase:
    def __init__(
        self,
        collection_name: str,
        label_alias: str = None,
        embedding_dim: int = None,
        memory_strategy: Literal["disk", "memory", "hybrid"] = None,
        memmap_threshold: int = None,
        delete_existing: bool = False,
        **kwargs,
    ):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.label_alias = label_alias
        collections = client.get_collections()
        collection_names = {x.name for x in collections.collections}
        if delete_existing and collection_name in collection_names:
            self.delete_database()
            collection_names = collection_names - set([collection_name])
        if collection_name not in collection_names:
            if embedding_dim is None:
                raise ValueError("embedding_dim must be provided.")
            if memory_strategy is None:
                raise ValueError("memory_strategy must be provided.")
            if label_alias is None:
                raise ValueError("label_alias must be provided.")
            self.create_collection(
                embedding_dim=embedding_dim,
                memory_strategy=memory_strategy,
                memmap_threshold=memmap_threshold,
                **kwargs,
            )
        else:
            self.check_status()

    def create_collection(
        self,
        embedding_dim: int,
        memory_strategy: Literal["disk", "memory", "hybrid"],
        memmap_threshold: int,
        **kwargs,
    ):
        if memory_strategy == "memory":
            on_disk = False
            always_ram = True
            if memmap_threshold is not None:
                print(
                    "Memmap threshold is not recommended with an in-memory \
                    configuration. Recommend rebuilding with \
                    'memmap_threshold' equal to zero."
                )
            else:
                memmap_threshold = 0
        elif memory_strategy == "hybrid":
            on_disk = True
            always_ram = True
        elif memory_strategy == "disk":
            on_disk = True
            always_ram = False
        else:
            raise ValueError(
                f"memory_strategy expects one of 'disk', \
                    'memory', or 'hybrid'. You passed {memory_strategy}"
            )
        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=embedding_dim,
                distance=default_dist_metric,
                on_disk=on_disk,
            ),
            # set indexing disabled before bulk insert
            optimizers_config=models.OptimizersConfigDiff(
                memmap_threshold=memmap_threshold, indexing_threshold=0
            ),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8, always_ram=always_ram
                )
            ),
            **kwargs,
        )

    def _check_status(self):
        self.collection_status = client.get_collection(
            collection_name=self.collection_name
        )

    def check_status(self, timeout: int = 30):
        self._check_status()
        start = time.time()
        while self.collection_status.status.value != "green":
            current = time.time()
            if current - start > timeout:
                break
            time.sleep(5)
            self._check_status()
        if self.collection_status.status.value != "green":
            raise TimeoutError(
                f"""
                There is an issue with the collection as loaded,
                 status checking timed out after {round(current-start, 1)}
                 seconds and the collection never returned green status. 
                 The curent status is {self.collection_status}""".replace(
                    "\n", ""
                )
                .replace("\t", "")
                .replace("    ", "")
            )

    def upload_data_batch(
        self,
        ids: List[int],
        vectors: Union[List[np.array], np.ndarray],
        payloads: List[Dict[str, Any]] = None,
    ):
        self.check_status()
        # convert list to 2d array (if relevant)
        if isinstance(vectors, list):
            vectors = np.array(vectors)
        # Ensure vector is of correct dimensionality for collection
        collection_info = client.get_collection(
            collection_name=self.collection_name
        )
        assert vectors.shape[1] == collection_info.config.params.vectors.size
        # do bulk uploading
        client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=ids,
                # Qdrant python API supports only native python objects - convert to list.
                vectors=vectors.tolist(),
                payloads=payloads,
            ),
        )

    def index_collection(self, indexing_threshold: int = 20000):
        self.check_status()
        # perform indexing
        client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=models.OptimizersConfigDiff(
                indexing_threshold=indexing_threshold
            ),
        )
        collection_info = client.get_collection(
            collection_name=self.collection_name
        )
        start = time.time()
        while collection_info.status != CollectionStatus.GREEN:
            current = time.time()
            print(
                f"Waiting for Collection to finish indexing. \
                {round(current-start, 2)} seconds have elapsed..."
            )
            time.sleep(10)
        print("Indexing complete. Waiting 5 seconds for cleanup.")
        time.sleep(5)

    def get_db_data(
        self,
        return_embeds: bool = False,
        return_data: bool = False,
        data_filter: models.Filter = None,
        limit: int = 100,
    ):
        if data_filter is None:
            data = client.scroll(
                collection_name=self.collection_name,
                with_vectors=return_embeds,
                with_payload=return_data,
                limit=limit,
            )
        else:
            data = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=data_filter,
                with_vectors=return_embeds,
                with_payload=return_data,
                limit=limit,
            )
        return data

    def delete_database(self):
        print(
            f"Permanently deleting collection {self.collection_name} and all associated data..."
        )
        # get db ids
        data = self.get_db_data(
            return_data=False, return_embeds=False, limit=1000000000
        )
        ids = [x.id for x in data[0]]
        if len(ids) != 0:
            print(f"Deleting {len(ids)} vectors...")
            client.delete_vectors(
                collection_name=self.collection_name, points=ids, vectors=[""]
            )
            print(f"Deleting {len(ids)} payloads...")
            for sub_ls in tqdm(batchify(ids, 1000)):
                tmp = client.clear_payload(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(
                        points=sub_ls,
                    ),
                )
            print(f"Deleting {len(ids)} points...")
            for sub_ls in tqdm(batchify(ids, 1000)):
                tmp = client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=sub_ls),
                )
        client.delete_collection(collection_name=self.collection_name)

    def retrieve(
        self,
        ids: List[int],
        return_embeds: bool = False,
        return_data: bool = True,
    ):
        """Retrieve datapoints from Qdrant database when primary key \
            IDs are known."""
        self.check_status()
        dat = client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_vectors=return_embeds,
            with_payload=return_data,
        )
        if return_embeds:
            for x in dat:
                # convert list to array.
                x.vector = np.array(x.vector)
        return dat

    def batch_search(
        self,
        queries: List[DataQuery],
        batch_size: int,
        max_results: int,
        return_embeds: bool = False,
        return_data: bool = True,
        query_filter: models.Filter = None,
        consistency=None,
        distance_cutoff: float = None,  # NOT squared distance!!!
        ignore_self_matches: bool = True,
    ) -> List[SearchResponse]:
        self.check_status()
        search_params = models.SearchParams(**default_search_params)
        batches = batchify(queries, bs=batch_size)
        all_results = []
        for batch in tqdm(batches, leave=False):
            batch_qids = []
            batch_reshape = []
            for entry in batch:
                batch_qids.append(entry["query_id"])
                batch_reshape.append(
                    SearchRequest(
                        vector=entry["embedding"].tolist(),
                        limit=max_results,
                        with_payload=True,
                        with_vector=return_embeds,
                        filter=query_filter,
                        score_threshold=distance_cutoff,
                        params=search_params,
                    )
                )
            results = client.search_batch(
                collection_name=self.collection_name,
                requests=batch_reshape,
                consistency=consistency,
                timeout=30,
            )
            for qid, result in zip(batch_qids, results):
                hits = []
                for r in result:
                    dist = r.score
                    data = r.payload if return_data else {}
                    if r.id == qid and ignore_self_matches:
                        continue
                    hits.append(
                        {
                            "subject_id": r.id,
                            "distance": r.score,
                            "label": r.payload[self.label_alias],
                            "data": data,
                        }
                    )
                all_results.append({"query_id": qid, "hits": hits})
        return all_results
