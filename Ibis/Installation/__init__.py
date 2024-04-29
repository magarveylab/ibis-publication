import os
from glob import glob

import requests
from dotenv import find_dotenv, get_key
from tqdm import tqdm

from Ibis import curdir


def setup_qdrant_docker():
    # docker commands
    # docker run -it -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
    host = get_key(find_dotenv(), "QDRANT_HOST")
    port = get_key(find_dotenv(), "QDRANT_PORT")
    node_url = f"http://{host}:{port}"
    local_snapshot_paths = glob(f"{curdir}/QdrantSnapshots/*")
    for snapshot_path in tqdm(local_snapshot_paths):
        snapshot_name = snapshot_path.split("/")[-1]
        collection_name = snapshot_name.split("-")[0]
        requests.post(
            f"{node_url}/collections/{collection_name}/snapshots/upload?priority=snapshot",
            files={"snapshot": (snapshot_name, open(snapshot_path, "rb"))},
        )


if __name__ == "__main__":
    setup_qdrant_docker()
