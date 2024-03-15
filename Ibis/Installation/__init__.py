from Ibis import curdir
from glob import glob
import requests
import os


def setup_qdrant_docker():
    host = os.environ.get("QDRANT_HOST")
    port = os.environ.get("QDRANT_PORT")
    node_url = f"http://{host}:{port}"
    local_snapshot_paths = glob(f"{curdir}/QdrantSnapshots/*")
    for snapshot_path in local_snapshot_paths:
        snapshot_name = snapshot_path.split("/")[-1]
        collection_name = snapshot_name.split("-")[0]
        requests.post(
            f"{node_url}/collections/{collection_name}/snapshots/upload?priority=snapshot",
            files={"snapshot": (snapshot_name, open(snapshot_path, "rb"))},
        )
