import os
from glob import glob

import requests

node_url = "http://localhost:6333"
local_snapshot_paths = glob("snapshots/*.snapshot")
for snapshot_path in local_snapshot_paths:
    snapshot_name = os.path.basename(snapshot_path)
    collection_name = snapshot_name.split("-")[0]
    requests.post(
        f"{node_url}/collections/{collection_name}/snapshots/upload?priority=snapshot",
        files={"snapshot": (snapshot_name, open(snapshot_path, "rb"))},
    )
