import json
import os
import subprocess as sp
import time
from typing import List

import requests
from dotenv import find_dotenv, get_key

airflow_base_url = get_key(find_dotenv(), "AIRFLOW_BASE_URL")
airflow_auth_token = get_key(find_dotenv(), "AIRFLOW_AUTH_TOKEN")


def get_free_gpus(min_memory: int = 100):
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_free_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    return set(
        i
        for i, x in enumerate(memory_free_info)
        if int(x.split()[0]) < min_memory
    )


def wait_for_gpu(gpus: List[int]) -> int:
    gpus = set(gpus)
    free_gpus = get_free_gpus() & gpus
    while len(free_gpus) == 0:
        # wait for 3 minutes
        print("Waiting for free gpu ...")
        time.sleep(180)
        free_gpus = get_free_gpus() & gpus
    return list(free_gpus)[0]


def split(a: list, n: int):
    k, m = divmod(len(a), n)
    return (
        a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)
    )


def batchify(a: list, bs=10):
    return [a[x : x + bs] for x in range(0, len(a), bs)]


def submit_genomes_to_airflow(
    nuc_fasta_fps: List[str],
    gpus: List[int] = [0],
    cpu_cores: int = 4,
):
    headers = {
        "authorization": f"Basic {airflow_auth_token}",
        "content-type": "application/json",
    }
    url = f"{airflow_base_url}/ibis_submission/dagRuns"
    data = {
        "conf": {
            "nuc_fasta_filenames": nuc_fasta_fps,
            "gpus": gpus,
            "cpu_cores": cpu_cores,
        }
    }
    r = requests.post(url, headers=headers, data=json.dumps(data))
