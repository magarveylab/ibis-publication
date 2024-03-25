import subprocess as sp
import time
from typing import List


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
