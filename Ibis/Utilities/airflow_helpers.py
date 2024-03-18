import subprocess as sp
import time


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


def wait_for_gpu(gpu_id: int):
    while gpu_id not in get_free_gpus():
        # wait for 3 minutes
        print("Waiting for free gpu ...")
        time.sleep(180)
    return True
