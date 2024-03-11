from collections import deque
from itertools import islice
import math
from typing import List


def sliding_window(iterable, size=2, step=1, fillvalue=None):
    if size < 0 or step < 1:
        raise ValueError
    it = iter(iterable)
    q = deque(islice(it, size), maxlen=size)
    if not q:
        return  # empty iterable or size == 0
    q.extend(fillvalue for _ in range(size - len(q)))  # pad to size
    while True:
        yield iter(q)  # iter() to avoid accidental outside modifications
        try:
            q.append(next(it))
        except StopIteration:  # Python 3.5 pep 479 support
            return
        q.extend(next(it, fillvalue) for _ in range(step - 1))


def slice_proteins(iterable, size=512, step=256) -> List[str]:
    return [
        "".join([x for x in i if x != None])
        for i in sliding_window(iterable, size=size, step=step)
    ]


def batchify(array, bs=10):
    return [array[x : x + bs] for x in range(0, len(array), bs)]


def batchify_tokenized_inputs(tokenized_inputs: dict, bs=10):
    input_ids = batchify(tokenized_inputs["input_ids"], bs=bs)
    attention_mask = batchify(tokenized_inputs["attention_mask"], bs=bs)
    token_type_ids = batchify(tokenized_inputs["token_type_ids"], bs=bs)
    batched_input = []
    for x, y, z in zip(input_ids, attention_mask, token_type_ids):
        batched_input.append(
            {"input_ids": x, "attention_mask": y, "token_type_ids": z}
        )
    return batched_input


def get_indices(min_slice_size: float, sequence: str, lengths: List[int]):
    # skip protein windows that are small
    min_target_size = math.ceil(min_slice_size * len(sequence))
    if min_target_size >= 512:
        min_target_size = 512
    return [idx for idx, l in enumerate(lengths) if l >= min_target_size]
