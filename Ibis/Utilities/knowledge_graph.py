import os
import time
from typing import List, Optional, Set, TypedDict

from dotenv import find_dotenv, get_key
from neo4j.exceptions import TransientError
from neomodel import db
from tqdm import tqdm

# initialize database
neo4j_username = get_key(find_dotenv(), "NEO4J_USERNAME")
neo4j_password = get_key(find_dotenv(), "NEO4J_PASSWORD")
neo4j_host = get_key(find_dotenv(), "NEO4J_HOST")
neo4j_port = get_key(find_dotenv(), "NEO4J_PORT")
neo4j_url = (
    f"bolt://{neo4j_username}:{neo4j_password}@{neo4j_host}:{neo4j_port}"
)
db.set_connection(neo4j_url)


def run_cypher(call: str, num_retries: int = 5, retry_pause: int = 2):
    try_num = 1
    while try_num < num_retries:
        try:
            call = call.replace("\n", "")
            call = " ".join(call.split())
            return db.cypher_query(call)
        except TransientError:
            try_num += 1
            time.sleep(retry_pause)
    else:
        raise TimeoutError(
            "The number of transient errors has exceeded the limit specified by 'retries'"
        )


def batchify(l: list, bs: int = 1000):
    return [l[x : x + bs] for x in range(0, len(l), bs)]


def stringfy_dicts(l: List[dict], keys: List[str]):
    # filter dictionary keys
    l = [{k: i[k] for k in keys} for i in l]
    # string format
    l = str(l)
    for k in keys:
        l = l.replace(f"""'{k}'""", f"{k}")
        l = l.replace(f'''"{k}"''', f"{k}")
    return l


def get_existing_hash_ids(node_type: str, ids: List[int]) -> Set[int]:
    response = run_cypher(
        f"""UNWIND {ids} as row
            MATCH (n: {node_type} {{hash_id: row}})
            RETURN n.hash_id
    """
    )
    present_ids = set(i[0] for i in response[0])
    response = run_cypher(
        f"""UNWIND {list(present_ids)} as row
            MATCH (n: {node_type} {{hash_id: row, embedding: null}})
            RETURN n.hash_id
    """
    )
    ids_with_missing_embeddings = set(i[0] for i in response[0])
    return present_ids - ids_with_missing_embeddings


class EmbeddingDict(TypedDict):
    hash_id: int
    embedding: List[float]


def upload_embeddings(
    node_type: str,
    data: List[EmbeddingDict],
    filter_ids: bool = True,
    bs: int = 10,
):
    # batchify data
    batches = batchify(data, bs=bs)
    # submit batches
    for batch in tqdm(batches, desc=f"Uploading embeddings for {node_type}"):
        # check if ids exist in database
        if filter_ids == True:
            present_ids = set(i["hash_id"] for i in batch)
            existing_ids = get_existing_hash_ids(
                node_type=node_type, ids=[i["hash_id"] for i in batch]
            )
            missing = set(present_ids) - existing_ids
            if len(missing) == 0:
                continue  # skip iteration if no data to upload
            batch = [i for i in batch if i["hash_id"] in missing]
        # create ids
        to_create = [i["hash_id"] for i in batch]
        run_cypher(
            f"""UNWIND {to_create} as row
                MERGE (n: {node_type} {{hash_id: row}})
        """
        )
        # upload embeddings
        run_cypher(
            f"""
            UNWIND {stringfy_dicts(batch, keys=['hash_id', 'embedding'])} as row
            MATCH (n: {node_type} {{hash_id: row.hash_id}})
            CALL db.create.setNodeVectorProperty(n, 'embedding', row.embedding)
        """
        )
