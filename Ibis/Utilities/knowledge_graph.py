from neo4j.exceptions import TransientError
from neomodel import db
import time
import os
from typing import List

# initialize database
neo4j_username = os.environ.get("NEO4J_USERNAME")
neo4j_password = os.environ.get("NEO4J_PASSWORD")
neo4j_host = os.environ.get("NEO4J_HOST")
neo4j_port = os.environ.get("NEO4J_PORT")
neo4j_url = f"bolt://{neo4j_username}:{neo4j_passwordpassword}@{neo4j_host}:{neo4j_port}"
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
    l = [{k: l[k] for k in keys} for i in l]
    # string format
    l = str(l)
    for k in keys:
        l = l.replace(f"""'{k}'""", f"{k}")
        l = l.replace(f'''"{k}"''', f"{k}")
    return l
