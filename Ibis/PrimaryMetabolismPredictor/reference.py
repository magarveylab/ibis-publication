import json
import os
import pickle
from typing import Dict, List, Set, Tuple, Union

import pandas as pd

from Ibis import curdir

_dat_dir = os.path.join(curdir, "PrimaryMetabolismPredictor", "dat")


def ec3_converter(ec):
    if ec is None:
        return None
    ec_split = ec.split(".")
    if len(ec_split) == 4:
        # convert 4th level ecs to third level
        return ".".join(ec_split[:3])
    else:
        # if ec above 4th level, don't change.
        return ec


def get_reference_pathways() -> pd.DataFrame:
    df_fp = os.path.join(_dat_dir, "cactus_db_pathways.csv")
    df = pd.read_csv(df_fp)
    df = df[df["ec_numbers"].notna()].copy()
    df["ec_numbers"] = df["ec_numbers"].map(lambda x: set(x.split("|")))
    return df.to_dict("records")


def get_microbeannotator_ko_mods() -> Tuple[Set[str], Dict[str, str]]:
    # from Talos.utils.get_microbeannotator_ko_pathways.py
    ko_dat = pickle.load(
        open(
            os.path.join(_dat_dir, "kofam_mapping.pkl"),
            "rb",
        )
    )
    ko_mod_to_path = {}
    ko_mods = set()
    for ko_p, ko_ms in ko_dat.items():
        for ko_m in ko_ms:
            # Previously split branched ko modules with appended enum split by '_'
            # revert this change to match the KO database exactly.
            if "_" in ko_m:
                clean_ko_m = ko_m.split("_")[0]
                ko_mods.add(clean_ko_m)
                if ko_mod_to_path.get(clean_ko_m) is None:
                    ko_mod_to_path[clean_ko_m] = {}
                if ko_mod_to_path[clean_ko_m].get(ko_m) is None:
                    ko_mod_to_path[clean_ko_m][ko_m] = set()
                ko_mod_to_path[clean_ko_m][ko_m].add(ko_p)
            else:
                ko_mods.add(ko_m)
                if ko_mod_to_path.get(ko_m) is None:
                    ko_mod_to_path[ko_m] = set()
                ko_mod_to_path[ko_m].add(ko_p)
    return ko_mods, ko_mod_to_path
