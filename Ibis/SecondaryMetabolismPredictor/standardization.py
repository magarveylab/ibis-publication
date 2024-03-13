from Ibis import curdir
import pandas as pd
from typing import Dict, Set

table_dir = f"{curdir}/SecondaryMetabolismPredictor/tables"


def get_internal_chemotype_lookup(
    table_fp: str = f"{table_dir}/internal_chemotype_standardized.csv",
) -> Dict[str, Set[str]]:
    lookup = {
        "Alkaloid": set(),
        "NonRibosomalPeptide": set(),
        "Polyketide": set(),
        "Ripp": set(),
        "Saccharide": set(),
        "Terpene": set(),
        "Other": set(),
    }
    for rec in pd.read_csv(table_fp).to_dict("records"):
        lookup[rec["standardized_label"]].add(rec["label"])
    return lookup


def get_mibig_chemotype_standardization(
    table_fp: str = f"{table_dir}/mibig_chemotype_standardized.csv",
) -> Dict[str, str]:
    df = pd.read_csv(table_dir)
    return dict(zip(df.standardized_label, df.label))
