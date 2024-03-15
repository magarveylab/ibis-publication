import json
import os
import re
from typing import Dict, List, Union

import pandas as pd
from tqdm import tqdm

from Ibis.PrimaryMetabolismPredictor.datastructs import (
    AnnotationOutput,
    EnzymeData,
    EnzymeKOData,
)
from Ibis.PrimaryMetabolismPredictor.reference import (
    _dat_dir,
    ec3_converter,
    get_reference_pathways,
)


def annotate_pathways(
    enzyme_orfs: List[EnzymeKOData],
    ec_homology_cutoff: float = 0.6,
    ko_homology_cutoff: float = 0.2,
    module_score: float = 0.7,
    allow_inf_ec: bool = True,
    save_fp: str = None,
):
    ec_results = annotate_enzyme_orfs_with_pathways(
        orfs=enzyme_orfs,
        homology_score_threshold=ec_homology_cutoff,
        module_completeness_threshold=module_score,
        annotate_kegg=True,
    )
    ko_annotator = KOAnnotator(
        allow_inferred_kegg_ecs=allow_inf_ec,
        ec_homology_cutoff=ec_homology_cutoff,
        ko_homology_cutoff=ko_homology_cutoff,
        module_completeness_threshold=module_score,
    )
    ko_results = ko_annotator.run_annotation(genome_orfs=enzyme_orfs)
    json.dump(
        {"ko_results": ko_results, "ec_results": ec_results},
        open(save_fp, "w"),
    )


def get_genome_ec_lookup(
    orfs: List[EnzymeData], homology_score_threshold: float = 0.6
) -> Dict[str, Union[str, int]]:
    # generate EC lookups for genome's orfs.
    ec_to_orf = {}
    for orf in orfs:
        homol_score = orf.get("homology_score")
        if homol_score is None:
            continue  # catch non-enzymes (e.g. KOs with no EC)
        if homol_score < homology_score_threshold:
            continue
        ec = orf["ec_number"]
        ec3 = ec3_converter(ec)
        orf_id = orf["orf_id"]
        if ec_to_orf.get(ec) is None:
            ec_to_orf[ec] = set()
        if ec_to_orf.get(ec3) is None:
            ec_to_orf[ec3] = set()
        ec_to_orf[ec].add(orf_id)
        ec_to_orf[ec3].add(orf_id)
    return ec_to_orf


def annotate_enzyme_orfs_with_pathways(
    orfs: List[EnzymeData] = None,
    homology_score_threshold: float = 0.6,
    module_completeness_threshold: float = 0.7,
    annotate_kegg: bool = True,
) -> List[AnnotationOutput]:
    genome_ec_lookup = get_genome_ec_lookup(
        orfs=orfs, homology_score_threshold=homology_score_threshold
    )
    genome_ecs = set(genome_ec_lookup.keys())
    pathways = get_reference_pathways()
    out = []
    for pathway in tqdm(pathways, leave=False):
        req_enz = pathway["ec_numbers"]
        matches = genome_ecs.intersection(req_enz)
        score = len(matches) / len(req_enz)
        if score >= module_completeness_threshold:
            if not annotate_kegg and pathway["kegg_module_id"] is not None:
                continue
            missing_enzymes = req_enz - matches
            out.append(
                {
                    **{
                        k: v
                        for k, v in pathway.items()
                        if k
                        in [
                            "pathway_id",
                            "pathway_description",
                            "kegg_module_id",
                        ]
                    },
                    "completeness_score": round(score, 3),
                    "candidate_orfs": {
                        x: list(genome_ec_lookup[x]) for x in matches
                    },
                    "missing_criteria": list(missing_enzymes),
                    "matched_criteria": list(matches),
                }
            )
    return out


def convert_rule_to_ko_list(rule: str):
    repl = re.sub(r"[\(\)&\|]", r" ", rule)
    return repl.split()


class KOAnnotator:
    def __init__(
        self,
        allow_inferred_kegg_ecs: bool = False,
        ec_homology_cutoff: float = 0.6,
        ko_homology_cutoff: float = 0.6,
        module_completeness_threshold: float = 0.7,
    ):
        self.use_inf_ko_ecs = allow_inferred_kegg_ecs
        self.ec_homology_cutoff = ec_homology_cutoff
        self.ko_homology_cutoff = ko_homology_cutoff
        self.module_completeness_threshold = module_completeness_threshold
        self.load_ko_rules()
        self.load_ko_complement()
        self.load_ec_to_ko_mapper()

    def load_ko_complement(self):
        dat_fp = os.path.join(_dat_dir, "ko_data_summary.csv")
        dat = pd.read_csv(dat_fp)
        ko_df_ids = set(
            dat["ko_id"].tolist()
        )  # actually have protein sequences for these
        all_kos = set()
        for rule in self.rules:
            for sr in rule["rule"]:
                all_kos.update(convert_rule_to_ko_list(sr))
        # identify ko ids that have no sequences in the dataset and remove them.
        # Note that these are absent from the source KEGG database {'K18513', 'K21477', 'K16883', 'K23646'}
        not_repr = all_kos - ko_df_ids
        self.ko_complement = {
            x: False for x in dat["ko_id"].tolist() + list(not_repr)
        }
        self.ko_to_orf_mapper = {
            x: {} for x in dat["ko_id"].tolist() + list(not_repr)
        }

    def load_ko_rules(self):
        rule_fp = os.path.join(_dat_dir, "module_rules_converted.tsv")
        rule_df = pd.read_csv(rule_fp, sep="\t")
        rule_df["rule"] = rule_df["converted_rule"].map(
            lambda x: x.split("__")
        )
        self.rules = rule_df.to_dict("records")

    def load_ec_to_ko_mapper(self):
        if self.use_inf_ko_ecs:
            self.ec_to_ko = json.load(
                open(
                    os.path.join(
                        _dat_dir,
                        "ec_to_ko_lookup_w_inferred.json",
                    )
                )
            )
        else:
            self.ec_to_ko = json.load(
                open(
                    os.path.join(
                        _dat_dir,
                        "ec_to_ko_lookup_no_inferred.json",
                    )
                )
            )

    def _add_ko_to_mapper(
        self, ko_id: str, orf_id: Union[int, str], method: str
    ):
        if self.ko_to_orf_mapper[ko_id].get(orf_id) is None:
            self.ko_to_orf_mapper[ko_id][orf_id] = set()
        self.ko_to_orf_mapper[ko_id][orf_id].add(method)
        if self.ko_complement[ko_id] is False:
            self.ko_complement[ko_id] = True

    def assign_ko_complement_to_orfs(
        self, genome_orfs: List[Union[EnzymeData, EnzymeKOData]]
    ):
        for orf in genome_orfs:
            orf_id = orf["orf_id"]
            ec = orf.get("ec_number")
            ec_kos = self.ec_to_ko.get(ec)  # list of possible KOs.
            ec_score = orf.get("homology_score", 0)
            if ec_kos is not None and ec_score >= self.ec_homology_cutoff:
                for ec_ko in ec_kos:
                    self._add_ko_to_mapper(
                        ko_id=ec_ko, orf_id=orf_id, method="ec"
                    )
            ko = orf.get("ko_ortholog")
            ko_score = orf.get("ko_homology_score", 0)
            if ko is not None and ko_score >= self.ko_homology_cutoff:
                self._add_ko_to_mapper(ko_id=ko, orf_id=orf_id, method="ko")

    def evaluate_pathways(self):
        out = []
        for rule_dat in self.rules:
            mod_name = rule_dat["module_name"]
            mod_desc = rule_dat["pathway_name"]
            rule = rule_dat["rule"]
            kos = []
            true_srs = []
            false_srs = []
            for subrule in rule:
                # must constrain eval to avoid potentially dangerous behaviour
                # eval(expr, {}, {dict with ko to bool mapping})
                subr_bool = eval(subrule, {}, self.ko_complement)
                if subr_bool is True:
                    true_srs.append(subrule)
                else:
                    false_srs.append(subrule)
                kos.extend(convert_rule_to_ko_list(subrule))
            score = len(true_srs) / len(rule)
            if score >= self.module_completeness_threshold:
                out.append(
                    {
                        "pathway_id": mod_name,
                        "pathway_description": mod_desc,
                        "kegg_module_id": mod_name,
                        "completeness_score": round(score, 3),
                        "candidate_orfs": {
                            x: list(self.ko_to_orf_mapper[x]) for x in kos
                        },
                        "missing_criteria": false_srs,
                        "matched_criteria": true_srs,
                    }
                )
        return out

    def run_annotation(
        self, genome_orfs: List[Union[EnzymeData, EnzymeKOData]]
    ):
        self.assign_ko_complement_to_orfs(genome_orfs=genome_orfs)
        return self.evaluate_pathways()
