import ast
from typing import List, Tuple, Union

import pandas as pd

from Ibis import curdir
from Ibis.ModulePredictor.Domain import (
    CondensationDomain,
    Domain,
    KetosynthaseDomain,
    ThiolationDomain,
)


def load_tag_lookup():
    # load tag lookup
    df = pd.read_csv(f"{curdir}/ModulePredictor/dat/module_tags.csv")
    return dict(zip(df.name, df.tag_id))


def load_characterizations():
    # load characterizations
    characterizations = pd.read_csv(
        f"{curdir}/ModulePredictor/dat/characterizations.csv"
    ).to_dict(orient="records")
    for c in characterizations:
        c["annotations"] = set(ast.literal_eval(c["annotations"]))
    return characterizations


characterizations = load_characterizations()


class Module:

    def __init__(self, module_idx: int, domains: List[Domain]):
        self.module_idx = module_idx
        self.domains = domains
        self.annotations = self.get_annotations()
        self.substrates = [s for d in self.domains for s in d.substrates]
        if len(self.substrates) == 0:
            if self.contains({"KS", "AT"}):
                self.substrates = [{"label": "Mal", "rank": 1}]
        start_pos = [d.start for d in self.domains if d.start != None]
        stop_pos = [d.stop for d in self.domains if d.stop != None]
        self.module_start = None if len(start_pos) == 0 else min(start_pos)
        self.module_stop = None if len(stop_pos) == 0 else max(stop_pos)
        protein_ids = [
            d.protein_id for d in self.domains if d.protein_id != None
        ]
        self.protein_id = None if len(protein_ids) == 0 else protein_ids[0]

    def get_annotations(self):
        # parse domains
        annotations = set()
        for d in self.domains:
            if d.functional == True:
                annotations.add(d.label)
        return annotations

    def __str__(self):
        return " ".join([str(d) for d in self.domains])

    def contains(self, annotations) -> bool:
        return True if len(self.annotations & annotations) > 0 else False

    @property
    def module_type(self) -> str:
        # determine module type based on annotations
        if self.contains({"KS"}):
            return "pks"
        elif self.contains({"A"}):
            return "nrps"
        elif self.contains({"KR", "DH", "ER"}):
            return "pks"
        elif self.module_idx == 1 and self.contains({"AT"}):
            return "starter"
        elif self.contains({"AT"}):
            return "pks"
        else:
            return "other"

    @property
    def module_tags(self) -> List[dict]:
        tags = []
        matching_rules = [
            r
            for r in characterizations
            if r["annotations"].issubset(self.annotations)
            and r["module_type"] == self.module_type
        ]
        if len(matching_rules) > 0:
            tag_formula = max(matching_rules, key=lambda x: x["priority"])[
                "formula"
            ]
            for s in self.substrates:
                tag_name = tag_formula.format(substrate=s["label"])
                if tag_name in ["Butyl-DH | TE", "Butyl-DH"]:
                    continue
                tags.append(
                    {
                        "tag": tag_name,
                        "rank": s["rank"],
                    }
                )
            # sometimes DH paired with invalid substrates (assume as Mal)
            if len(tags) == 0 and self.module_type == "pks":
                for s in [{"label": "Mal", "rank": 1}]:
                    tags.append(
                        {
                            "tag": tag_formula.format(substrate=s["label"]),
                            "rank": s["rank"],
                        }
                    )
        return tags

    @property
    def report(self) -> dict:
        domains = [
            f"{d.protein_id}_{d.start}_{d.stop}"
            for d in self.domains
            if d.protein_id != None
        ]
        return {
            "protein_id": self.protein_id,
            "module_idx": self.module_idx,
            "protein_start": self.module_start,
            "protein_stop": self.module_stop,
            "tags": self.module_tags,
            "domains": domains,
        }

    ####################################################################
    # Module Loading Functions
    ####################################################################

    @staticmethod
    def patch_module_boundaries(
        domains: List[Domain],
        boundaries: List[Tuple[str, str]],
        target_domain: Union[CondensationDomain, KetosynthaseDomain],
    ) -> List[Domain]:
        # capture all indexes of boundary end domains
        boundary_end_domains = set(d for b in boundaries for d in b)
        boundary_ends = []
        for idx, domain in enumerate(domains):
            if domain.label in boundary_end_domains:
                boundary_ends.append(idx)
        # check domains within boundary for target domain
        positions_to_add = []
        for idx, pos_first in enumerate(boundary_ends[:-1]):
            pos_last = boundary_ends[idx + 1]
            if (
                domains[pos_first].label,
                domains[pos_last].label,
            ) in boundaries:
                module = [d.label for d in domains[pos_first:pos_last]]
                if target_domain.label not in module:
                    positions_to_add.append(pos_last)
        # add target domain such as C to create complete nrps modules
        patched_domains = []
        for idx, domain in enumerate(domains):
            if idx in positions_to_add:
                patched_domains.append(target_domain)
            patched_domains.append(domain)
        return patched_domains

    @classmethod
    def load_from_domains(cls, domains: List[Domain]) -> List["Module"]:
        # filter domains
        domains_to_consider = [
            "T",
            "TE",
            "C",
            "A",
            "AT",
            "KS",
            "KR",
            "DH",
            "ER",
            "PS",
            "OMT",
            "NMT",
            "CMT",
        ]
        domains = [d for d in domains if d.label in domains_to_consider]
        modules = []
        # patch missing domains for module calling
        # sometimes in NRPS modules, we are missing condensation domains (C)
        target_domain = CondensationDomain()
        boundaries = [("A", "A"), ("KS", "A"), ("AT", "A")]
        domains = cls.patch_module_boundaries(
            domains, boundaries=boundaries, target_domain=target_domain
        )
        # sometimes in PKS modules, we are missing ketosynthase domains (KS)
        target_domain = KetosynthaseDomain()
        boundaries = [("KR", "AT"), ("PS", "AT"), ("T", "AT")]
        for b in boundaries:
            domains = cls.patch_module_boundaries(
                domains, boundaries=[b], target_domain=target_domain
            )
        # add in thiolation domains
        target_domain = ThiolationDomain()
        # consider each boundary seperately to accurately place T domain
        boundaries = [
            ("A", "C"),
            ("KR", "KS"),
            ("AT", "KS"),
            ("KS", "KS"),
            ("AT", "AT"),
        ]
        for b in boundaries:
            domains = cls.patch_module_boundaries(
                domains, boundaries=[b], target_domain=target_domain
            )
        # cache starting index
        # define boundary as it must start with the domain
        # and it goes up to the end domain
        # note the end domain is the start of another module
        boundaries = [
            ("C", "KS"),
            ("C", "AT"),
            ("C", "C"),
            ("KS", "KS"),
            ("AT", "AT"),
            ("KS", "C"),
            ("AT", "C"),
            ("AT", "KS"),
            ("A", "KS"),
            ("A", "C"),
            ("A", "AT"),
            ("A", "A"),
        ]
        boundary_end_domains = ["C", "KS", "AT", "A"]
        # capture all indexes of boundary end domains
        domain_str = []
        boundary_ends = []
        for idx, domain in enumerate(domains):
            domain_str.append(domain.label)
            if domain.label in boundary_end_domains and domain.functional:
                boundary_ends.append(idx)
        # capture start and end of valid modules
        module_boundaries = [0]
        current_idx = 0
        if len(boundary_ends) == 1 and boundary_ends[0] != 0:
            module_boundaries.append(boundary_ends[0])
        for idx, pos_first in enumerate(boundary_ends[:-1]):
            if idx < current_idx:
                continue
            for pos_last in boundary_ends[idx + 1 :]:
                domain_first = domains[pos_first].label
                domain_last = domains[pos_last].label
                if (domain_first, domain_last) in boundaries:
                    module_boundaries.append(pos_last)
                    current_idx = boundary_ends.index(pos_last)
                    break
        # if no module boundaries detected, then current algo reads only 1 module
        # this is incorrect as there could be a partial module
        # a partial module is determined if KR, DH, ER comes before KS
        # maybe apply rules for nrps systems too (future)
        if len(module_boundaries) == 1:
            # look for KS index
            if "KS" in domain_str:
                break_point = domain_str.index("KS")
                other_domains = ["KR", "DH", "ER"]
                for od in other_domains:
                    if od in domain_str and domain_str.index(od) < break_point:
                        module_boundaries.append(break_point)
                        break
        module_boundaries.append(len(domains))
        # split list of domains to modules
        module_idx = 1
        for idx, value in enumerate(module_boundaries[:-1]):
            module_domains = domains[value : module_boundaries[idx + 1]]
            if len(module_domains) > 0:
                module = cls(module_idx, module_domains)
                if module.module_type != "other" and module.protein_id != None:
                    modules.append(module)
                    module_idx += 1
        return modules
