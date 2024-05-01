from typing import List, Optional, TypedDict


class SubstrateDict(TypedDict):
    label: str
    rank: int


class Domain:

    def __init__(
        self,
        label: str,
        protein_id: int,
        start: int,
        stop: int,
        substrates: List[SubstrateDict] = [],
        functional: bool = True,
        subclass: Optional[str] = None,
    ):
        self.label = label
        self.protein_id = protein_id
        self.start = start
        self.stop = stop
        self.substrates = substrates
        self.functional = functional
        self.subclass = subclass
        self.insilico = False

    def __str__(self):
        if self.functional == True:
            if len(self.substrates) > 0:
                return f"{self.label}({self.substrates[0]['label']})"
            elif self.subclass != None:
                return f"{self.label}({self.subclass})"
            else:
                return self.label
        else:
            return f"{self.label}0"

    @property
    def domain_id(self):
        return f"{self.protein_id}_{self.start}_{self.stop}"


class CondensationDomain:
    # dummy domain annotation to call module boundaries
    def __init__(self):
        self.label = "C"
        self.protein_id = None
        self.start = None
        self.stop = None
        self.substrates = []
        self.functional = True
        self.subclass = None
        self.insilico = True

    def __str__(self):
        return "C(i)"

    @property
    def domain_id(self):
        return None


class KetosynthaseDomain:
    # dummy domain annotation to call module boundaries
    def __init__(self):
        self.label = "KS"
        self.protein_id = None
        self.start = None
        self.stop = None
        self.substrates = []
        self.functional = True
        self.subclass = None
        self.insilico = True

    def __str__(self):
        return "KS(i)"

    @property
    def domain_id(self):
        return None


class ThiolationDomain:
    # dummy domain annotation to determine incomplete modules
    def __init__(self):
        self.label = "T"
        self.protein_id = None
        self.start = None
        self.stop = None
        self.substrates = []
        self.functional = True
        self.subclass = None
        self.insilico = True

    def __str__(self):
        return "T(i)"

    @property
    def domain_id(self):
        return None
