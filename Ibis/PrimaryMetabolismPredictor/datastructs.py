from typing import Dict, List, Optional, Set, TypedDict, Union

orf_id_type = Union[int, str]


# To retrieve enzyme data from Neo4j.
class OrfData(TypedDict):
    orf_id: orf_id_type
    protein_id: int


# input to pathway annotation.
class EnzymeData(TypedDict):
    orf_id: orf_id_type
    ec_number: str
    homology_score: float


class EnzymeKOData(TypedDict):
    orf_id: orf_id_type
    ec_number: str
    homology_score: float
    ko_ortholog: str
    ko_homology_score: float
    ko_similarity_score: Optional[float]


class AnnotationOutput(TypedDict):
    neo4j_id: int
    pathway_description: str
    completeness_score: int
    candidate_orfs: Dict[str, Set[orf_id_type]]
    kegg_module_id: str
    missing_criteria: List[str]
    matched_criteria: List[str]


################
# upload types #
################


class PredictedPathwayDict(TypedDict):
    prediction_id: str  # {genome_enum_id}_{pathway_id}
    module_completeness_score: float
    detected_labels: List[str]
    missing_labels: List[str]
    orf_ids: List[str]
