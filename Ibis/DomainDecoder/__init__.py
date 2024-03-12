from Ibis.Utilities.Qdrant.classification import (
    ontology_neighborhood_classification,
    neighborhood_classification,
    KNNClassification,
)
from Ibis.ProteinDecoder.databases import (
    IbisAdenylation,
    IbisAcyltransferase,
    IbisKetosynthase,
    IbisKetoreductase,
    IbisDehydratase,
    IbisEnoylreductase,
    IbisThiolation,
)
from functools import partial

decode_adenylation = partial(
    KNNClassification,
    label_type="SubstrateLabel",
    qdrant_db=IbisAdenylation,
    classification_method=ontology_neighborhood_classification,
    top_n=3,
    apply_cutoff_before_homology=False,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)

decode_acyltransferase = partial(
    KNNClassification,
    label_type="SubstrateLabel",
    qdrant_db=IbisAcyltransferase,
    classification_method=ontology_neighborhood_classification,
    top_n=5,
    apply_cutoff_before_homology=False,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)

decode_ketosynthase = partial(
    KNNClassification,
    label_type="DomainFunctionalLabel",
    qdrant_db=IbisKetosynthase,
    classification_method=neighborhood_classification,
    top_n=5,
    apply_cutoff_before_homology=False,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)

decode_ketoreductase = partial(
    KNNClassification,
    label_type="DomainFunctionalLabel",
    qdrant_db=IbisKetoreductase,
    classification_method=neighborhood_classification,
    top_n=5,
    apply_cutoff_before_homology=False,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)

decode_dehydratase = partial(
    KNNClassification,
    label_type="DomainFunctionalLabel",
    qdrant_db=IbisDehydratase,
    classification_method=neighborhood_classification,
    top_n=5,
    apply_cutoff_before_homology=False,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)

decode_enoylreductase = partial(
    KNNClassification,
    label_type="DomainFunctionalLabel",
    qdrant_db=IbisDehydratase,
    classification_method=neighborhood_classification,
    top_n=5,
    apply_cutoff_before_homology=False,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)
