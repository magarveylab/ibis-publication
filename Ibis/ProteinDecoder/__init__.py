from Ibis.Utilities.Qdrant.classification import (
    ontology_neighborhood_classification,
    neighborhood_classification,
    KNNClassification,
)
from Ibis.ProteinDecoder.databases import (
    IbisEC,
    IbisKO,
    IbisGeneFamily,
    IbisGene,
    IbisMolecule,
)
from functools import partial

decode_ec = partial(
    KNNClassification,
    label_type="EC4Label",
    qdrant_db=IbisEC,
    classification_method=ontology_neighborhood_classification,
    top_n=5,
    dist_cutoff=25.36,
    apply_cutoff_before_homology=True,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)
decode_ko = partial(
    KNNClassification,
    qdrant_db=IbisKO,
    label_type="KeggOrthologLabel",
    classification_method=neighborhood_classification,
    top_n=5,
    dist_cutoff=26.06,
    apply_cutoff_before_homology=True,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)

decode_molecule = partial(
    KNNClassification,
    label_type="BioactivePeptideLabel",
    qdrant_db=IbisMolecule,
    classification_method=neighborhood_classification,
    top_n=5,
    dist_cutoff=31.69,
    apply_cutoff_before_homology=False,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)

decode_gene_family = partial(
    KNNClassification,
    label_type="GeneFamilyLabel",
    qdrant_db=IbisGeneFamily,
    classification_method=neighborhood_classification,
    partition_names=None,
    top_n=5,
    dist_cutoff=29.58,
    apply_cutoff_before_homology=True,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)

decode_gene = partial(
    KNNClassification,
    label_type="GeneLabel",
    qdrant_db=IbisGene,
    classification_method=neighborhood_classification,
    top_n=1,
    dist_cutoff=29.58,
    apply_cutoff_before_homology=True,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)
