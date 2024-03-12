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

protein_decoder = {}
protein_decoder["ec"] = partial(
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
protein_decoder["ko"] = partial(
    KNNClassification,
    qdrant_db=IbisKO,
    label_type="KeggOrthologLabel",
    classification_method=neighborhood_classification,
    top_n=5,
    dist_cutoff=None,
    sim_cutoff=0.03696,
    apply_cutoff_before_homology=True,
    apply_homology_cutoff=False,
    apply_cutoff_after_homology=False,
)
