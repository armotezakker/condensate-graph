"""Chain-level contact networks for protein-RNA condensate simulations.

All lengths in the public API are in nanometres (nm).
"""

from .box import BoxError, box_from_dimensions, resolve_box, unwrap_chains, validate_box, wrap_positions
from .bridging import BridgingResult, bridging_metrics, run_bridging
from .graph import GRAPH_TYPES, SMALL_WORLD_WINDOW, GraphResult, graph_metrics, run_graph_metrics, small_world
from .contacts import (
    DEFAULT_CUTOFF_NM,
    ContactSettings,
    FrameContacts,
    iter_frame_contacts,
    chain_contacts,
    protein_protein_contacts,
    protein_rna_contacts,
)
from .system import (
    ChainIdentificationWarning,
    ChainSet,
    ChainSystem,
    Frame,
    build_system,
    chains_from_lengths,
    chains_from_topology,
    load_system,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "GRAPH_TYPES",
    "SMALL_WORLD_WINDOW",
    "small_world",
    "GraphResult",
    "graph_metrics",
    "run_graph_metrics",
    "BridgingResult",
    "bridging_metrics",
    "run_bridging",
    "BoxError",
    "box_from_dimensions",
    "resolve_box",
    "validate_box",
    "wrap_positions",
    "unwrap_chains",
    "ContactSettings",
    "FrameContacts",
    "iter_frame_contacts",
    "DEFAULT_CUTOFF_NM",
    "chain_contacts",
    "protein_protein_contacts",
    "protein_rna_contacts",
    "ChainIdentificationWarning",
    "ChainSet",
    "ChainSystem",
    "Frame",
    "build_system",
    "chains_from_lengths",
    "chains_from_topology",
    "load_system",
]
