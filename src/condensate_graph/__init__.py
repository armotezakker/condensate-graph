"""Chain-level contact networks for protein-RNA condensate simulations.

All lengths in the public API are in nanometres (nm).
"""

from .box import BoxError, box_from_dimensions, resolve_box, validate_box, wrap_positions
from .contacts import (
    DEFAULT_CUTOFF_NM,
    chain_contacts,
    protein_protein_contacts,
    protein_rna_contacts,
)
from .system import ChainSet, ChainSystem, Frame, chains_from_atoms, load_system

__version__ = "0.1.0.dev0"

__all__ = [
    "BoxError",
    "box_from_dimensions",
    "resolve_box",
    "validate_box",
    "wrap_positions",
    "DEFAULT_CUTOFF_NM",
    "chain_contacts",
    "protein_protein_contacts",
    "protein_rna_contacts",
    "ChainSet",
    "ChainSystem",
    "Frame",
    "chains_from_atoms",
    "load_system",
]
