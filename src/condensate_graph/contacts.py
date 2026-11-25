"""Chain-level contacts with minimum-image periodic distances.

Two chains are in contact in a frame if any bead of one lies within
``cutoff`` (nm) of any bead of the other, using the minimum-image distance
in an orthorhombic periodic box. All coordinates, box lengths and cutoffs
are in nanometres (nm).

The functions here work on plain arrays. :mod:`condensate_graph.system`
provides the coordinates (converted to nm) and the per-bead chain indices.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .box import validate_box, wrap_positions

DEFAULT_CUTOFF_NM = 1.5

__all__ = [
    "DEFAULT_CUTOFF_NM",
    "check_cutoff",
    "chain_contacts",
    "protein_rna_contacts",
    "protein_protein_contacts",
]


def check_cutoff(cutoff, box) -> float:
    """Validate a contact cutoff against the box.

    The minimum-image convention is only unambiguous when the cutoff is no
    larger than half of the shortest box edge.

    Parameters
    ----------
    cutoff : float
        Contact cutoff in nm.
    box : array_like, shape (3,)
        Box edge lengths in nm.

    Returns
    -------
    float
        The cutoff in nm.

    Raises
    ------
    ValueError
        If the cutoff is not positive or exceeds half the shortest box edge.
    """
    c = float(cutoff)
    L = validate_box(box)
    if not np.isfinite(c) or c <= 0.0:
        raise ValueError(f"cutoff must be a positive number of nm, got {cutoff}")
    if c > 0.5 * L.min():
        raise ValueError(
            f"cutoff {c} nm exceeds half the shortest box edge ({0.5 * L.min()} nm); "
            "minimum-image contacts are not defined"
        )
    return c


def _chain_index(chain_of_bead, n_chains, n_beads, name):
    idx = np.asarray(chain_of_bead)
    if idx.shape != (n_beads,):
        raise ValueError(f"{name} must have one chain index per bead ({n_beads}), got shape {idx.shape}")
    if not np.issubdtype(idx.dtype, np.integer):
        raise ValueError(f"{name} must contain integer chain indices")
    if n_beads and (idx.min() < 0 or idx.max() >= n_chains):
        raise ValueError(f"{name} values must lie in [0, {n_chains})")
    return idx.astype(np.intp, copy=False)


def chain_contacts(positions_a, chain_of_bead_a, n_chains_a,
                   positions_b, chain_of_bead_b, n_chains_b,
                   box, cutoff=DEFAULT_CUTOFF_NM) -> np.ndarray:
    """Contacts between the chains of two disjoint bead sets in one frame.

    Parameters
    ----------
    positions_a : array_like, shape (n_a, 3)
        Coordinates of set A in nm.
    chain_of_bead_a : array_like of int, shape (n_a,)
        Chain index (0 to ``n_chains_a - 1``) of each bead in set A.
    n_chains_a : int
        Number of chains in set A.
    positions_b : array_like, shape (n_b, 3)
        Coordinates of set B in nm.
    chain_of_bead_b : array_like of int, shape (n_b,)
        Chain index (0 to ``n_chains_b - 1``) of each bead in set B.
    n_chains_b : int
        Number of chains in set B.
    box : array_like, shape (3,)
        Orthorhombic box edge lengths in nm. Required; there is no
        non-periodic mode.
    cutoff : float, optional
        Contact cutoff in nm (default 1.5 nm). A pair of beads is in contact
        if its minimum-image distance is less than or equal to the cutoff.

    Returns
    -------
    numpy.ndarray of bool, shape (n_chains_a, n_chains_b)
        ``True`` where chain i of A and chain j of B are in contact.

    Raises
    ------
    BoxError
        If the box is missing or invalid.
    ValueError
        If shapes, chain indices or the cutoff are invalid.
    """
    L = validate_box(box)
    c = check_cutoff(cutoff, L)
    pa = wrap_positions(np.reshape(positions_a, (-1, 3)), L)
    pb = wrap_positions(np.reshape(positions_b, (-1, 3)), L)
    ca = _chain_index(chain_of_bead_a, n_chains_a, len(pa), "chain_of_bead_a")
    cb = _chain_index(chain_of_bead_b, n_chains_b, len(pb), "chain_of_bead_b")

    contact = np.zeros((int(n_chains_a), int(n_chains_b)), dtype=bool)
    if len(pa) == 0 or len(pb) == 0:
        return contact
    tree_a = cKDTree(pa, boxsize=L)
    tree_b = cKDTree(pb, boxsize=L)
    pairs = tree_a.sparse_distance_matrix(tree_b, max_distance=c, output_type="ndarray")
    if len(pairs):
        contact[ca[pairs["i"]], cb[pairs["j"]]] = True
    return contact


def protein_rna_contacts(protein_positions, protein_chain_of_bead, n_protein,
                         rna_positions, rna_chain_of_bead, n_rna,
                         box, cutoff=DEFAULT_CUTOFF_NM) -> np.ndarray:
    """Protein-RNA chain contacts in one frame.

    Parameters
    ----------
    protein_positions : array_like, shape (n_protein_beads, 3)
        Protein bead coordinates in nm.
    protein_chain_of_bead : array_like of int, shape (n_protein_beads,)
        Protein chain index of each protein bead.
    n_protein : int
        Number of protein chains.
    rna_positions : array_like, shape (n_rna_beads, 3)
        RNA bead coordinates in nm.
    rna_chain_of_bead : array_like of int, shape (n_rna_beads,)
        RNA chain index of each RNA bead.
    n_rna : int
        Number of RNA chains.
    box : array_like, shape (3,)
        Orthorhombic box edge lengths in nm.
    cutoff : float, optional
        Contact cutoff in nm (default 1.5 nm).

    Returns
    -------
    numpy.ndarray of bool, shape (n_rna, n_protein)
        ``True`` where RNA chain i contacts protein chain j.
    """
    return chain_contacts(rna_positions, rna_chain_of_bead, n_rna,
                          protein_positions, protein_chain_of_bead, n_protein,
                          box, cutoff)


def protein_protein_contacts(protein_positions, protein_chain_of_bead, n_protein,
                             box, cutoff=DEFAULT_CUTOFF_NM) -> np.ndarray:
    """Inter-chain protein-protein contacts in one frame.

    Bead pairs within the same chain are ignored, so the diagonal is always
    ``False``.

    Parameters
    ----------
    protein_positions : array_like, shape (n_protein_beads, 3)
        Protein bead coordinates in nm.
    protein_chain_of_bead : array_like of int, shape (n_protein_beads,)
        Protein chain index of each protein bead.
    n_protein : int
        Number of protein chains.
    box : array_like, shape (3,)
        Orthorhombic box edge lengths in nm.
    cutoff : float, optional
        Contact cutoff in nm (default 1.5 nm).

    Returns
    -------
    numpy.ndarray of bool, shape (n_protein, n_protein)
        Symmetric matrix, ``True`` where two distinct protein chains are in
        contact.
    """
    L = validate_box(box)
    c = check_cutoff(cutoff, L)
    p = wrap_positions(np.reshape(protein_positions, (-1, 3)), L)
    ch = _chain_index(protein_chain_of_bead, n_protein, len(p), "protein_chain_of_bead")

    contact = np.zeros((int(n_protein), int(n_protein)), dtype=bool)
    if len(p) < 2:
        return contact
    pairs = cKDTree(p, boxsize=L).query_pairs(r=c, output_type="ndarray")
    if len(pairs):
        ci, cj = ch[pairs[:, 0]], ch[pairs[:, 1]]
        inter = ci != cj
        contact[ci[inter], cj[inter]] = True
        contact[cj[inter], ci[inter]] = True
    return contact
