"""Load a trajectory and identify protein and RNA chains.

Chains are identified in exactly one of three ways, chosen explicitly:

* ``by="segment"``: one chain per segment / chain ID in the topology;
* ``by="fragment"``: one chain per bonded fragment (needs bonds in the topology);
* ``chain_lengths={"protein": [...], "rna": [...]}``: explicit chain lengths in
  beads, in file order, which must add up exactly to each selection.

Nothing is guessed. MDAnalysis stores coordinates and cell lengths in
angstroms; everything this module returns is converted to nanometres (nm).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import MDAnalysis as mda

from .box import ANGSTROM_PER_NM, resolve_box, validate_box

CHAIN_DEFINITIONS = ("segment", "fragment")
SELECTION_NAMES = ("protein", "rna")

__all__ = [
    "CHAIN_DEFINITIONS",
    "ChainIdentificationWarning",
    "ChainSet",
    "Frame",
    "ChainSystem",
    "chains_from_topology",
    "chains_from_lengths",
    "build_system",
    "load_system",
]


class ChainIdentificationWarning(UserWarning):
    """Warning raised when the chain identification is probably wrong."""


@dataclass(frozen=True)
class ChainSet:
    """A group of atoms partitioned into chains.

    Attributes
    ----------
    atoms : MDAnalysis.core.groups.AtomGroup
        The atoms (beads) of this set, in file order.
    chain_of_atom : numpy.ndarray of int, shape (n_atoms,)
        Chain index (0 to ``n_chains - 1``) of each atom in ``atoms``.
    labels : tuple
        Identifier of each chain in chain-index order: the segid for
        ``by="segment"``, the fragment index for ``by="fragment"``, or the
        position in the list for explicit chain lengths.
    """

    atoms: object
    chain_of_atom: np.ndarray
    labels: tuple

    @property
    def n_chains(self) -> int:
        """Number of chains in the set."""
        return len(self.labels)

    def chain_sizes(self) -> np.ndarray:
        """Number of atoms (beads) in each chain, in chain-index order."""
        return np.bincount(self.chain_of_atom, minlength=self.n_chains)


@dataclass(frozen=True)
class Frame:
    """Coordinates of one trajectory frame, in nm.

    Attributes
    ----------
    index : int
        Frame index in the trajectory.
    time_ps : float
        Simulation time of the frame in picoseconds, as reported by MDAnalysis.
    box : numpy.ndarray, shape (3,)
        Orthorhombic box edge lengths in nm.
    protein_positions : numpy.ndarray, shape (n_protein_atoms, 3)
        Protein bead coordinates in nm (float64, not wrapped).
    rna_positions : numpy.ndarray, shape (n_rna_atoms, 3)
        RNA bead coordinates in nm (float64, not wrapped).
    """

    index: int
    time_ps: float
    box: np.ndarray
    protein_positions: np.ndarray
    rna_positions: np.ndarray


def _ordered_chain_index(keys):
    """Map per-atom keys to chain indices 0..n-1 in order of first appearance."""
    keys = np.asarray(keys)
    unique, first = np.unique(keys, return_index=True)
    ordered = unique[np.argsort(first)]
    index_of_key = {k: i for i, k in enumerate(ordered.tolist())}
    chain_of_atom = np.fromiter((index_of_key[k] for k in keys.tolist()), dtype=np.intp, count=len(keys))
    return chain_of_atom, ordered


def _n_bonds(universe) -> int:
    try:
        return len(universe.bonds)
    except (AttributeError, mda.exceptions.NoDataError):
        return 0


def chains_from_topology(atoms, by: str, name: str = "selection") -> ChainSet:
    """Partition an AtomGroup into chains from topology information.

    Parameters
    ----------
    atoms : MDAnalysis.core.groups.AtomGroup
        Atoms to partition. Must not be empty.
    by : {"segment", "fragment"}
        ``"segment"``: one chain per segment / chain ID. ``"fragment"``: one
        chain per bonded fragment; the topology must contain bonds.
    name : str, optional
        Name of the selection, used in messages.

    Returns
    -------
    ChainSet

    Raises
    ------
    ValueError
        If the group is empty, ``by`` is unknown, or ``by="fragment"`` is
        used on a topology without bonds.

    Warns
    -----
    ChainIdentificationWarning
        If ``by="segment"`` finds a single chain in a selection that spans
        more than one residue. Segment IDs alone cannot tell whether that is
        one long molecule or many molecules written under one chain ID, so
        the result is kept but flagged.
    """
    if by not in CHAIN_DEFINITIONS:
        raise ValueError(f"by must be one of {CHAIN_DEFINITIONS}, got {by!r}")
    if len(atoms) == 0:
        raise ValueError(f"{name} selection matched no atoms")
    if by == "segment":
        chain_of_atom, ordered = _ordered_chain_index(atoms.segindices)
        segid_of = {s.segindex: s.segid for s in atoms.segments}
        labels = tuple(segid_of[k] for k in ordered.tolist())
        if len(labels) == 1 and len(atoms.residues) > 1:
            warnings.warn(
                f"by='segment' found only 1 chain in the {name} selection "
                f"({len(atoms)} beads, {len(atoms.residues)} residues, segment {labels[0]!r}). "
                "If this selection contains more than one molecule, the chains are wrong. "
                "Use by='fragment' (needs bonds in the topology) or give explicit "
                "chain_lengths={'protein': [...], 'rna': [...]} instead.",
                ChainIdentificationWarning,
                stacklevel=3,
            )
        return ChainSet(atoms=atoms, chain_of_atom=chain_of_atom, labels=labels)
    if _n_bonds(atoms.universe) == 0:
        raise ValueError(
            "by='fragment' needs bonds in the topology, but it has none; use a topology "
            "with bonds (e.g. PSF, TPR, LAMMPS data, PDB with CONECT records) or give "
            "explicit chain_lengths"
        )
    chain_of_atom, ordered = _ordered_chain_index(atoms.fragindices)
    return ChainSet(atoms=atoms, chain_of_atom=chain_of_atom, labels=tuple(int(k) for k in ordered))


def chains_from_lengths(atoms, lengths, name: str = "selection") -> ChainSet:
    """Partition an AtomGroup into consecutive chains of given lengths.

    Parameters
    ----------
    atoms : MDAnalysis.core.groups.AtomGroup
        Atoms to partition, in file order.
    lengths : sequence of int
        Number of beads in each chain, in file order. All must be positive and
        they must add up exactly to ``len(atoms)``.
    name : str, optional
        Name of the selection, used in messages.

    Returns
    -------
    ChainSet

    Raises
    ------
    ValueError
        If any length is not a positive integer or the lengths do not add up
        to the number of atoms in the selection.
    """
    arr = np.asarray(lengths)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"chain_lengths[{name!r}] must be a non-empty list of bead counts")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"chain_lengths[{name!r}] must contain integers, got dtype {arr.dtype}")
    if np.any(arr <= 0):
        raise ValueError(f"chain_lengths[{name!r}] must all be positive, got {arr.tolist()}")
    total = int(arr.sum())
    if total != len(atoms):
        raise ValueError(
            f"chain_lengths[{name!r}] add up to {total} beads, but the {name} selection "
            f"has {len(atoms)} atoms"
        )
    chain_of_atom = np.repeat(np.arange(arr.size, dtype=np.intp), arr)
    return ChainSet(atoms=atoms, chain_of_atom=chain_of_atom, labels=tuple(range(arr.size)))


@dataclass
class ChainSystem:
    """A trajectory with its protein and RNA chains.

    Attributes
    ----------
    universe : MDAnalysis.Universe
        The loaded trajectory.
    protein : ChainSet
        Protein chains.
    rna : ChainSet
        RNA chains.
    box : numpy.ndarray of shape (3,) or None
        Explicit box edge lengths in nm, used for every frame when given.
        When None, the box is read from each frame and an error is raised
        if a frame has no valid unit cell.
    """

    universe: object
    protein: ChainSet
    rna: ChainSet
    box: np.ndarray | None = field(default=None)

    def frame_box(self) -> np.ndarray:
        """Box edge lengths in nm for the current frame.

        Raises
        ------
        condensate_graph.box.BoxError
            If no explicit box was given and the frame has no valid unit cell.
        """
        return resolve_box(self.universe.trajectory.ts.dimensions, self.box)

    def iter_frames(self, start=None, stop=None, step=None) -> Iterator[Frame]:
        """Iterate over frames, yielding coordinates in nm.

        Parameters
        ----------
        start, stop, step : int or None, optional
            Frame slice, with the same meaning as Python slicing over the
            trajectory (negative values count from the end).

        Yields
        ------
        Frame
            Box and protein/RNA coordinates in nm for each frame.
        """
        for ts in self.universe.trajectory[start:stop:step]:
            yield Frame(
                index=int(ts.frame),
                time_ps=float(ts.time),
                box=self.frame_box(),
                protein_positions=self.protein.atoms.positions.astype(np.float64) / ANGSTROM_PER_NM,
                rna_positions=self.rna.atoms.positions.astype(np.float64) / ANGSTROM_PER_NM,
            )


def build_system(universe, protein_selection: str, rna_selection: str, *,
                 by: str | None = None, chain_lengths: dict | None = None, box=None) -> ChainSystem:
    """Identify protein and RNA chains in an existing MDAnalysis Universe.

    Exactly one of ``by`` and ``chain_lengths`` must be given.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Topology and trajectory.
    protein_selection : str
        MDAnalysis selection string for the protein beads, e.g. ``"name CA"``.
    rna_selection : str
        MDAnalysis selection string for the RNA beads, e.g. ``"name P S"``.
    by : {"segment", "fragment"} or None, optional
        Identify chains from segment / chain IDs or from bonded fragments.
    chain_lengths : dict or None, optional
        ``{"protein": [...], "rna": [...]}``: chain lengths in beads, in file
        order, for each selection. Each list must add up exactly to the
        number of atoms in its selection.
    box : array_like of shape (3,) or None, optional
        Explicit orthorhombic box edge lengths in nm, for trajectories that
        store no unit cell. When None, every frame must carry a valid cell.

    Returns
    -------
    ChainSystem

    Raises
    ------
    ValueError
        If neither or both of ``by`` and ``chain_lengths`` are given, a
        selection is empty, the selections overlap, or the chain
        identification fails (see :func:`chains_from_topology` and
        :func:`chains_from_lengths`).
    condensate_graph.box.BoxError
        If ``box`` is given but invalid.
    """
    if (by is None) == (chain_lengths is None):
        raise ValueError("give exactly one of by='segment', by='fragment' or chain_lengths={...}")
    selections = {"protein": protein_selection, "rna": rna_selection}
    groups = {}
    for key, sel in selections.items():
        groups[key] = universe.select_atoms(sel)
        if len(groups[key]) == 0:
            raise ValueError(f"{key} selection {sel!r} matched no atoms")
    overlap = groups["protein"] & groups["rna"]
    if len(overlap):
        raise ValueError(f"protein and RNA selections share {len(overlap)} atoms")

    if chain_lengths is not None:
        if not isinstance(chain_lengths, dict) or set(chain_lengths) != set(SELECTION_NAMES):
            raise ValueError("chain_lengths must be a dict with exactly the keys 'protein' and 'rna'")
        chains = {k: chains_from_lengths(groups[k], chain_lengths[k], name=k) for k in SELECTION_NAMES}
    else:
        chains = {k: chains_from_topology(groups[k], by, name=k) for k in SELECTION_NAMES}
    return ChainSystem(
        universe=universe,
        protein=chains["protein"],
        rna=chains["rna"],
        box=None if box is None else validate_box(box),
    )


def load_system(topology, trajectory, protein_selection: str, rna_selection: str, *,
                by: str | None = None, chain_lengths: dict | None = None, box=None,
                **universe_kwargs) -> ChainSystem:
    """Load a trajectory from files and identify its protein and RNA chains.

    Parameters
    ----------
    topology : str or path-like
        Topology file readable by MDAnalysis (PDB, PSF, GRO, TPR, LAMMPS data, ...).
    trajectory : str, path-like or list
        Trajectory file(s) readable by MDAnalysis (DCD, XTC, LAMMPS dump, ...).
    protein_selection, rna_selection : str
        MDAnalysis selection strings for the protein and RNA beads.
    by, chain_lengths, box
        See :func:`build_system`. Lengths are in beads; the box is in nm.
    **universe_kwargs
        Passed to :class:`MDAnalysis.Universe` (e.g. ``format="LAMMPSDUMP"``).

    Returns
    -------
    ChainSystem
    """
    u = mda.Universe(topology, trajectory, **universe_kwargs)
    return build_system(u, protein_selection, rna_selection, by=by, chain_lengths=chain_lengths, box=box)
