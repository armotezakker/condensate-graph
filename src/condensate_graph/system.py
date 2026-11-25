"""Load a trajectory and identify protein and RNA chains.

Chains are taken from the topology (MDAnalysis segments, bonded fragments or
molecule numbers), never by slicing the atom list into equal blocks, so
chains of different lengths and mixed systems are handled correctly.

MDAnalysis stores coordinates and cell lengths in angstroms; everything this
module returns is converted to nanometres (nm).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import MDAnalysis as mda

from .box import ANGSTROM_PER_NM, resolve_box, validate_box

CHAIN_DEFINITIONS = ("segment", "fragment", "molnum")

__all__ = ["CHAIN_DEFINITIONS", "ChainSet", "Frame", "ChainSystem", "chains_from_atoms", "load_system"]


@dataclass(frozen=True)
class ChainSet:
    """A group of atoms partitioned into chains.

    Attributes
    ----------
    atoms : MDAnalysis.core.groups.AtomGroup
        The atoms (beads) of this set, in topology order.
    chain_of_atom : numpy.ndarray of int, shape (n_atoms,)
        Chain index (0 to ``n_chains - 1``) of each atom in ``atoms``.
    labels : tuple
        Topology identifier of each chain (segid, fragment index or molnum),
        in chain-index order.
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


def chains_from_atoms(atoms, by: str = "segment") -> ChainSet:
    """Partition an AtomGroup into chains using topology information.

    Parameters
    ----------
    atoms : MDAnalysis.core.groups.AtomGroup
        Atoms to partition. Must not be empty.
    by : {"segment", "fragment", "molnum"}, optional
        How chains are defined: one chain per MDAnalysis segment (segid),
        per bonded fragment (requires bonds in the topology), or per
        molecule number (``molnums``, available for example from GROMACS
        TPR files). Default ``"segment"``.

    Returns
    -------
    ChainSet
        The atoms with a chain index per atom.

    Raises
    ------
    ValueError
        If the group is empty, ``by`` is unknown, or the topology lacks the
        information needed for the chosen definition.
    """
    if by not in CHAIN_DEFINITIONS:
        raise ValueError(f"by must be one of {CHAIN_DEFINITIONS}, got {by!r}")
    if len(atoms) == 0:
        raise ValueError("selection matched no atoms")
    if by == "segment":
        keys = atoms.segindices
        label_of = {s.segindex: s.segid for s in atoms.segments}
    elif by == "fragment":
        if not hasattr(atoms, "fragindices"):
            raise ValueError("chain definition 'fragment' needs bonds in the topology")
        try:
            keys = atoms.fragindices
        except mda.exceptions.NoDataError as exc:
            raise ValueError("chain definition 'fragment' needs bonds in the topology") from exc
        label_of = None
    else:
        if not hasattr(atoms, "molnums"):
            raise ValueError("chain definition 'molnum' needs molnums in the topology (e.g. a GROMACS TPR)")
        keys = atoms.molnums
        label_of = None
    unique, first = np.unique(keys, return_index=True)
    ordered = unique[np.argsort(first)]          # chains in order of first appearance
    index_of_key = {k: i for i, k in enumerate(ordered.tolist())}
    chain_of_atom = np.fromiter((index_of_key[k] for k in keys.tolist()), dtype=np.intp, count=len(keys))
    labels = tuple(label_of[k] for k in ordered.tolist()) if label_of else tuple(int(k) for k in ordered)
    return ChainSet(atoms=atoms, chain_of_atom=chain_of_atom, labels=labels)


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


def load_system(topology, trajectory, protein_selection: str, rna_selection: str,
                by: str = "segment", box=None, **universe_kwargs) -> ChainSystem:
    """Load a trajectory and identify its protein and RNA chains.

    Parameters
    ----------
    topology : str or path-like
        Topology file readable by MDAnalysis (PDB, PSF, GRO, TPR, LAMMPS data, ...).
    trajectory : str, path-like or list
        Trajectory file(s) readable by MDAnalysis (DCD, XTC, LAMMPS dump, ...).
    protein_selection : str
        MDAnalysis selection string for the protein beads, e.g. ``"name CA"``.
    rna_selection : str
        MDAnalysis selection string for the RNA beads, e.g. ``"name P S"``.
    by : {"segment", "fragment", "molnum"}, optional
        Chain definition passed to :func:`chains_from_atoms`. Default ``"segment"``.
    box : array_like of shape (3,) or None, optional
        Explicit orthorhombic box edge lengths in nm, for trajectories that
        store no unit cell. When None, every frame must carry a valid cell.
    **universe_kwargs
        Passed to :class:`MDAnalysis.Universe` (e.g. ``format="LAMMPSDUMP"``).

    Returns
    -------
    ChainSystem

    Raises
    ------
    ValueError
        If a selection is empty, the two selections overlap, or the chain
        definition cannot be applied.
    condensate_graph.box.BoxError
        If ``box`` is given but invalid.
    """
    u = mda.Universe(topology, trajectory, **universe_kwargs)
    protein_atoms = u.select_atoms(protein_selection)
    rna_atoms = u.select_atoms(rna_selection)
    if len(protein_atoms) == 0:
        raise ValueError(f"protein selection {protein_selection!r} matched no atoms")
    if len(rna_atoms) == 0:
        raise ValueError(f"RNA selection {rna_selection!r} matched no atoms")
    overlap = protein_atoms & rna_atoms
    if len(overlap):
        raise ValueError(f"protein and RNA selections share {len(overlap)} atoms")
    return ChainSystem(
        universe=u,
        protein=chains_from_atoms(protein_atoms, by=by),
        rna=chains_from_atoms(rna_atoms, by=by),
        box=None if box is None else validate_box(box),
    )
