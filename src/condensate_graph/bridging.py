"""RNA bridging metrics from chain-level protein-RNA contacts.

All lengths (coordinates, box, cutoff, region bounds) are in nanometres (nm).
Contacts are computed with minimum-image periodic distances by
:mod:`condensate_graph.contacts`; the same cutoff is used for protein-RNA and
protein-protein contacts.

Per frame and per RNA chain:

* valency: the number of distinct protein chains the RNA chain contacts;
* basic bridge: valency >= 2;
* productive bridge: the RNA chain contacts at least one pair of protein
  chains that are not in direct contact with each other in that frame;
* persistent bridge (``k`` frames): the RNA chain contacted the same pair of
  protein chains in each of the last ``k`` consecutive analysed frames.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .box import validate_box
from .contacts import DEFAULT_CUTOFF_NM, protein_protein_contacts, protein_rna_contacts

MAX_VALENCY_BIN = 4
VALENCY_LABELS = ("0", "1", "2", "3", ">=4")

__all__ = [
    "VALENCY_LABELS",
    "BridgingResult",
    "valency",
    "valency_distribution",
    "basic_bridge_flags",
    "productive_bridge_flags",
    "persistent_bridge_flags",
    "chain_center_z",
    "region_mask",
    "bridging_metrics",
    "run_bridging",
]


# --------------------------------------------------------------------------- single-frame pieces

def valency(protein_rna_contact) -> np.ndarray:
    """Number of distinct protein chains each RNA chain contacts.

    Parameters
    ----------
    protein_rna_contact : array_like of bool, shape (n_rna, n_protein)
        Chain-level contact matrix for one frame (see
        :func:`condensate_graph.contacts.protein_rna_contacts`).

    Returns
    -------
    numpy.ndarray of int, shape (n_rna,)
    """
    return np.asarray(protein_rna_contact, dtype=bool).sum(axis=1)


def valency_distribution(valencies, counted=None) -> np.ndarray:
    """Distribution P(n) of RNA valency for n = 0, 1, 2, 3 and >= 4.

    Parameters
    ----------
    valencies : array_like of int, shape (n_rna,)
        Valency of each RNA chain (see :func:`valency`).
    counted : array_like of bool, shape (n_rna,), optional
        Which RNA chains to include. Default: all.

    Returns
    -------
    numpy.ndarray of float, shape (5,)
        Fractions of the counted RNA chains with valency 0, 1, 2, 3 and >= 4;
        sums to 1, or all NaN if no RNA chain is counted.
    """
    v = np.asarray(valencies)
    if counted is not None:
        v = v[np.asarray(counted, dtype=bool)]
    if v.size == 0:
        return np.full(MAX_VALENCY_BIN + 1, np.nan)
    counts = np.bincount(np.minimum(v, MAX_VALENCY_BIN), minlength=MAX_VALENCY_BIN + 1)
    return counts / v.size


def basic_bridge_flags(protein_rna_contact) -> np.ndarray:
    """RNA chains that contact two or more distinct protein chains.

    Parameters
    ----------
    protein_rna_contact : array_like of bool, shape (n_rna, n_protein)

    Returns
    -------
    numpy.ndarray of bool, shape (n_rna,)
    """
    return valency(protein_rna_contact) >= 2


def productive_bridge_flags(protein_rna_contact, protein_protein_contact) -> np.ndarray:
    """RNA chains that bridge at least one pair of non-contacting proteins.

    An RNA chain is a productive bridge in a frame if it contacts at least one
    pair of protein chains that are not in direct contact with each other in
    that frame (same cutoff for both kinds of contact).

    Parameters
    ----------
    protein_rna_contact : array_like of bool, shape (n_rna, n_protein)
        Chain-level protein-RNA contacts for one frame.
    protein_protein_contact : array_like of bool, shape (n_protein, n_protein)
        Symmetric chain-level protein-protein contacts for the same frame,
        with a False diagonal.

    Returns
    -------
    numpy.ndarray of bool, shape (n_rna,)
    """
    pr = np.asarray(protein_rna_contact, dtype=np.int64)
    pp = np.asarray(protein_protein_contact, dtype=np.int64)
    k = pr.sum(axis=1)
    pairs_contacted = k * (k - 1) // 2
    # for each RNA: number of protein-protein contacts among its partners
    pairs_in_contact = ((pr @ pp) * pr).sum(axis=1) // 2
    return pairs_contacted > pairs_in_contact


def persistent_bridge_flags(contact_window) -> np.ndarray:
    """RNA chains that contacted the same pair of proteins in every frame of a window.

    Parameters
    ----------
    contact_window : array_like of bool, shape (k, n_rna, n_protein)
        Protein-RNA contact matrices of ``k`` consecutive frames, oldest first.

    Returns
    -------
    numpy.ndarray of bool, shape (n_rna,)
        True if the RNA chain contacted at least two protein chains in all
        ``k`` frames (that is, at least one common pair).
    """
    window = np.asarray(contact_window, dtype=bool)
    return window.all(axis=0).sum(axis=1) >= 2


def chain_center_z(positions, chain_of_bead, n_chains, box_z, masses=None) -> np.ndarray:
    """Periodic-aware z coordinate of each chain's centre, in nm.

    Each chain is made whole along z by unwrapping every bead relative to the
    previous bead of the same chain (in file order), then the (mass-weighted)
    mean z is wrapped back into ``[0, box_z)``. This assumes consecutive beads
    of a chain are closer than ``box_z / 2`` along z, which holds for bonded
    beads.

    Parameters
    ----------
    positions : array_like, shape (n_beads, 3)
        Bead coordinates in nm.
    chain_of_bead : array_like of int, shape (n_beads,)
        Chain index of each bead.
    n_chains : int
        Number of chains.
    box_z : float
        Box length along z in nm.
    masses : array_like, shape (n_beads,), optional
        Bead masses (any unit) for a centre of mass. Default: equal weights
        (centre of geometry).

    Returns
    -------
    numpy.ndarray, shape (n_chains,)
        Centre z of each chain in nm, in ``[0, box_z)``; NaN for a chain
        with no beads.
    """
    z = np.asarray(positions, dtype=np.float64)[:, 2]
    ch = np.asarray(chain_of_bead, dtype=np.intp)
    w = np.ones_like(z) if masses is None else np.asarray(masses, dtype=np.float64)
    order = np.argsort(ch, kind="stable")          # group beads by chain, keep file order inside
    z, ch, w = z[order], ch[order], w[order]
    dz = np.diff(z)
    dz -= box_z * np.round(dz / box_z)
    same = ch[1:] == ch[:-1]
    step = np.concatenate(([0.0], np.where(same, dz, 0.0)))
    cum = np.cumsum(step)
    starts = np.concatenate(([True], ~same))
    group_start_cum = np.maximum.accumulate(np.where(starts, np.arange(len(z)), 0))
    first_z = z[group_start_cum]
    unwrapped = first_z + cum - cum[group_start_cum]
    wsum = np.bincount(ch, weights=w, minlength=n_chains)
    zsum = np.bincount(ch, weights=w * unwrapped, minlength=n_chains)
    with np.errstate(invalid="ignore", divide="ignore"):
        centre = zsum / wsum
    return np.mod(centre, box_z)


def region_mask(center_z, region, box_z) -> np.ndarray:
    """Chains whose centre z lies inside a slab region.

    Parameters
    ----------
    center_z : array_like, shape (n_chains,)
        Chain centre z in nm, in ``[0, box_z)`` (see :func:`chain_center_z`).
    region : tuple of float
        ``(zmin, zmax)`` in nm, with ``0 <= zmin < zmax <= box_z``. Both
        bounds are inclusive.
    box_z : float
        Box length along z in nm.

    Returns
    -------
    numpy.ndarray of bool, shape (n_chains,)

    Raises
    ------
    ValueError
        If the region is not a valid interval inside the box.
    """
    zmin, zmax = (float(x) for x in region)
    if not (0.0 <= zmin < zmax <= box_z):
        raise ValueError(f"region must satisfy 0 <= zmin < zmax <= box_z ({box_z} nm), got ({zmin}, {zmax})")
    c = np.asarray(center_z, dtype=np.float64)
    return (c >= zmin) & (c <= zmax)


# --------------------------------------------------------------------------- results

@dataclass(frozen=True)
class BridgingResult:
    """Per-frame bridging metrics.

    All per-frame fractions are over the RNA chains counted in that frame
    (all RNA chains, or those inside ``region``). A frame with no counted RNA
    chain gives NaN.

    Attributes
    ----------
    frame_index : numpy.ndarray of int, shape (n_frames,)
        Trajectory frame index of each analysed frame.
    counted : numpy.ndarray of bool, shape (n_frames, n_rna)
        RNA chains counted in each frame.
    n_rna_counted : numpy.ndarray of int, shape (n_frames,)
    valency_distribution : numpy.ndarray, shape (n_frames, 5)
        P(n) for n = 0, 1, 2, 3, >= 4 in each frame.
    mean_valency : numpy.ndarray, shape (n_frames,)
        B, the mean number of distinct protein chains per counted RNA chain.
    f_bridge_basic : numpy.ndarray, shape (n_frames,)
        Fraction of counted RNA chains with valency >= 2.
    f_bridge_productive : numpy.ndarray, shape (n_frames,)
        Fraction of counted RNA chains that are productive bridges.
    f_bridge_persistent : dict[int, numpy.ndarray]
        For each ``k``, the fraction of counted RNA chains that are
        persistent bridges over ``k`` frames. NaN for the first ``k - 1``
        analysed frames, which have too little history.
    persistent_flags : dict[int, numpy.ndarray]
        For each ``k``, per-frame per-RNA persistent-bridge flags,
        shape (n_frames, n_rna); False in the first ``k - 1`` frames.
    cutoff : float
        Contact cutoff in nm.
    region : tuple or None
        ``(zmin, zmax)`` in nm, or None.
    """

    frame_index: np.ndarray
    counted: np.ndarray
    n_rna_counted: np.ndarray
    valency_distribution: np.ndarray
    mean_valency: np.ndarray
    f_bridge_basic: np.ndarray
    f_bridge_productive: np.ndarray
    f_bridge_persistent: dict = field(default_factory=dict)
    persistent_flags: dict = field(default_factory=dict)
    cutoff: float = DEFAULT_CUTOFF_NM
    region: tuple | None = None

    @property
    def n_frames(self) -> int:
        """Number of analysed frames."""
        return len(self.frame_index)

    def means(self) -> dict:
        """Time averages over frames (NaN frames ignored).

        Returns
        -------
        dict
            ``B``, ``P_n`` (array of 5), ``f_bridge_basic``,
            ``f_bridge_productive`` and ``f_bridge_persistent`` (dict by k).
        """
        def avg(x):
            x = np.asarray(x, dtype=np.float64)
            return float(np.nanmean(x)) if np.any(np.isfinite(x)) else float("nan")

        return {
            "B": avg(self.mean_valency),
            "P_n": np.array([avg(col) for col in self.valency_distribution.T]),
            "f_bridge_basic": avg(self.f_bridge_basic),
            "f_bridge_productive": avg(self.f_bridge_productive),
            "f_bridge_persistent": {k: avg(v) for k, v in self.f_bridge_persistent.items()},
        }


# --------------------------------------------------------------------------- drivers

def _fraction(flags, counted):
    n = counted.sum()
    return float(flags[counted].sum() / n) if n else float("nan")


def _as_min_frames(min_frames):
    ks = (min_frames,) if np.isscalar(min_frames) else tuple(min_frames)
    ks = tuple(sorted({int(k) for k in ks}))
    if not ks or ks[0] < 1:
        raise ValueError(f"min_frames must be integers >= 1, got {min_frames}")
    return ks


def bridging_metrics(frames, protein_chain_of_bead, n_protein, rna_chain_of_bead, n_rna, *,
                     cutoff=DEFAULT_CUTOFF_NM, region=None, min_frames=(1,), rna_masses=None) -> BridgingResult:
    """Compute bridging metrics over a sequence of frames.

    Parameters
    ----------
    frames : iterable
        Objects with attributes ``box`` (shape (3,), nm), ``protein_positions``
        and ``rna_positions`` (shape (n, 3), nm) and optionally ``index``,
        such as :class:`condensate_graph.system.Frame`.
    protein_chain_of_bead : array_like of int, shape (n_protein_beads,)
        Protein chain index of each protein bead.
    n_protein : int
        Number of protein chains.
    rna_chain_of_bead : array_like of int, shape (n_rna_beads,)
        RNA chain index of each RNA bead.
    n_rna : int
        Number of RNA chains.
    cutoff : float, optional
        Contact cutoff in nm (default 1.5 nm), used for both protein-RNA
        and protein-protein contacts.
    region : tuple of float or None, optional
        ``(zmin, zmax)`` in nm. If given, only RNA chains whose periodic-aware
        centre z (see :func:`chain_center_z`) lies in this range are counted
        in each frame. Contacts are always computed with all chains. No slab
        detection is done.
    min_frames : int or sequence of int, optional
        Window lengths ``k`` for the persistent bridge (default ``(1,)``;
        ``k = 1`` equals the basic bridge). Persistence assumes the frames
        are consecutive and equally spaced in time, so a window of ``k``
        frames spans ``k`` times the spacing of the analysed frames (the
        trajectory saving interval multiplied by any step used when
        selecting frames). Do not pass frames with gaps.
    rna_masses : array_like, shape (n_rna_beads,), optional
        Bead masses for the RNA centre of mass in the region filter.
        Default: equal weights.

    Returns
    -------
    BridgingResult
    """
    ks = _as_min_frames(min_frames)
    k_max = ks[-1]
    history = deque(maxlen=k_max)
    rows = {key: [] for key in ("index", "counted", "pn", "B", "basic", "productive")}
    persistent = {k: [] for k in ks}
    flags_by_k = {k: [] for k in ks}

    for t, f in enumerate(frames):
        box = validate_box(f.box)
        pr = protein_rna_contacts(f.protein_positions, protein_chain_of_bead, n_protein,
                                  f.rna_positions, rna_chain_of_bead, n_rna, box, cutoff)
        pp = protein_protein_contacts(f.protein_positions, protein_chain_of_bead, n_protein, box, cutoff)
        if region is None:
            counted = np.ones(n_rna, dtype=bool)
        else:
            cz = chain_center_z(f.rna_positions, rna_chain_of_bead, n_rna, box[2], rna_masses)
            counted = region_mask(cz, region, box[2])
        val = valency(pr)
        history.append(pr)

        rows["index"].append(int(getattr(f, "index", t)))
        rows["counted"].append(counted)
        rows["pn"].append(valency_distribution(val, counted))
        rows["B"].append(float(val[counted].mean()) if counted.any() else float("nan"))
        rows["basic"].append(_fraction(val >= 2, counted))
        rows["productive"].append(_fraction(productive_bridge_flags(pr, pp), counted))
        for k in ks:
            if len(history) >= k:
                fl = persistent_bridge_flags(list(history)[-k:])
                persistent[k].append(_fraction(fl, counted))
            else:
                fl = np.zeros(n_rna, dtype=bool)
                persistent[k].append(float("nan"))
            flags_by_k[k].append(fl)

    n_frames = len(rows["index"])
    return BridgingResult(
        frame_index=np.array(rows["index"], dtype=np.int64),
        counted=np.array(rows["counted"], dtype=bool).reshape(n_frames, n_rna),
        n_rna_counted=np.array([c.sum() for c in rows["counted"]], dtype=np.int64),
        valency_distribution=np.array(rows["pn"], dtype=np.float64).reshape(n_frames, MAX_VALENCY_BIN + 1),
        mean_valency=np.array(rows["B"], dtype=np.float64),
        f_bridge_basic=np.array(rows["basic"], dtype=np.float64),
        f_bridge_productive=np.array(rows["productive"], dtype=np.float64),
        f_bridge_persistent={k: np.array(v, dtype=np.float64) for k, v in persistent.items()},
        persistent_flags={k: np.array(v, dtype=bool).reshape(n_frames, n_rna) for k, v in flags_by_k.items()},
        cutoff=float(cutoff),
        region=None if region is None else (float(region[0]), float(region[1])),
    )


def run_bridging(system, start=None, stop=None, step=None, **kwargs) -> BridgingResult:
    """Compute bridging metrics for a :class:`condensate_graph.system.ChainSystem`.

    Parameters
    ----------
    system : condensate_graph.system.ChainSystem
        Trajectory with identified protein and RNA chains.
    start, stop, step : int or None, optional
        Frame slice over the trajectory. With ``step > 1`` the persistence
        window spans ``k * step`` saved frames.
    **kwargs
        ``cutoff`` (nm), ``region`` (nm), ``min_frames`` and ``rna_masses``,
        passed to :func:`bridging_metrics`.

    Returns
    -------
    BridgingResult
    """
    return bridging_metrics(
        system.iter_frames(start, stop, step),
        system.protein.chain_of_atom, system.protein.n_chains,
        system.rna.chain_of_atom, system.rna.n_chains,
        **kwargs,
    )
