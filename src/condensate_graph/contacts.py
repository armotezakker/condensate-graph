"""Chain-level contacts with minimum-image periodic distances.

Two chains are in contact in a frame if any bead of one lies within
``cutoff`` (nm) of any bead of the other, using the minimum-image distance
in an orthorhombic periodic box (distance <= cutoff counts). All coordinates,
box lengths and cutoffs are in nanometres (nm).

Optional hysteresis (``cutoff_off > cutoff``): a chain-pair contact switches
on when the minimum bead-bead distance is within ``cutoff`` and stays on in
later frames until that distance exceeds ``cutoff_off``. This makes contacts
depend on earlier frames, so it is applied by :func:`iter_frame_contacts`,
which walks through frames in order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .box import unwrap_chains, validate_box, wrap_positions

DEFAULT_CUTOFF_NM = 1.5

__all__ = [
    "DEFAULT_CUTOFF_NM",
    "ContactSettings",
    "FrameContacts",
    "check_cutoff",
    "chain_contacts",
    "protein_rna_contacts",
    "protein_protein_contacts",
    "update_hysteresis",
    "iter_frame_contacts",
]


# --------------------------------------------------------------------------- validation

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


@dataclass(frozen=True)
class ContactSettings:
    """Cutoffs for protein-RNA and protein-protein contacts, in nm.

    Attributes
    ----------
    cutoff : float
        Protein-RNA contact cutoff in nm.
    cutoff_off : float or None
        Protein-RNA hysteresis release distance in nm (> ``cutoff``), or None
        for a plain single cutoff.
    protein_protein_cutoff : float
        Protein-protein contact cutoff in nm.
    protein_protein_cutoff_off : float or None
        Protein-protein hysteresis release distance in nm, or None.
    """

    cutoff: float
    cutoff_off: float | None
    protein_protein_cutoff: float
    protein_protein_cutoff_off: float | None

    @classmethod
    def create(cls, cutoff=DEFAULT_CUTOFF_NM, cutoff_off=None,
               protein_protein_cutoff=None, protein_protein_cutoff_off=None) -> "ContactSettings":
        """Build settings with the package defaults.

        Parameters
        ----------
        cutoff : float, optional
            Protein-RNA cutoff in nm (default 1.5 nm).
        cutoff_off : float or None, optional
            Protein-RNA hysteresis release distance in nm, > ``cutoff``.
        protein_protein_cutoff : float or None, optional
            Protein-protein cutoff in nm. Default: same as ``cutoff``.
        protein_protein_cutoff_off : float or None, optional
            Protein-protein release distance in nm. Default: ``cutoff_off``
            when ``protein_protein_cutoff`` is not given (both kinds of
            contact then use identical settings), otherwise None.

        Returns
        -------
        ContactSettings

        Raises
        ------
        ValueError
            If a cutoff is not positive or a release distance is not larger
            than its cutoff.
        """
        same = protein_protein_cutoff is None
        pp = float(cutoff) if same else float(protein_protein_cutoff)
        pp_off = (cutoff_off if same else None) if protein_protein_cutoff_off is None else protein_protein_cutoff_off
        for name, c, off in (("cutoff", cutoff, cutoff_off), ("protein_protein_cutoff", pp, pp_off)):
            if not np.isfinite(float(c)) or float(c) <= 0:
                raise ValueError(f"{name} must be a positive number of nm, got {c}")
            if off is not None and not float(off) > float(c):
                raise ValueError(f"{name}_off ({off} nm) must be larger than {name} ({c} nm)")
        return cls(float(cutoff), None if cutoff_off is None else float(cutoff_off),
                   pp, None if pp_off is None else float(pp_off))

    def search_radius(self, kind: str) -> float:
        """Largest distance in nm at which a contact of ``kind`` ("pr" or "pp") can be on."""
        if kind == "pr":
            return self.cutoff if self.cutoff_off is None else self.cutoff_off
        return self.protein_protein_cutoff if self.protein_protein_cutoff_off is None else self.protein_protein_cutoff_off


# --------------------------------------------------------------------------- bead-level pair search

def _pairs_between(pa, pb, L, radius):
    """Bead pairs (i in A, j in B) within radius; returns i, j, distance, minimum-image vector b - a."""
    if len(pa) == 0 or len(pb) == 0:
        e = np.empty(0, dtype=np.intp)
        return e, e, np.empty(0), np.empty((0, 3))
    m = cKDTree(pa, boxsize=L).sparse_distance_matrix(cKDTree(pb, boxsize=L), max_distance=radius,
                                                      output_type="ndarray")
    i, j = m["i"].astype(np.intp), m["j"].astype(np.intp)
    v = pb[j] - pa[i]
    v -= L * np.round(v / L)
    return i, j, np.linalg.norm(v, axis=1), v


def _pairs_within(p, L, radius):
    """Bead pairs (i < j) within one set; returns i, j, distance, minimum-image vector j - i."""
    if len(p) < 2:
        e = np.empty(0, dtype=np.intp)
        return e, e, np.empty(0), np.empty((0, 3))
    pairs = cKDTree(p, boxsize=L).query_pairs(r=radius, output_type="ndarray")
    i, j = pairs[:, 0].astype(np.intp), pairs[:, 1].astype(np.intp)
    v = p[j] - p[i]
    v -= L * np.round(v / L)
    return i, j, np.linalg.norm(v, axis=1), v


def _chain_matrix(ci, cj, n_i, n_j, symmetric=False):
    m = np.zeros((n_i, n_j), dtype=bool)
    m[ci, cj] = True
    if symmetric:
        m[cj, ci] = True
    return m


def _image_shifts(ci, cj, ua, ub, v, L, on):
    """Unique (chain_i, chain_j, nx, ny, nz) rows for bead pairs whose chain pair is on.

    ``ua``/``ub`` are whole-chain (unwrapped) positions of the paired beads and
    ``v`` their minimum-image separation. The integer shift n says that the
    copy of chain j touching chain i sits at (whole chain j) - n * L.
    """
    keep = on[ci, cj]
    if not keep.any():
        return np.empty((0, 5), dtype=np.int64)
    n = np.rint((ub[keep] - ua[keep] - v[keep]) / L).astype(np.int64)
    rows = np.column_stack([ci[keep], cj[keep], n])
    return np.unique(rows, axis=0)


# --------------------------------------------------------------------------- single-frame contacts

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
    i, j, _, _ = _pairs_between(pa, pb, L, c)
    return _chain_matrix(ca[i], cb[j], int(n_chains_a), int(n_chains_b))


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
    i, j, _, _ = _pairs_within(p, L, c)
    inter = ch[i] != ch[j]
    return _chain_matrix(ch[i][inter], ch[j][inter], int(n_protein), int(n_protein), symmetric=True)


# --------------------------------------------------------------------------- multi-frame contacts

def update_hysteresis(previous, within_on, within_off) -> np.ndarray:
    """Advance hysteresis contact states by one frame.

    A contact is on in this frame if its minimum distance is within the
    on-cutoff now, or if it was on in the previous frame and its minimum
    distance is still within the release distance ``cutoff_off``.

    Parameters
    ----------
    previous : array_like of bool or None
        Contact states of the previous frame, or None for the first frame.
    within_on : array_like of bool
        Pairs whose minimum distance is <= cutoff (nm) in this frame.
    within_off : array_like of bool
        Pairs whose minimum distance is <= cutoff_off (nm) in this frame.

    Returns
    -------
    numpy.ndarray of bool
        Contact states for this frame, same shape as the inputs.
    """
    on = np.asarray(within_on, dtype=bool)
    if previous is None:
        return on.copy()
    return on | (np.asarray(previous, dtype=bool) & np.asarray(within_off, dtype=bool))


@dataclass(frozen=True)
class FrameContacts:
    """Chain-level contacts of one frame, after hysteresis if enabled.

    Attributes
    ----------
    index : int
        Trajectory frame index (or the position in the sequence).
    box : numpy.ndarray, shape (3,)
        Box edge lengths in nm.
    protein_rna : numpy.ndarray of bool, shape (n_rna, n_protein)
    protein_protein : numpy.ndarray of bool, shape (n_protein, n_protein)
    protein_rna_shifts : numpy.ndarray of int, shape (m, 5) or None
        Rows ``(rna, protein, nx, ny, nz)``: for each protein-RNA contact
        that is on, every distinct periodic image shift through which it is
        made (see :func:`iter_frame_contacts`). None unless requested.
    protein_protein_shifts : numpy.ndarray of int, shape (m, 5) or None
        Rows ``(protein_i, protein_j, nx, ny, nz)`` with i < j.
    """

    index: int
    box: np.ndarray
    protein_rna: np.ndarray
    protein_protein: np.ndarray
    protein_rna_shifts: np.ndarray | None = None
    protein_protein_shifts: np.ndarray | None = None


def iter_frame_contacts(frames, protein_chain_of_bead, n_protein, rna_chain_of_bead, n_rna,
                        settings: ContactSettings | None = None, image_shifts: bool = False):
    """Chain-level contacts frame by frame, with optional hysteresis.

    Parameters
    ----------
    frames : iterable
        Objects with ``box`` (shape (3,), nm), ``protein_positions`` and
        ``rna_positions`` (shape (n, 3), nm) and optionally ``index``, such
        as :class:`condensate_graph.system.Frame`, in time order.
    protein_chain_of_bead, rna_chain_of_bead : array_like of int
        Chain index of each protein / RNA bead.
    n_protein, n_rna : int
        Number of protein / RNA chains.
    settings : ContactSettings, optional
        Cutoffs in nm. Default: :meth:`ContactSettings.create` (1.5 nm, no
        hysteresis).
    image_shifts : bool, optional
        Also return, for every contact that is on, the periodic image shifts
        through which it is made. Chains are made whole with
        :func:`condensate_graph.box.unwrap_chains`; for a contacting bead pair
        the shift is the whole-chain separation minus the minimum-image
        separation, in units of the box length. Used for percolation.

    Yields
    ------
    FrameContacts

    Raises
    ------
    ValueError
        If a cutoff or release distance exceeds half the shortest box edge.
    """
    s = settings or ContactSettings.create()
    state_pr = state_pp = None
    for t, f in enumerate(frames):
        L = validate_box(f.box)
        r_pr, r_pp = check_cutoff(s.search_radius("pr"), L), check_cutoff(s.search_radius("pp"), L)
        check_cutoff(s.cutoff, L), check_cutoff(s.protein_protein_cutoff, L)
        pp_raw = np.reshape(f.protein_positions, (-1, 3))
        rna_raw = np.reshape(f.rna_positions, (-1, 3))
        pp_pos, rna_pos = wrap_positions(pp_raw, L), wrap_positions(rna_raw, L)
        cp = _chain_index(protein_chain_of_bead, n_protein, len(pp_pos), "protein_chain_of_bead")
        cr = _chain_index(rna_chain_of_bead, n_rna, len(rna_pos), "rna_chain_of_bead")

        i, j, d, v = _pairs_between(rna_pos, pp_pos, L, r_pr)
        on_now = _chain_matrix(cr[i[d <= s.cutoff]], cp[j[d <= s.cutoff]], n_rna, n_protein)
        if s.cutoff_off is None:
            state_pr = on_now
        else:
            state_pr = update_hysteresis(state_pr, on_now, _chain_matrix(cr[i], cp[j], n_rna, n_protein))

        a, b, dp, vp = _pairs_within(pp_pos, L, r_pp)
        inter = cp[a] != cp[b]
        a, b, dp, vp = a[inter], b[inter], dp[inter], vp[inter]
        near = dp <= s.protein_protein_cutoff
        on_pp = _chain_matrix(cp[a[near]], cp[b[near]], n_protein, n_protein, symmetric=True)
        if s.protein_protein_cutoff_off is None:
            state_pp = on_pp
        else:
            state_pp = update_hysteresis(state_pp, on_pp,
                                         _chain_matrix(cp[a], cp[b], n_protein, n_protein, symmetric=True))

        pr_shifts = pp_shifts = None
        if image_shifts:
            whole_p = unwrap_chains(pp_raw, cp, L)
            whole_r = unwrap_chains(rna_raw, cr, L)
            pr_shifts = _image_shifts(cr[i], cp[j], whole_r[i], whole_p[j], v, L, state_pr)
            # orient protein pairs as (lower index, higher index)
            swap = cp[a] > cp[b]
            a2, b2 = np.where(swap, b, a), np.where(swap, a, b)
            vp2 = np.where(swap[:, None], -vp, vp)
            pp_shifts = _image_shifts(cp[a2], cp[b2], whole_p[a2], whole_p[b2], vp2, L, state_pp)

        yield FrameContacts(index=int(getattr(f, "index", t)), box=L,
                            protein_rna=state_pr, protein_protein=state_pp,
                            protein_rna_shifts=pr_shifts, protein_protein_shifts=pp_shifts)
