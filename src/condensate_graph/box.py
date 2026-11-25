"""Periodic box handling.

All lengths in this module are in nanometres (nm), except where a function
explicitly takes MDAnalysis ``dimensions``, which MDAnalysis reports in
angstroms (A) together with the three cell angles in degrees.

Only orthorhombic boxes (all angles 90 degrees) are supported. There is no
non-periodic fallback: if no valid periodic box is available, a
:class:`BoxError` is raised so that contacts are never computed silently
without periodic images.
"""

from __future__ import annotations

import numpy as np

ANGSTROM_PER_NM = 10.0
_ANGLE_TOLERANCE_DEG = 1e-3

__all__ = ["BoxError", "validate_box", "box_from_dimensions", "resolve_box", "wrap_positions", "unwrap_chains"]


class BoxError(ValueError):
    """Raised when no valid periodic box is available or a box is malformed."""


def validate_box(box) -> np.ndarray:
    """Check an orthorhombic box and return it as a float64 array.

    Parameters
    ----------
    box : array_like, shape (3,)
        Box edge lengths ``(Lx, Ly, Lz)`` in nm.

    Returns
    -------
    numpy.ndarray, shape (3,), float64
        The box edge lengths in nm.

    Raises
    ------
    BoxError
        If the box is not three finite, strictly positive lengths.
    """
    arr = np.asarray(box, dtype=np.float64)
    if arr.shape != (3,):
        raise BoxError(
            f"box must have shape (3,) with edge lengths (Lx, Ly, Lz) in nm, got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise BoxError(f"box edge lengths must be finite and > 0 nm, got {arr.tolist()}")
    return arr


def box_from_dimensions(dimensions) -> np.ndarray:
    """Convert MDAnalysis ``dimensions`` to an orthorhombic box in nm.

    Parameters
    ----------
    dimensions : array_like of shape (6,) or None
        MDAnalysis unit cell ``[A, B, C, alpha, beta, gamma]``, with lengths
        in angstroms and angles in degrees (for example
        ``universe.trajectory.ts.dimensions``).

    Returns
    -------
    numpy.ndarray, shape (3,), float64
        Box edge lengths in nm.

    Raises
    ------
    BoxError
        If ``dimensions`` is None, has zero or invalid lengths (as written by
        many readers when the file stores no unit cell), or describes a
        non-orthorhombic cell.
    """
    if dimensions is None:
        raise BoxError(
            "the trajectory frame has no unit cell; pass the periodic box explicitly "
            "as box=(Lx, Ly, Lz) in nm"
        )
    dims = np.asarray(dimensions, dtype=np.float64)
    if dims.shape != (6,):
        raise BoxError(f"dimensions must have shape (6,), got shape {dims.shape}")
    lengths_a, angles = dims[:3], dims[3:]
    if not np.all(np.isfinite(lengths_a)) or np.any(lengths_a <= 0.0):
        raise BoxError(
            f"the trajectory frame has no valid unit cell (lengths {lengths_a.tolist()} A); "
            "pass the periodic box explicitly as box=(Lx, Ly, Lz) in nm"
        )
    if not np.allclose(angles, 90.0, atol=_ANGLE_TOLERANCE_DEG):
        raise BoxError(f"only orthorhombic boxes are supported, got cell angles {angles.tolist()} degrees")
    return validate_box(lengths_a / ANGSTROM_PER_NM)


def resolve_box(dimensions=None, box=None) -> np.ndarray:
    """Return the periodic box for one frame, in nm.

    An explicit ``box`` takes precedence over the trajectory's ``dimensions``.
    There is no fallback: if neither gives a valid periodic box, an error is
    raised.

    Parameters
    ----------
    dimensions : array_like of shape (6,) or None, optional
        MDAnalysis unit cell for the frame, lengths in angstroms and angles
        in degrees.
    box : array_like of shape (3,) or None, optional
        Explicit box edge lengths ``(Lx, Ly, Lz)`` in nm, for trajectories
        that do not store a unit cell.

    Returns
    -------
    numpy.ndarray, shape (3,), float64
        Box edge lengths in nm.

    Raises
    ------
    BoxError
        If ``box`` is given but invalid, or if ``box`` is None and
        ``dimensions`` does not describe a valid orthorhombic cell.
    """
    if box is not None:
        return validate_box(box)
    return box_from_dimensions(dimensions)


def wrap_positions(positions, box) -> np.ndarray:
    """Wrap positions into the primary cell ``[0, L)`` along each axis.

    The wrap is done in float64. Doing it in float32 can map a coordinate
    just below zero to exactly ``L``, which lies outside the half-open cell
    and is rejected by periodic KD-trees.

    Parameters
    ----------
    positions : array_like, shape (n, 3)
        Coordinates in nm, any float dtype.
    box : array_like, shape (3,)
        Box edge lengths in nm.

    Returns
    -------
    numpy.ndarray, shape (n, 3), float64
        Wrapped coordinates in nm, each component in ``[0, L)``.

    Raises
    ------
    BoxError
        If the box is invalid.
    ValueError
        If ``positions`` is not an (n, 3) array of finite values.
    """
    L = validate_box(box)
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape (n, 3) in nm, got shape {pos.shape}")
    if not np.all(np.isfinite(pos)):
        raise ValueError("positions contain NaN or infinite values")
    wrapped = np.mod(pos, L)
    # np.mod can return exactly L for tiny negative inputs after rounding
    wrapped[wrapped >= L] = 0.0
    return wrapped


def unwrap_chains(positions, chain_of_bead, box) -> np.ndarray:
    """Make every chain whole across the periodic boundaries.

    The first bead of each chain (in file order) is the chain's reference and
    keeps its given position. Every following bead of the same chain is
    placed at the minimum-image position relative to the previous bead. This
    is exact as long as consecutive beads of a chain are closer than half the
    box length along each axis, which holds for bonded beads.

    Parameters
    ----------
    positions : array_like, shape (n_beads, 3)
        Coordinates in nm, wrapped or not.
    chain_of_bead : array_like of int, shape (n_beads,)
        Chain index of each bead. Beads of a chain need not be contiguous in
        the array; their relative file order is kept.
    box : array_like, shape (3,)
        Box edge lengths in nm.

    Returns
    -------
    numpy.ndarray, shape (n_beads, 3), float64
        Coordinates in nm with each chain whole (not wrapped into the cell).

    Raises
    ------
    BoxError
        If the box is invalid.
    ValueError
        If the shapes do not match.
    """
    L = validate_box(box)
    pos = np.asarray(positions, dtype=np.float64)
    ch = np.asarray(chain_of_bead, dtype=np.intp)
    if pos.ndim != 2 or pos.shape[1] != 3 or ch.shape != (len(pos),):
        raise ValueError("positions must have shape (n, 3) and chain_of_bead shape (n,)")
    if len(pos) == 0:
        return pos.copy()
    order = np.argsort(ch, kind="stable")      # group by chain, keep file order inside each chain
    p, c = pos[order], ch[order]
    d = np.diff(p, axis=0)
    d -= L * np.round(d / L)
    same = c[1:] == c[:-1]
    step = np.vstack([np.zeros((1, 3)), np.where(same[:, None], d, 0.0)])
    cum = np.cumsum(step, axis=0)
    starts = np.concatenate(([True], ~same))
    first = np.maximum.accumulate(np.where(starts, np.arange(len(p)), 0))
    out = np.empty_like(pos)
    out[order] = p[first] + cum - cum[first]
    return out
