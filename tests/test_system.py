"""Chain identification on small MDAnalysis universes built in memory."""
import warnings

import numpy as np
import pytest
import MDAnalysis as mda

from condensate_graph.box import BoxError
from condensate_graph.system import ChainIdentificationWarning, build_system

# 2 protein chains (3 + 2 beads, one residue per bead) and 2 RNA chains (2 + 4 beads)
PROTEIN_LENGTHS = [3, 2]
RNA_LENGTHS = [2, 4]


def make_universe(segments="per_chain", bonds=True, dimensions=(150.0, 150.0, 600.0, 90.0, 90.0, 90.0)):
    """Build a universe of 11 beads.

    segments="per_chain": each chain its own segment (A, B, R1, R2).
    segments="merged_rna": both RNA chains share one segment R, as in files
    that write all RNA under a single chain ID.
    """
    chain_lengths = PROTEIN_LENGTHS + RNA_LENGTHS
    n_atoms = sum(chain_lengths)
    chain_of_atom = np.repeat(np.arange(4), chain_lengths)
    if segments == "per_chain":
        segids, residue_segindex = ["A", "B", "R1", "R2"], chain_of_atom
    else:
        segids, residue_segindex = ["A", "B", "R"], np.minimum(chain_of_atom, 2)
    u = mda.Universe.empty(n_atoms, n_residues=n_atoms, n_segments=len(segids),
                           atom_resindex=np.arange(n_atoms), residue_segindex=residue_segindex,
                           trajectory=True)
    u.add_TopologyAttr("name", ["CA"] * 5 + ["P", "S", "P", "S", "P", "S"])
    u.add_TopologyAttr("resid", np.arange(1, n_atoms + 1))
    u.add_TopologyAttr("segid", segids)
    if bonds:
        b = [(i, i + 1) for i in range(n_atoms - 1) if chain_of_atom[i] == chain_of_atom[i + 1]]
        u.add_TopologyAttr("bonds", b)
    u.atoms.positions = np.arange(n_atoms * 3, dtype=np.float32).reshape(n_atoms, 3) * 10.0
    if dimensions is not None:
        u.dimensions = dimensions
    return u


SEL = dict(protein_selection="name CA", rna_selection="name P S")


def test_by_segment():
    s = build_system(make_universe(), **SEL, by="segment")
    assert s.protein.labels == ("A", "B") and s.rna.labels == ("R1", "R2")
    assert s.protein.chain_sizes().tolist() == PROTEIN_LENGTHS
    assert s.rna.chain_sizes().tolist() == RNA_LENGTHS


def test_by_segment_single_chain_warns():
    with pytest.warns(ChainIdentificationWarning, match="only 1 chain in the rna selection") as rec:
        s = build_system(make_universe(segments="merged_rna"), **SEL, by="segment")
    assert "by='fragment'" in str(rec[0].message) and "chain_lengths" in str(rec[0].message)
    assert s.rna.n_chains == 1


def test_by_segment_no_warning_when_chains_found():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ChainIdentificationWarning)
        build_system(make_universe(), **SEL, by="segment")


def test_by_fragment():
    s = build_system(make_universe(segments="merged_rna"), **SEL, by="fragment")
    assert s.protein.chain_sizes().tolist() == PROTEIN_LENGTHS
    assert s.rna.chain_sizes().tolist() == RNA_LENGTHS


def test_by_fragment_without_bonds_raises():
    with pytest.raises(ValueError, match="needs bonds in the topology"):
        build_system(make_universe(bonds=False), **SEL, by="fragment")


def test_chain_lengths():
    s = build_system(make_universe(segments="merged_rna", bonds=False), **SEL,
                     chain_lengths={"protein": PROTEIN_LENGTHS, "rna": RNA_LENGTHS})
    assert s.protein.chain_of_atom.tolist() == [0, 0, 0, 1, 1]
    assert s.rna.chain_of_atom.tolist() == [0, 0, 1, 1, 1, 1]


def test_chain_lengths_sum_mismatch_reports_both_numbers():
    with pytest.raises(ValueError, match=r"add up to 7 beads, but the rna selection has 6 atoms"):
        build_system(make_universe(), **SEL, chain_lengths={"protein": PROTEIN_LENGTHS, "rna": [3, 4]})


@pytest.mark.parametrize("bad", [[5, 0], [6, -1], [2.0, 4.0], []])
def test_chain_lengths_invalid_values_raise(bad):
    with pytest.raises(ValueError, match="chain_lengths"):
        build_system(make_universe(), **SEL, chain_lengths={"protein": PROTEIN_LENGTHS, "rna": bad})


def test_chain_lengths_missing_key_raises():
    with pytest.raises(ValueError, match="exactly the keys"):
        build_system(make_universe(), **SEL, chain_lengths={"rna": RNA_LENGTHS})


@pytest.mark.parametrize("kwargs", [{}, {"by": "segment", "chain_lengths": {"protein": [5], "rna": [6]}}])
def test_neither_or_both_methods_raise(kwargs):
    with pytest.raises(ValueError, match="exactly one of"):
        build_system(make_universe(), **SEL, **kwargs)


def test_unknown_by_raises():
    with pytest.raises(ValueError, match="by must be one of"):
        build_system(make_universe(), **SEL, by="molnum")


def test_empty_selection_raises():
    with pytest.raises(ValueError, match="matched no atoms"):
        build_system(make_universe(), protein_selection="name XX", rna_selection="name P S", by="segment")


def test_overlapping_selections_raise():
    with pytest.raises(ValueError, match="share"):
        build_system(make_universe(), protein_selection="name CA P", rna_selection="name P S", by="segment")


def test_iter_frames_converts_to_nm_and_reads_box():
    s = build_system(make_universe(), **SEL, by="segment")
    f = next(s.iter_frames())
    np.testing.assert_allclose(f.box, [15.0, 15.0, 60.0])
    np.testing.assert_allclose(f.protein_positions[1], [3.0, 4.0, 5.0])  # (30, 40, 50) A


def test_iter_frames_without_box_raises():
    s = build_system(make_universe(dimensions=None), **SEL, by="segment")
    with pytest.raises(BoxError):
        next(s.iter_frames())


def test_explicit_box_used_when_trajectory_has_none():
    s = build_system(make_universe(dimensions=None), **SEL, by="segment", box=(15.0, 15.0, 60.0))
    np.testing.assert_allclose(next(s.iter_frames()).box, [15.0, 15.0, 60.0])
