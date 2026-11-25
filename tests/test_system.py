"""Chain identification on small MDAnalysis universes built in memory."""
import warnings

import numpy as np
import pytest
import MDAnalysis as mda

from condensate_graph.box import BoxError
from condensate_graph.system import ChainIdentificationWarning, build_system, chains_from_topology, fragment_problems

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


# ------------------------------------------------ broken fragments (bonds partly missing)

def chain_universe(n_chains=20, length=30, drop_bond=None, one_segment=False):
    """n_chains linear chains of `length` beads; drop_bond(chain, i) -> True removes bond i-(i+1)."""
    n = n_chains * length
    chain = np.repeat(np.arange(n_chains), length)
    segindex = np.zeros(n, dtype=int) if one_segment else chain
    n_seg = 1 if one_segment else n_chains
    u = mda.Universe.empty(n, n_residues=n, n_segments=n_seg, atom_resindex=np.arange(n),
                           residue_segindex=segindex, trajectory=True)
    u.add_TopologyAttr("name", ["CA"] * n)
    u.add_TopologyAttr("segid", [f"S{k}" for k in range(n_seg)])
    bonds = [(i, i + 1) for i in range(n - 1) if chain[i] == chain[i + 1]
             and not (drop_bond and drop_bond(chain[i], i % length))]
    u.add_TopologyAttr("bonds", bonds)
    u.atoms.positions = np.zeros((n, 3), np.float32)
    u.dimensions = [1000.0, 1000.0, 1000.0, 90, 90, 90]
    return u


def drop_some(chain, i):
    # chains 0-9 lose every 6th bond, chains 10-19 keep all their bonds
    return chain < 10 and i % 6 == 5


def test_fragment_with_missing_bonds_warns_with_both_counts():
    u = chain_universe(drop_bond=drop_some)
    with pytest.warns(ChainIdentificationWarning) as rec:
        chains = chains_from_topology(u.atoms, by="fragment", name="protein")
    msg = str(rec[0].message)
    assert chains.n_chains == 10 * 5 + 10
    assert "60 fragments but 20 segments / chain IDs" in msg
    assert "try by='segment'" in msg
    assert "50 of 60 fragments" in msg  # 6-bead pieces are below a quarter of the typical 30


def test_fragment_size_check_works_without_segment_information():
    u = chain_universe(drop_bond=drop_some, one_segment=True)
    with pytest.warns(ChainIdentificationWarning, match="fragments .* have fewer than") as rec:
        chains_from_topology(u.atoms, by="fragment")
    assert "fragments but" not in str(rec[0].message)


def test_by_segment_recovers_chains_when_bonds_are_missing():
    u = chain_universe(drop_bond=drop_some)
    chains = chains_from_topology(u.atoms, by="segment")
    assert chains.n_chains == 20 and set(chains.chain_sizes().tolist()) == {30}


def test_complete_fragments_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ChainIdentificationWarning)
        chains = chains_from_topology(chain_universe().atoms, by="fragment")
    assert chains.n_chains == 20
    assert fragment_problems(chains) == []


def test_fragment_crossing_segments_is_reported():
    # chain bonds complete, plus one extra bond joining chains 0 and 1
    u = chain_universe()
    u.add_bonds([(29, 30)])
    with pytest.warns(ChainIdentificationWarning, match=r"19 fragments but 20 segments .*1 fragments span more than one segment"):
        chains_from_topology(u.atoms, by="fragment")
