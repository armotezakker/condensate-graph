"""Bridging metrics on synthetic systems with hand-computable answers.

Box 15 x 15 x 60 nm, cutoff 1.5 nm. Protein and RNA chains are single beads
or a few beads placed by hand.
"""
import numpy as np
import pytest

from condensate_graph.bridging import (
    bridging_metrics,
    chain_center_z,
    productive_bridge_flags,
    region_mask,
    valency_distribution,
)
from condensate_graph.system import Frame

BOX = np.array([15.0, 15.0, 60.0])
FAR = [7.5, 7.5, 50.0]  # a place far from everything else


def frame(protein, rna, index=0):
    return Frame(index=index, time_ps=0.0, box=BOX,
                 protein_positions=np.array(protein, dtype=np.float64),
                 rna_positions=np.array(rna, dtype=np.float64))


def run(frames, n_protein, rna_chain, n_rna, **kw):
    return bridging_metrics(frames, np.arange(n_protein), n_protein, np.asarray(rna_chain), n_rna, **kw)


def test_bridge_between_touching_proteins_is_basic_not_productive():
    # proteins 1.0 nm apart (in contact); RNA bead 0.5/0.8 nm from them
    f = frame([[5.0, 5.0, 30.0], [6.0, 5.0, 30.0]], [[5.5, 5.8, 30.0]])
    r = run([f], 2, [0], 1)
    assert r.f_bridge_basic[0] == 1.0
    assert r.f_bridge_productive[0] == 0.0
    assert r.mean_valency[0] == 2.0


def test_bridge_between_separated_proteins_is_productive():
    # proteins 5 nm apart; a two-bead RNA touches each at 0.5 nm
    f = frame([[5.0, 5.0, 30.0], [10.0, 5.0, 30.0]], [[5.5, 5.0, 30.0], [9.5, 5.0, 30.0]])
    r = run([f], 2, [0, 0], 1)
    assert r.f_bridge_basic[0] == 1.0
    assert r.f_bridge_productive[0] == 1.0


def test_bridge_only_across_periodic_boundary_is_counted():
    # proteins at x = 1.0 and x = 14.0: 2.0 nm apart via the boundary, not in contact.
    # RNA at x = 14.9: 0.9 nm from protein 1, 1.1 nm from protein 0 across the x boundary.
    f = frame([[1.0, 7.0, 30.0], [14.0, 7.0, 30.0]], [[14.9, 7.0, 30.0]])
    r = run([f], 2, [0], 1)
    assert r.f_bridge_basic[0] == 1.0
    assert r.f_bridge_productive[0] == 1.0


def test_productive_with_three_partners_two_touching():
    # partners 0,1,2; only 0-1 touch -> pairs (0,2), (1,2) are non-touching -> productive
    pr = np.array([[True, True, True]])
    pp = np.zeros((3, 3), bool); pp[0, 1] = pp[1, 0] = True
    assert productive_bridge_flags(pr, pp)[0]
    pp[:] = True; np.fill_diagonal(pp, False)  # all three touch each other
    assert not productive_bridge_flags(pr, pp)[0]


def test_persistence_k2_exact_flags():
    proteins = [[5.0, 5.0, 30.0], [6.0, 5.0, 30.0]]
    bridged, away = [[5.5, 5.8, 30.0]], [FAR]
    rna_by_frame = [bridged, bridged, away, bridged, bridged]  # frames 1..5
    frames = [frame(proteins, rna, i) for i, rna in enumerate(rna_by_frame)]
    r = run(frames, 2, [0], 1, min_frames=(1, 2))
    np.testing.assert_array_equal(r.persistent_flags[2][:, 0], [False, True, False, False, True])
    np.testing.assert_array_equal(r.f_bridge_persistent[2], [np.nan, 1.0, 0.0, 0.0, 1.0])
    # k = 1 equals the basic bridge
    np.testing.assert_array_equal(r.f_bridge_persistent[1], r.f_bridge_basic)
    np.testing.assert_array_equal(r.f_bridge_basic, [1.0, 1.0, 0.0, 1.0, 1.0])
    assert r.means()["f_bridge_persistent"][2] == pytest.approx(0.5)  # mean of 1, 0, 0, 1


def test_persistence_requires_the_same_pair():
    # frame 0: RNA bridges proteins 0 and 1; frame 1: proteins 1 and 2 -> no common pair
    proteins = [[3.0, 7.0, 30.0], [6.0, 7.0, 30.0], [9.0, 7.0, 30.0]]
    f0 = frame(proteins, [[4.5, 7.0, 30.0]], 0)   # 1.5 from protein 0 and 1.5 from protein 1
    f1 = frame(proteins, [[7.5, 7.0, 30.0]], 1)   # 1.5 from protein 1 and 1.5 from protein 2
    r = run([f0, f1], 3, [0], 1, min_frames=2)
    np.testing.assert_array_equal(r.f_bridge_basic, [1.0, 1.0])
    np.testing.assert_array_equal(r.persistent_flags[2][:, 0], [False, False])


def test_chain_center_z_unwraps_across_boundary():
    # beads at z = 59.8 and z = 0.4: whole chain centre at z = 0.1, not the naive 30.1
    cz = chain_center_z([[7, 7, 59.8], [7, 7, 0.4]], [0, 0], 1, 60.0)
    assert cz[0] == pytest.approx(0.1)


def test_region_filter_includes_and_excludes_right_chains():
    proteins = [[5.0, 5.0, 30.0], [6.0, 5.0, 30.0]]
    rna = [
        [5.5, 5.8, 30.0],                     # chain 0: centre z = 30, bridging
        [5.5, 5.8, 10.0],                     # chain 1: centre z = 10
        [7.0, 7.0, 59.8], [7.0, 7.0, 0.4],    # chain 2: straddles z boundary, centre 0.1 (naive mean 30.1)
    ]
    chain = [0, 1, 2, 2]
    r = run([frame(proteins, rna)], 2, chain, 3, region=(25.0, 35.0))
    np.testing.assert_array_equal(r.counted[0], [True, False, False])
    assert r.n_rna_counted[0] == 1 and r.f_bridge_basic[0] == 1.0
    r = run([frame(proteins, rna)], 2, chain, 3, region=(0.0, 1.0))
    np.testing.assert_array_equal(r.counted[0], [False, False, True])
    assert r.f_bridge_basic[0] == 0.0
    r = run([frame(proteins, rna)], 2, chain, 3)
    assert r.counted.all() and r.f_bridge_basic[0] == pytest.approx(1 / 3)


def test_region_outside_box_raises():
    with pytest.raises(ValueError, match="region"):
        region_mask([1.0], (10.0, 70.0), 60.0)


def test_empty_region_gives_nan():
    r = run([frame([[5.0, 5.0, 30.0]], [[5.5, 5.0, 30.0]])], 1, [0], 1, region=(40.0, 45.0))
    assert np.isnan(r.f_bridge_basic[0]) and np.isnan(r.valency_distribution[0]).all()


def test_valency_distribution_sums_to_one_every_frame():
    rng = np.random.default_rng(7)
    n_protein, n_rna, beads = 12, 20, 3
    slab = BOX * [1, 1, 0.05]  # both species in a dense 15 x 15 x 3 nm slab
    frames = [frame(rng.uniform(0, 1, (n_protein, 3)) * slab,
                    rng.uniform(0, 1, (n_rna * beads, 3)) * slab, i) for i in range(6)]
    r = run(frames, n_protein, np.repeat(np.arange(n_rna), beads), n_rna)
    np.testing.assert_allclose(r.valency_distribution.sum(axis=1), 1.0)
    assert (r.valency_distribution[:, 1:].sum(axis=1) > 0).all()  # contacts actually occur


def test_valency_distribution_bins():
    np.testing.assert_allclose(valency_distribution([0, 1, 2, 3, 4, 7, 2, 0]),
                               [2 / 8, 1 / 8, 2 / 8, 1 / 8, 2 / 8])


@pytest.mark.parametrize("bad", [0, (0, 2), ()])
def test_invalid_min_frames_raise(bad):
    with pytest.raises(ValueError, match="min_frames"):
        run([frame([[5.0, 5.0, 30.0]], [[5.5, 5.0, 30.0]])], 1, [0], 1, min_frames=bad)
