"""Hysteresis contacts and separate protein-protein cutoffs (box 15 x 15 x 60 nm)."""
import numpy as np
import pytest

from condensate_graph.bridging import bridging_metrics
from condensate_graph.contacts import ContactSettings, iter_frame_contacts, update_hysteresis
from condensate_graph.system import Frame

BOX = np.array([15.0, 15.0, 60.0])


def frame(protein, rna, index=0):
    return Frame(index=index, time_ps=0.0, box=BOX,
                 protein_positions=np.array(protein, dtype=np.float64),
                 rna_positions=np.array(rna, dtype=np.float64))


def pair_states(distances, **settings):
    """One protein bead and one RNA bead at the given separations along x (nm)."""
    frames = [frame([[5.0, 7.0, 30.0]], [[5.0 + d, 7.0, 30.0]], i) for i, d in enumerate(distances)]
    s = ContactSettings.create(**settings)
    return [bool(fc.protein_rna[0, 0]) for fc in iter_frame_contacts(frames, [0], 1, [0], 1, s)]


def test_oscillating_pair_is_continuous_with_hysteresis():
    d = [1.4, 1.8, 1.4, 1.8, 1.4, 1.8]
    assert pair_states(d, cutoff=1.5, cutoff_off=2.0) == [True] * 6


def test_oscillating_pair_is_broken_without_hysteresis():
    d = [1.4, 1.8, 1.4, 1.8, 1.4, 1.8]
    assert pair_states(d, cutoff=1.5) == [True, False, True, False, True, False]


def test_hysteresis_needs_switch_on_and_releases_beyond_cutoff_off():
    # starts between the cutoffs (never switched on), switches on at 1.4,
    # survives 1.8, breaks at 2.2, does not come back at 1.8
    assert pair_states([1.8, 1.4, 1.8, 2.2, 1.8, 1.4], cutoff=1.5, cutoff_off=2.0) == \
        [False, True, True, False, False, True]


def test_hysteresis_across_periodic_boundary():
    # separation measured across the x boundary: protein at x = 0.5, RNA at 15.5 - d
    frames = [frame([[0.5, 7.0, 30.0]], [[15.5 - d, 7.0, 30.0]], i) for i, d in enumerate([1.4, 1.8, 1.4])]
    s = ContactSettings.create(cutoff=1.5, cutoff_off=2.0)
    assert [bool(fc.protein_rna[0, 0]) for fc in iter_frame_contacts(frames, [0], 1, [0], 1, s)] == [True] * 3


def test_update_hysteresis_rule():
    prev = np.array([True, True, False, False])
    on = np.array([False, False, True, False])
    off = np.array([True, False, True, True])
    np.testing.assert_array_equal(update_hysteresis(prev, on, off), [True, False, True, False])
    np.testing.assert_array_equal(update_hysteresis(None, on, off), on)


def bridge_frames(far_distances):
    # proteins 5 nm apart (never in direct contact); RNA bead 0 sits 0.5 nm from
    # protein 0, RNA bead 1 is at the given distance from protein 1
    proteins = [[5.0, 7.0, 30.0], [10.0, 7.0, 30.0]]
    return [frame(proteins, [[5.5, 7.0, 30.0], [10.0 - d, 7.0, 30.0]], i) for i, d in enumerate(far_distances)]


def test_persistence_with_and_without_hysteresis():
    frames = bridge_frames([1.4, 1.8, 1.4, 1.8, 1.4, 1.8])
    plain = bridging_metrics(frames, [0, 1], 2, [0, 0], 1, min_frames=4)
    hyst = bridging_metrics(frames, [0, 1], 2, [0, 0], 1, min_frames=4, cutoff_off=2.0)
    np.testing.assert_array_equal(plain.persistent_flags[4][:, 0], [False] * 6)
    np.testing.assert_array_equal(hyst.persistent_flags[4][:, 0], [False, False, False, True, True, True])
    np.testing.assert_array_equal(plain.f_bridge_basic, [1, 0, 1, 0, 1, 0])
    np.testing.assert_array_equal(hyst.f_bridge_basic, [1] * 6)


def test_separate_protein_protein_cutoff_changes_productive_only():
    # proteins 1.2 nm apart: in contact at 1.5 nm, not at 1.0 nm. RNA touches both.
    f = frame([[5.0, 7.0, 30.0], [6.2, 7.0, 30.0]], [[5.6, 7.8, 30.0]])
    same = bridging_metrics([f], [0, 1], 2, [0], 1)
    sep = bridging_metrics([f], [0, 1], 2, [0], 1, protein_protein_cutoff=1.0)
    assert same.f_bridge_basic[0] == sep.f_bridge_basic[0] == 1.0
    assert same.f_bridge_productive[0] == 0.0
    assert sep.f_bridge_productive[0] == 1.0


def test_settings_defaults_and_validation():
    s = ContactSettings.create(cutoff=1.5, cutoff_off=2.0)
    assert (s.protein_protein_cutoff, s.protein_protein_cutoff_off) == (1.5, 2.0)
    s = ContactSettings.create(cutoff=1.5, cutoff_off=2.0, protein_protein_cutoff=1.0)
    assert (s.protein_protein_cutoff, s.protein_protein_cutoff_off) == (1.0, None)
    with pytest.raises(ValueError, match="must be larger"):
        ContactSettings.create(cutoff=1.5, cutoff_off=1.5)
    with pytest.raises(ValueError, match="must be larger"):
        ContactSettings.create(protein_protein_cutoff=1.0, protein_protein_cutoff_off=0.8)


def test_release_distance_beyond_half_box_raises():
    s = ContactSettings.create(cutoff=1.5, cutoff_off=8.0)
    with pytest.raises(ValueError, match="half the shortest box edge"):
        list(iter_frame_contacts([frame([[5.0, 7.0, 30.0]], [[6.0, 7.0, 30.0]])], [0], 1, [0], 1, s))
