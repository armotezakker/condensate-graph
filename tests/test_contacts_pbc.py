"""Synthetic periodic-boundary checks, built in code with no input files.

Box 15 x 15 x 60 nm, cutoff 1.5 nm unless stated otherwise.
"""
import numpy as np
import pytest

from condensate_graph.box import BoxError
from condensate_graph.contacts import (
    chain_contacts,
    check_cutoff,
    protein_protein_contacts,
    protein_rna_contacts,
)

BOX = np.array([15.0, 15.0, 60.0])


def one_bead_pair(p, r):
    """One protein bead and one RNA bead, each its own chain; returns the (1, 1) contact."""
    return protein_rna_contacts([p], [0], 1, [r], [0], 1, BOX)[0, 0]


def test_contact_across_x_boundary_found():
    # raw distance 14.5 nm, minimum-image distance 0.5 nm
    assert one_bead_pair([0.2, 7.0, 30.0], [14.7, 7.0, 30.0])


def test_contact_across_z_boundary_found():
    # raw distance 59.5 nm, minimum-image distance 0.5 nm
    assert one_bead_pair([7.0, 7.0, 0.2], [7.0, 7.0, 59.7])


def test_contact_across_corner_found():
    # 0.3 nm in each axis across all three boundaries: distance ~0.52 nm
    assert one_bead_pair([0.1, 0.1, 0.1], [14.8, 14.8, 59.8])


def test_pair_3nm_apart_not_found():
    assert not one_bead_pair([5.0, 7.0, 30.0], [8.0, 7.0, 30.0])


def test_pair_3nm_apart_across_boundary_not_found():
    assert not one_bead_pair([1.0, 7.0, 30.0], [13.0, 7.0, 30.0])


def test_float32_just_below_zero_does_not_crash():
    # in float32, -1e-8 mod 15 rounds to exactly 15.0, outside [0, 15)
    p = np.array([[-1e-8, 7.0, 30.0]], dtype=np.float32)
    r = np.array([[14.6, 7.0, 30.0]], dtype=np.float32)
    assert np.float32(np.mod(p[0, 0], np.float32(15.0))) == np.float32(15.0)  # the failure case exists
    assert protein_rna_contacts(p, [0], 1, r, [0], 1, BOX)[0, 0]


def test_float64_tiny_negative_does_not_crash():
    # -1e-20 mod 15 rounds to exactly 15.0 even in float64
    assert one_bead_pair([-1e-20, 7.0, 30.0], [14.6, 7.0, 30.0])


def test_missing_box_raises():
    with pytest.raises(BoxError):
        protein_rna_contacts([[0.0, 0.0, 0.0]], [0], 1, [[0.5, 0.0, 0.0]], [0], 1, None)


def test_chain_level_aggregation_and_orientation():
    # protein chain 0 = beads 0,1; chain 1 = bead 2. RNA chain 0 = bead 0; chain 1 = beads 1,2.
    prot = [[1.0, 1.0, 10.0], [5.0, 5.0, 10.0], [10.0, 10.0, 10.0]]
    rna = [[14.6, 1.0, 10.0], [10.0, 10.0, 11.0], [5.0, 5.0, 11.2]]
    c = protein_rna_contacts(prot, [0, 0, 1], 2, rna, [0, 1, 1], 2, BOX)
    assert c.shape == (2, 2)  # (n_rna, n_protein)
    np.testing.assert_array_equal(c, [[True, False], [True, True]])


def test_protein_protein_symmetric_and_excludes_self():
    # chain 0: two beads 0.5 nm apart (intra-chain, ignored); chain 1 0.5 nm from chain 0 across x
    pos = [[0.2, 7.0, 30.0], [0.7, 7.0, 30.0], [14.7, 7.0, 30.0], [7.0, 7.0, 45.0]]
    c = protein_protein_contacts(pos, [0, 0, 1, 2], 3, BOX)
    np.testing.assert_array_equal(c, c.T)
    assert not c.diagonal().any()
    assert c[0, 1] and not c[0, 2] and not c[1, 2]


def test_cutoff_inclusive_boundary():
    assert chain_contacts([[0.0, 0.0, 0.0]], [0], 1, [[1.5, 0.0, 0.0]], [0], 1, BOX, cutoff=1.5)[0, 0]


def test_cutoff_larger_than_half_box_raises():
    with pytest.raises(ValueError, match="half the shortest box edge"):
        check_cutoff(8.0, BOX)


def test_empty_sets_return_empty_contacts():
    c = protein_rna_contacts(np.empty((0, 3)), np.empty(0, int), 0, [[1.0, 1.0, 1.0]], [0], 1, BOX)
    assert c.shape == (1, 0)
