"""Graph types, hubs and periodic percolation on synthetic systems.

Box 15 x 15 x 60 nm, cutoff 1.5 nm (inclusive).
"""
import numpy as np
import pytest

from condensate_graph.graph import build_graphs, graph_metrics, spanning_axes, top_hubs
from condensate_graph.system import Frame

BOX = np.array([15.0, 15.0, 60.0])
NO_RNA = (np.empty((0, 3)), np.empty(0, dtype=int), 0)


def frame(protein, rna):
    return Frame(index=0, time_ps=0.0, box=BOX,
                 protein_positions=np.array(protein, dtype=np.float64).reshape(-1, 3),
                 rna_positions=np.array(rna, dtype=np.float64).reshape(-1, 3))


def run(protein, protein_chain, n_protein, rna=NO_RNA, **kw):
    rna_pos, rna_chain, n_rna = rna
    f = frame(protein, rna_pos)
    return graph_metrics([f], np.asarray(protein_chain), n_protein, np.asarray(rna_chain), n_rna, **kw)


def ring_chains(axis, skip=None, offset=1.0):
    """Five 3-bead protein chains along ``axis``, beads 1 nm apart, chains 1 nm apart.

    Chain k has beads at 3k + 0.25 + offset + (0, 1, 2) along the axis, so the
    ring closes across the periodic boundary of a 15 nm axis. With offset 1,
    chain 4 itself straddles the boundary (beads at 13.25, 14.25, 15.25 -> 0.25).
    """
    pos, chain = [], []
    for k in range(5):
        if k == skip:
            continue
        for b in range(3):
            p = [7.0, 7.0, 30.0]
            p[axis] = (3 * k + 0.25 + offset + b) % 15.0
            pos.append(p)
            chain.append(k)
    _, chain = np.unique(chain, return_inverse=True)  # renumber chains 0..n-1
    return pos, chain, len(set(chain.tolist()))


@pytest.mark.parametrize("axis", [0, 1])
def test_ring_across_boundary_percolates_along_that_axis_only(axis):
    pos, chain, n = ring_chains(axis)
    s = run(pos, chain, n).series["pp"]
    assert s.largest_fraction[0] == 1.0
    expected = np.zeros(3, dtype=bool); expected[axis] = True
    np.testing.assert_array_equal(s.percolates[0], expected)


def test_open_line_crossing_boundary_does_not_percolate():
    # same ring with chain 2 removed: chains 3-4-0-1 form an open line that
    # crosses the x boundary but does not close
    pos, chain, n = ring_chains(0, skip=2)
    s = run(pos, chain, n).series["pp"]
    assert s.largest_fraction[0] == 1.0 and s.n_components[0] == 1
    np.testing.assert_array_equal(s.percolates[0], [False, False, False])


def test_large_compact_component_does_not_percolate():
    # 4 x 4 x 4 lattice of single-bead chains, spacing 1 nm, in the middle of the box
    g = np.arange(4) + 5.5
    pos = np.array([[x, y, z + 24.0] for x in g for y in g for z in g])
    s = run(pos, np.arange(len(pos)), len(pos)).series["pp"]
    assert s.largest_fraction[0] == 1.0
    np.testing.assert_array_equal(s.percolates[0], [False, False, False])


def test_graph_types_pp_bridged_combined_bipartite():
    # proteins 0-1 touch (1.0 nm), proteins 2-3 touch (1.0 nm), the pairs are 4 nm apart;
    # one two-bead RNA touches protein 1 and protein 2 (0.71 nm) but not 0 or 3 (1.58 nm)
    protein = [[6.0, 4.0, 30.0], [6.0, 5.0, 30.0], [10.0, 5.0, 30.0], [10.0, 4.0, 30.0]]
    rna = (np.array([[6.5, 5.5, 30.0], [9.5, 5.5, 30.0]]), np.array([0, 0]), 1)
    r = run(protein, np.arange(4), 4, rna)
    comps = {k: v.n_components[0] for k, v in r.series.items()}
    frac = {k: v.largest_fraction[0] for k, v in r.series.items()}
    assert comps == {"pp": 2, "bridged": 3, "combined": 1, "bipartite": 3}
    assert frac == {"pp": 0.5, "bridged": 0.5, "combined": 1.0, "bipartite": 0.5}
    g = build_graphs(np.array([[False, True, True, False]]),
                     np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=bool))
    assert set(g["bridged"].edges()) == {(("P", 1), ("P", 2))}
    assert g["combined"].number_of_edges() == 3


def test_ring_bridged_only_by_rna_percolates_in_bridged_not_pp():
    # proteins 3 nm apart along x (no direct contact); single-bead RNA chains
    # midway, each 1.5 nm from two proteins; the last RNA sits at x = 0 and
    # bridges proteins at x = 13.5 and x = 1.5 across the boundary
    protein = [[x, 7.0, 30.0] for x in (1.5, 4.5, 7.5, 10.5, 13.5)]
    rna_pos = np.array([[x, 7.0, 30.0] for x in (3.0, 6.0, 9.0, 12.0, 0.0)])
    r = run(protein, np.arange(5), 5, (rna_pos, np.arange(5), 5))
    perc = {k: v.percolates[0].tolist() for k, v in r.series.items()}
    assert perc["pp"] == [False, False, False]
    for name in ("bridged", "combined", "bipartite"):
        assert perc[name] == [True, False, False]


def test_star_hub_by_degree_and_betweenness():
    centre = [7.0, 7.0, 30.0]
    leaves = [[7.0 + dx, 7.0 + dy, 30.0 + dz] for dx, dy, dz in
              [(1.4, 0, 0), (-1.4, 0, 0), (0, 1.4, 0), (0, -1.4, 0), (0, 0, 1.4), (0, 0, -1.4)]]
    s = run([centre] + leaves, np.arange(7), 7).series["pp"]
    assert s.top_degree[0][0] == (("P", 0), 6)
    hub, value = s.top_betweenness[0][0]
    assert hub == ("P", 0) and value == pytest.approx(1.0)
    assert s.degree_histogram[0].tolist() == [0, 6, 0, 0, 0, 0, 1]


def test_spanning_axes_rules():
    zero, x = (0, 0, 0), (1, 0, 0)
    # triangle with no net shift: compact
    assert spanning_axes([("a", "b", zero), ("b", "c", zero), ("c", "a", zero)], "a").tolist() == [False] * 3
    # same triangle, one edge crosses the box: winds once in x
    assert spanning_axes([("a", "b", zero), ("b", "c", zero), ("c", "a", x)], "a").tolist() == [True, False, False]
    # crossing the boundary and coming back (net shift zero) does not span
    assert spanning_axes([("a", "b", x), ("b", "c", (-1, 0, 0)), ("c", "a", zero)], "a").tolist() == [False] * 3
    # two contacts between the same chains through different images: spans
    assert spanning_axes([("a", "b", zero), ("a", "b", (0, 0, 1))], "a").tolist() == [False, False, True]
    # isolated start node
    assert spanning_axes([], "a").tolist() == [False] * 3


def test_top_hubs_invalid_ranking_raises():
    g = build_graphs(np.zeros((0, 2), bool), np.zeros((2, 2), bool))["pp"]
    with pytest.raises(ValueError, match="by must be"):
        top_hubs(g, by="closeness")
