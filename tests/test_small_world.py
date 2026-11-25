"""Small-world coefficients on graphs with known behaviour.

The coefficients are stochastic; tolerances are wide enough for any seed in
practice, and fixed seeds keep the tests reproducible.
"""
import math

import networkx as nx
import numpy as np
import pytest

from condensate_graph.graph import SMALL_WORLD_WINDOW, graph_metrics, small_world
from condensate_graph.system import Frame


def test_ring_lattice_omega_near_minus_one_not_small_world():
    ring = nx.watts_strogatz_graph(200, 4, 0.0, seed=1)  # p = 0: regular ring lattice
    sw = small_world(ring, niter=2, nrand=2, seed=7)
    assert sw["omega"] == pytest.approx(-1.0, abs=0.25)
    assert not (SMALL_WORLD_WINDOW[0] <= sw["omega"] <= SMALL_WORLD_WINDOW[1])


def test_watts_strogatz_small_rewiring_has_sigma_above_one():
    ws = nx.connected_watts_strogatz_graph(200, 6, 0.1, seed=1)
    sw = small_world(ws, niter=2, nrand=2, seed=7)
    assert sw["sigma"] > 1.5
    assert SMALL_WORLD_WINDOW[0] <= sw["omega"] <= SMALL_WORLD_WINDOW[1]


def test_random_graph_omega_near_plus_one_side():
    er = nx.gnm_random_graph(200, 800, seed=3)
    sw = small_world(er, niter=2, nrand=2, seed=7)
    assert sw["sigma"] == pytest.approx(1.0, abs=0.5)
    assert sw["omega"] > 0.3


def test_same_seed_same_result_and_seeds_differ():
    ws = nx.connected_watts_strogatz_graph(100, 6, 0.1, seed=2)
    a = small_world(ws, niter=2, nrand=2, seed=11)
    b = small_world(ws, niter=2, nrand=2, seed=11)
    c = small_world(ws, niter=2, nrand=2, seed=12)
    assert (a["sigma"], a["omega"]) == (b["sigma"], b["omega"])
    assert (a["sigma"], a["omega"]) != (c["sigma"], c["omega"])


def test_largest_component_used_and_reported():
    g = nx.disjoint_union(nx.connected_watts_strogatz_graph(60, 4, 0.1, seed=1), nx.path_graph(5))
    sw = small_world(g, niter=1, nrand=1, seed=0)
    assert sw["n_nodes"] == 60 and sw["n_edges"] == 120


def test_too_small_or_complete_component_gives_nan():
    for g in (nx.path_graph(3), nx.complete_graph(6)):
        sw = small_world(g, niter=1, nrand=1, seed=0)
        assert math.isnan(sw["sigma"]) and math.isnan(sw["omega"])


def test_disconnected_node_set_raises():
    g = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    with pytest.raises(ValueError, match="connected"):
        small_world(g, nodes=g.nodes)


def test_zero_clustering_gives_nan_with_reason():
    sw = small_world(nx.cycle_graph(20), niter=1, nrand=1, seed=0)
    assert math.isnan(sw["sigma"]) and "zero clustering" in sw["note"]


def test_graph_metrics_reports_small_world_when_requested():
    # 40 single-bead proteins 0.7 nm apart along x in a 28 nm box: first and second
    # neighbours are within 1.5 nm, so the pp graph is a k = 4 ring lattice
    box = np.array([28.0, 15.0, 60.0])
    f = Frame(0, 0.0, box, np.array([[0.7 * i + 0.35, 7.0, 30.0] for i in range(40)], float), np.empty((0, 3)))
    r = graph_metrics([f], np.arange(40), 40, np.empty(0, int), 0, graphs=("pp",), betweenness=False,
                      small_world_params=dict(niter=2, nrand=2, seed=0))
    assert r.series["pp"].n_components == [1]
    assert len(r.series["pp"].sigma) == 1
    assert r.series["pp"].omega[0] < SMALL_WORLD_WINDOW[0]  # lattice-like
    r = graph_metrics([f], np.arange(40), 40, np.empty(0, int), 0, graphs=("pp",), betweenness=False)
    assert r.series["pp"].sigma == []


# ---------------------------------------------------------------- Erdos-Renyi reference for sigma

def test_erdos_renyi_reference_is_seed_reproducible():
    ws = nx.connected_watts_strogatz_graph(100, 6, 0.1, seed=2)
    a = small_world(ws, niter=2, nrand=4, seed=11, reference="erdos_renyi")
    b = small_world(ws, niter=2, nrand=4, seed=11, reference="erdos_renyi")
    c = small_world(ws, niter=2, nrand=4, seed=12, reference="erdos_renyi")
    assert a == b
    assert a["sigma"] != c["sigma"]
    assert a["reference"] == "erdos_renyi"


@pytest.mark.slow
def test_large_erdos_renyi_graph_has_sigma_near_one_in_erdos_renyi_mode():
    er = nx.gnm_random_graph(500, 2500, seed=5)  # mean degree 10, connected
    assert nx.is_connected(er)
    sw = small_world(er, niter=1, nrand=10, seed=3, reference="erdos_renyi")
    assert sw["sigma"] == pytest.approx(1.0, abs=0.15)
    assert sw["n_disconnected_references"] == 0 and sw["note"] == ""


def test_erdos_renyi_mode_changes_sigma_only():
    ws = nx.connected_watts_strogatz_graph(120, 6, 0.1, seed=4)
    dp = small_world(ws, niter=2, nrand=3, seed=9)
    er = small_world(ws, niter=2, nrand=3, seed=9, reference="erdos_renyi")
    assert er["omega"] == dp["omega"]
    assert er["sigma"] != dp["sigma"] and er["sigma"] > 1.5


def test_disconnected_erdos_renyi_references_are_counted_in_note():
    # ring of 40 triangles sharing corners: 80 nodes, 120 edges, mean degree 3,
    # so G(n, m) samples with the same n and m are almost always disconnected
    g = nx.Graph()
    for k in range(40):
        a, b, c = 2 * k, 2 * k + 1, (2 * k + 2) % 80
        g.add_edges_from([(a, b), (b, c), (a, c)])
    sw = small_world(g, niter=1, nrand=10, seed=0, reference="erdos_renyi")
    assert sw["n_disconnected_references"] > 0
    assert f"{sw['n_disconnected_references']} of 10 Erdos-Renyi references were disconnected" in sw["note"]
    assert math.isfinite(sw["sigma"])


def test_unknown_reference_raises():
    with pytest.raises(ValueError, match="reference must be one of"):
        small_world(nx.cycle_graph(5), reference="lattice")
