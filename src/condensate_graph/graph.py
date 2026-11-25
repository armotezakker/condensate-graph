"""Chain-level contact networks and periodic percolation.

Per frame, four undirected graphs are built from the chain-level contacts of
:mod:`condensate_graph.contacts` (all distances in nm, minimum-image
periodic):

* ``"pp"``: protein chains, linked by direct protein-protein contacts;
* ``"bridged"``: protein chains, linked if at least one RNA chain contacts
  both;
* ``"combined"``: the union of ``"pp"`` and ``"bridged"``;
* ``"bipartite"``: protein and RNA chains as nodes, protein-RNA contacts as
  edges.

Nodes are ``("P", i)`` for protein chain i and ``("R", j)`` for RNA chain j.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .contacts import DEFAULT_CUTOFF_NM, ContactSettings, iter_frame_contacts

GRAPH_TYPES = ("pp", "bridged", "combined", "bipartite")
AXES = ("x", "y", "z")

__all__ = [
    "GRAPH_TYPES",
    "AXES",
    "GraphSeries",
    "GraphResult",
    "build_graphs",
    "largest_protein_component",
    "top_hubs",
    "spanning_axes",
    "graph_metrics",
    "run_graph_metrics",
]


# --------------------------------------------------------------------------- graphs

def build_graphs(protein_rna_contact, protein_protein_contact, graphs=GRAPH_TYPES) -> dict:
    """Build chain-level graphs for one frame.

    Parameters
    ----------
    protein_rna_contact : array_like of bool, shape (n_rna, n_protein)
        Protein-RNA chain contacts.
    protein_protein_contact : array_like of bool, shape (n_protein, n_protein)
        Symmetric protein-protein chain contacts.
    graphs : sequence of str, optional
        Which of ``"pp"``, ``"bridged"``, ``"combined"`` and ``"bipartite"``
        to build. Default: all four.

    Returns
    -------
    dict of str to networkx.Graph
        Every protein chain is a node of every graph, also when isolated;
        RNA chains are nodes of ``"bipartite"`` only.
    """
    pr = np.asarray(protein_rna_contact, dtype=bool)
    pp = np.asarray(protein_protein_contact, dtype=bool)
    n_rna, n_protein = pr.shape
    unknown = set(graphs) - set(GRAPH_TYPES)
    if unknown:
        raise ValueError(f"unknown graph types {sorted(unknown)}; choose from {GRAPH_TYPES}")
    proteins = [("P", i) for i in range(n_protein)]
    shared = (pr.T.astype(np.int64) @ pr.astype(np.int64)) > 0
    np.fill_diagonal(shared, False)
    adjacency = {"pp": pp, "bridged": shared, "combined": pp | shared}
    out = {}
    for name in graphs:
        g = nx.Graph()
        g.add_nodes_from(proteins, kind="protein")
        if name == "bipartite":
            g.add_nodes_from((("R", j) for j in range(n_rna)), kind="rna")
            r, p = np.nonzero(pr)
            g.add_edges_from((("R", int(a)), ("P", int(b))) for a, b in zip(r, p))
        else:
            i, j = np.nonzero(np.triu(adjacency[name], 1))
            g.add_edges_from((("P", int(a)), ("P", int(b))) for a, b in zip(i, j))
        out[name] = g
    return out


def largest_protein_component(graph) -> set:
    """The connected component holding the most protein chains.

    Ties are broken by total component size, then by the smallest protein
    index, so the result is deterministic.

    Parameters
    ----------
    graph : networkx.Graph
        A graph from :func:`build_graphs`.

    Returns
    -------
    set
        Nodes of that component.
    """
    def key(comp):
        prot = [n[1] for n in comp if n[0] == "P"]
        return (len(prot), len(comp), -min(prot) if prot else 0)

    return max(nx.connected_components(graph), key=key)


def top_hubs(graph, n=5, by="degree") -> list:
    """Most connected nodes of a graph.

    Parameters
    ----------
    graph : networkx.Graph
    n : int, optional
        Number of hubs to return (default 5).
    by : {"degree", "betweenness"}, optional
        Rank by node degree or by normalised betweenness centrality
        (:func:`networkx.betweenness_centrality`).

    Returns
    -------
    list of (node, value)
        Highest first; ties broken by node label.
    """
    if by == "degree":
        scores = dict(graph.degree())
    elif by == "betweenness":
        scores = nx.betweenness_centrality(graph, normalized=True)
    else:
        raise ValueError(f"by must be 'degree' or 'betweenness', got {by!r}")
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


# --------------------------------------------------------------------------- percolation

def spanning_axes(edges, start) -> np.ndarray:
    """Periodic spanning test for the component that contains ``start``.

    In plain words: every chain is made whole and given an image label, a
    vector of three integers saying which periodic copy of the chain we are
    holding. The search starts from ``start`` with label (0, 0, 0) and walks
    along contacts. Each contact records which copy of the neighbour touches
    the current chain (its image shift), so the neighbour gets the current
    label minus that shift. If the walk reaches a chain that already has a
    label, and the new label differs from it, then following contacts around
    that loop arrives at a different periodic copy of the same chain: the
    network wraps around the box and connects to its own image. It percolates
    along every axis on which the two labels differ. A large but compact
    cluster, or an open chain of contacts that merely crosses the boundary,
    always arrives back at the same copy and does not percolate.

    Parameters
    ----------
    edges : iterable of (u, v, shift)
        Undirected contacts. ``shift`` is a length-3 integer vector: the copy
        of ``v`` that touches ``u`` lies at (whole ``v``) - shift * L, with L
        the box edge lengths. Several edges between the same nodes with
        different shifts are allowed.
    start : hashable
        A node of the component to test.

    Returns
    -------
    numpy.ndarray of bool, shape (3,)
        Whether the component percolates along x, y and z.
    """
    nbrs = {}
    for u, v, s in edges:
        s = np.asarray(s, dtype=np.int64)
        nbrs.setdefault(u, []).append((v, s))
        nbrs.setdefault(v, []).append((u, -s))
    label = {start: np.zeros(3, dtype=np.int64)}
    spans = np.zeros(3, dtype=bool)
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v, s in nbrs.get(u, ()):
            expected = label[u] - s
            if v not in label:
                label[v] = expected
                queue.append(v)
            else:
                spans |= expected != label[v]
        if spans.all():
            break
    return spans


def _shift_edges(fc, name):
    pp = [(("P", int(a)), ("P", int(b)), row[2:]) for row in fc.protein_protein_shifts for a, b in [row[:2]]]
    pr = [(("R", int(r)), ("P", int(p)), row[2:]) for row in fc.protein_rna_shifts for r, p in [row[:2]]]
    if name == "pp":
        return pp
    if name in ("bridged", "bipartite"):
        return pr  # a bridged path between proteins runs protein - RNA - protein
    return pp + pr


# --------------------------------------------------------------------------- results

@dataclass
class GraphSeries:
    """Per-frame metrics of one graph type.

    Attributes
    ----------
    n_components : list of int
        Number of connected components (isolated chains count as components).
    largest_fraction : list of float
        Protein chains in the largest component (see
        :func:`largest_protein_component`) divided by the number of protein
        chains.
    percolates : list of numpy.ndarray of bool, shape (3,)
        Whether the largest component percolates along x, y, z.
    degree_histogram : list of numpy.ndarray of int
        ``degree_histogram[t][d]`` is the number of nodes with degree d.
    top_degree, top_betweenness : list of list of (node, value)
        Top hubs per frame (betweenness only if requested).
    """

    n_components: list = field(default_factory=list)
    largest_fraction: list = field(default_factory=list)
    percolates: list = field(default_factory=list)
    degree_histogram: list = field(default_factory=list)
    top_degree: list = field(default_factory=list)
    top_betweenness: list = field(default_factory=list)


@dataclass
class GraphResult:
    """Graph metrics for every analysed frame.

    Attributes
    ----------
    frame_index : numpy.ndarray of int, shape (n_frames,)
    series : dict of str to GraphSeries
        One entry per graph type.
    settings : condensate_graph.contacts.ContactSettings
        Cutoffs in nm used for the contacts.
    """

    frame_index: np.ndarray
    series: dict
    settings: ContactSettings

    def summary(self) -> dict:
        """Time averages per graph type.

        Returns
        -------
        dict of str to dict
            For each graph type: ``mean_n_components``,
            ``mean_largest_fraction`` and ``percolating_fraction`` (fraction of
            frames percolating along x, y, z, as an array of 3).
        """
        out = {}
        for name, s in self.series.items():
            out[name] = {
                "mean_n_components": float(np.mean(s.n_components)),
                "mean_largest_fraction": float(np.mean(s.largest_fraction)),
                "percolating_fraction": np.mean(np.array(s.percolates, dtype=float), axis=0),
            }
        return out


# --------------------------------------------------------------------------- drivers

def graph_metrics(frames, protein_chain_of_bead, n_protein, rna_chain_of_bead, n_rna, *,
                  cutoff=DEFAULT_CUTOFF_NM, cutoff_off=None,
                  protein_protein_cutoff=None, protein_protein_cutoff_off=None,
                  graphs=GRAPH_TYPES, top_n=5, betweenness=True) -> GraphResult:
    """Graph metrics and periodic percolation over a sequence of frames.

    Parameters
    ----------
    frames : iterable
        Objects with ``box`` (shape (3,), nm), ``protein_positions`` and
        ``rna_positions`` (shape (n, 3), nm) and optionally ``index``, in time
        order (needed for hysteresis).
    protein_chain_of_bead, rna_chain_of_bead : array_like of int
        Chain index of each protein / RNA bead.
    n_protein, n_rna : int
        Number of protein / RNA chains.
    cutoff, cutoff_off, protein_protein_cutoff, protein_protein_cutoff_off : float or None
        Contact cutoffs and hysteresis release distances in nm; see
        :meth:`condensate_graph.contacts.ContactSettings.create`.
    graphs : sequence of str, optional
        Graph types to analyse (default: all four).
    top_n : int, optional
        Number of hubs to report per frame (default 5).
    betweenness : bool, optional
        Also rank hubs by betweenness centrality (default True; slower).

    Returns
    -------
    GraphResult
    """
    settings = ContactSettings.create(cutoff, cutoff_off, protein_protein_cutoff, protein_protein_cutoff_off)
    series = {name: GraphSeries() for name in graphs}
    index = []
    for fc in iter_frame_contacts(frames, protein_chain_of_bead, n_protein, rna_chain_of_bead, n_rna,
                                  settings, image_shifts=True):
        index.append(fc.index)
        built = build_graphs(fc.protein_rna, fc.protein_protein, graphs)
        for name, g in built.items():
            s = series[name]
            comp = largest_protein_component(g)
            s.n_components.append(nx.number_connected_components(g))
            s.largest_fraction.append(sum(1 for n in comp if n[0] == "P") / n_protein)
            start = min(n for n in comp if n[0] == "P")
            s.percolates.append(spanning_axes(_shift_edges(fc, name), start))
            s.degree_histogram.append(np.bincount([d for _, d in g.degree()]))
            s.top_degree.append(top_hubs(g, top_n, "degree"))
            s.top_betweenness.append(top_hubs(g, top_n, "betweenness") if betweenness else [])
    return GraphResult(frame_index=np.array(index, dtype=np.int64), series=series, settings=settings)


def run_graph_metrics(system, start=None, stop=None, step=None, **kwargs) -> GraphResult:
    """Graph metrics for a :class:`condensate_graph.system.ChainSystem`.

    Parameters
    ----------
    system : condensate_graph.system.ChainSystem
    start, stop, step : int or None, optional
        Frame slice over the trajectory.
    **kwargs
        Cutoffs (nm), ``graphs``, ``top_n`` and ``betweenness``, passed to
        :func:`graph_metrics`.

    Returns
    -------
    GraphResult
    """
    return graph_metrics(
        system.iter_frames(start, stop, step),
        system.protein.chain_of_atom, system.protein.n_chains,
        system.rna.chain_of_atom, system.rna.n_chains,
        **kwargs,
    )
