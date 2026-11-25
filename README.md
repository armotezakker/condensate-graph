# condensate-graph

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24542827.svg)](https://doi.org/10.5281/zenodo.24542827)
[![tests](https://github.com/armotezakker/condensate-graph/actions/workflows/tests.yml/badge.svg)](https://github.com/armotezakker/condensate-graph/actions/workflows/tests.yml)

condensate-graph turns coarse-grained simulations of protein-RNA condensates
into chain-level contact networks and measures how RNA holds them together.
It reads any trajectory that MDAnalysis can read (CALVADOS, HPS, Mpipi,
LAMMPS, GROMACS and others), splits the beads into protein and RNA chains
using selections you give, finds contacts between chains with periodic
minimum-image distances, and reports RNA valency, bridging, persistence,
four kinds of chain graph, periodic percolation, hub chains and small-world
coefficients.

## Why periodic boundaries matter

Condensate simulations run in periodic boxes, and a slab or droplet often
crosses the box edge. If contacts are searched without the box, every pair of
chains that touches across the boundary is silently missed: bridging,
cluster sizes and percolation all come out too low, with no error.
condensate-graph therefore refuses to run without a box. It takes the box
from the trajectory or from an explicit `box=` argument (in nm) and raises
`BoxError` if neither is available. Contacts, chain centres and percolation
are all computed with periodic images.

## Install

Python 3.10 or newer.

```bash
pip install "git+https://github.com/armotezakker/condensate-graph.git"
```

For development:

```bash
git clone https://github.com/armotezakker/condensate-graph.git
cd condensate-graph
pip install -e ".[dev]"
pytest
```

## Quickstart

```python
from condensate_graph import load_system, run_bridging, run_graph_metrics

# CALVADOS-style files: RNA beads are named "r", everything else is protein
system = load_system("top.pdb", "traj.dcd",
                     protein_selection="not resname r", rna_selection="resname r",
                     by="segment")             # one chain per segment / chain ID

bridging = run_bridging(system, cutoff=1.5, cutoff_off=2.0, min_frames=(1, 2, 5))
means = bridging.means()
print("B =", means["B"], "basic =", means["f_bridge_basic"],
      "productive =", means["f_bridge_productive"])

graphs = run_graph_metrics(system, cutoff=1.5, top_n=5)
for name, s in graphs.summary().items():
    print(name, s["mean_largest_fraction"], s["percolating_fraction"])
```

Chains are identified in exactly one of three ways, chosen explicitly:
`by="segment"` (one chain per segment or chain ID), `by="fragment"` (one chain
per bonded fragment; needs bonds in the topology) or
`chain_lengths={"protein": [...], "rna": [...]}` (bead counts in file order).
Nothing is guessed, and the package warns when the result looks wrong, for
example a single chain where there should be many, or bonded fragments that
look broken because bonds are missing from the topology.

## What is measured

All lengths are in nm. "Contact" between two chains means that at least one
pair of their beads is within the cutoff (default 1.5 nm), using the
minimum-image distance. Time averages skip frames where a quantity is
undefined.

**RNA valency.** For each RNA chain in a frame, the number of distinct protein
chains it contacts (a count, no unit). **B** is the mean valency over RNA
chains and frames.

**P(n).** The fraction of RNA chains with valency n, for n = 0, 1, 2, 3 and
n ≥ 4. Sums to 1 in every frame.

**Basic bridge.** An RNA chain that contacts two or more protein chains.
f_bridge_basic is the fraction of RNA chains that are basic bridges.

**Productive bridge.** An RNA chain that contacts at least one pair of protein
chains that are not in direct contact with each other. It adds a link the
protein network does not already have. Direct protein-protein contact can use
its own cutoff (`protein_protein_cutoff`).

**Persistent bridge over k frames.** An RNA chain that contacted the same pair
of protein chains in each of the last k frames. It is undefined for the first
k - 1 frames. k = 1 equals the basic bridge. The time this covers is
(k - 1) times the frame spacing, so persistence depends on how often frames
were saved.

**Hysteresis.** Optional (`cutoff_off`). A contact switches on when two chains
come within `cutoff` and stays on until they are more than `cutoff_off` apart.
This removes flicker from chains that sit near the cutoff. It needs frames in
time order.

**Region filter.** Optional (`region=(z_min, z_max)`). Only RNA chains whose
centre of mass (periodic-aware) lies in that z range are counted, for example
the dense phase of a slab.

**The four graph types.** Nodes are chains.
- `pp`: protein chains; an edge for each direct protein-protein contact.
- `bridged`: protein chains; an edge when two proteins share at least one RNA
  partner.
- `combined`: protein chains; the union of `pp` and `bridged`.
- `bipartite`: protein and RNA chains; an edge for each protein-RNA contact.

**Largest component.** The connected component holding the most protein
chains. The reported fraction is the number of protein chains in it divided
by all protein chains.

**Periodic percolation.** Whether the largest component wraps around the
periodic box along x, y or z, so that it connects to its own periodic image.
Chains are made whole and every contact carries the image shift it was found
across; the component percolates along an axis if some closed path through
it has a non-zero net shift along that axis. A large compact droplet that
merely crosses the box edge does not count.

**Hubs.** Chains ranked by degree (number of neighbours in the graph) or by
betweenness centrality (the fraction of shortest paths between other chains
that pass through the chain, from 0 to 1, networkx normalisation).

**Small-world coefficients sigma and omega.** Computed for the largest
connected component and compared with random reference graphs, so they are
stochastic; the seed is explicit and the same seed gives the same result.
- sigma = (C / C_rand) / (L / L_rand), with C the average clustering
  coefficient and L the average shortest path length. sigma > 1 means more
  clustered than random at a similar path length; a regular lattice also
  gives sigma > 1.
- omega = L_rand / L - C / C_latt, near 0 for small-world, near -1 for a
  lattice and near +1 for a random graph. -0.3 to 0.3 is the usual
  small-world window.
- `reference="degree_preserving"` (default): C_rand and L_rand come from
  random graphs with the same degree of every node (networkx `sigma`).
- `reference="erdos_renyi"`: C_rand and L_rand come from Erdős-Rényi G(n, m)
  graphs with the same numbers of nodes and edges; disconnected samples use
  their largest component and are counted in the result. omega always uses
  the networkx references.

## Validation

- **Unit tests** on small synthetic systems with answers computed by hand,
  including contacts, bridges and percolation that exist only across the
  periodic boundary.
- **Published networks.** On 28 condensate systems from the Joseph lab
  (LCD_Network), graphs built with the published -5 kBT energy rule give
  exactly the published edge counts in every frame. omega matches the
  published values to within 0.001 in all systems; sigma matches to within
  about 12% with Erdős-Rényi references. See
  [docs/validation.md](docs/validation.md).
- **Worked example on a published trajectory.** The multiphase FUS-RGG3 /
  PLP / RNA droplet of Yasuda et al. (JCTC 2025, Fig 4h), downloaded at run
  time from Zenodo. The radial density profiles computed here reproduce the
  published curves exactly. See
  [examples/fig4h_multiphase](examples/fig4h_multiphase/README.md).

## Limitations

- Orthorhombic (rectangular) boxes only; triclinic boxes raise an error.
- Contact definitions depend on the cutoffs. Results should be checked for
  sensitivity to `cutoff`, `cutoff_off` and `protein_protein_cutoff`.
- Persistence depends on the frame spacing: k frames cover (k - 1) times the
  saving interval, and coarse-grained dynamics are faster than real ones.
- The cutoff, and a hysteresis release distance, must be at most half the
  shortest box edge.
- sigma and omega are stochastic and slow on large graphs; compute them on a
  subsample of frames.

## Citation

If you use condensate-graph, please cite it. The software is archived on
Zenodo: DOI [10.5281/zenodo.24542827](https://doi.org/10.5281/zenodo.24542827)
covers all versions, and version 0.1.0 has its own DOI,
[10.5281/zenodo.24542828](https://doi.org/10.5281/zenodo.24542828). The full
metadata are in [CITATION.cff](CITATION.cff) (GitHub shows a "Cite this
repository" button).
If you use the worked example, please also cite the article and data it is
based on, listed in its README.

## License

MIT; see [LICENSE](LICENSE).
