# Worked example: a multiphase FUS-RGG3 / PLP / RNA droplet

This example runs condensate-graph on a published, openly licensed trajectory:
the multiphase droplet of Fig 4h in

> Yasuda, I., von Bülow, S., Tesei, G., Yamamoto, E., Yasuoka, K. and
> Lindorff-Larsen, K. Coarse-Grained Model of Disordered RNA for Simulations of
> Biomolecular Condensates. J. Chem. Theory Comput. (2025).
> https://doi.org/10.1021/acs.jctc.4c01646

No simulation data is stored in this repository. The scripts download it at
run time into `~/.cache/condensate-graph/fig4h/`.

## The system

CALVADOS-RNA (CALVADOS 2 proteins, two beads per nucleotide for RNA), 293 K,
cubic periodic box of 110.8 nm, 54,694 beads:

| species | chains | beads per chain |
|---|---|---|
| FUS-RGG3 (here "RGG3") | 473 | 34 |
| PLP (FUS prion-like domain) | 164 | 173 |
| polyR40 RNA | 128 | 80 |

The trajectory has 1445 frames. In the paper, RGG3 and RNA form the core of
the droplet and PLP forms a shell around it.

## Data and licences

- Trajectory: Zenodo, https://doi.org/10.5281/zenodo.15674310, file
  `md_data.zip` (0.832 GB, md5 63d59912a024174bf049df06b86daafd), licence
  CC-BY-4.0. Only the `md_data/fig4h/` members are extracted (DCD 0.95 GB,
  PDB 5.6 MB).
- Published radial density profiles used for the sanity check: three small
  `.npy` files from https://github.com/KULL-Centre/_2024_yasuda_RNA
  (`data/view2/{RRP,PLP,RNA}_FUSRGG3-473_PLP-164_polyR40-128.npy`), pinned to
  commit c421eca, licence GPL-3.0. They are downloaded at run time and not
  copied into this repository.

Please cite the article and the Zenodo record if you use these data.

## Running it

```bash
python fetch_data.py          # asks before downloading 0.832 GB; --yes skips the question
python run_analysis.py        # writes output/results.json and output/per_frame.npz
python make_figures.py        # writes four PNG files to output/
```

Measured on a 112-core workstation, CPU only:

| step | time |
|---|---|
| `fetch_data.py` (download, md5 check, extraction) | 1 min 55 s |
| `run_analysis.py --workers 48` (1345 frames) | 26 min (1562 s) |
| `make_figures.py` | 2 s |

Most of the analysis time goes into exact betweenness centrality for every
graph in every frame and into the small-world estimates. With fewer workers
the run time grows roughly in proportion.

## Choices and assumptions

- **Chains.** `build_system(u, "not resname r", "resname r", by="segment")`.
  RNA beads are named `r`; all other beads are protein. Protein chains are
  labelled RGG3 or PLP from their length (34 or 173 beads). The PDB in the
  Zenodo record lacks many CONECT records, so `by="fragment"` would split the
  protein chains into 6698 pieces; the package warns about this.
- **Frames.** Frames 100 onward are analysed, as in the paper's analysis
  scripts.
- **Frame spacing.** Not stated in the data files. **10 ns per frame is
  assumed** (the paper's condensate runs were written every 10 ns). All times
  in the output depend on this assumption.
- **Contacts.** Two chains are in contact if any pair of their beads is within
  1.5 nm (minimum image). With hysteresis, a contact that exists stays on until
  the beads are more than 2.0 nm apart.
- **Small-world.** sigma and omega of the largest component of the combined
  graph, on every 10th analysed frame (135 frames), niter 5, nrand 10, seed 0,
  degree-preserving references.

## Results (frames 100 to 1444)

**Density profiles (sanity check, figure d).** Radial density profiles of
RGG3, PLP and RNA around the droplet centre, 1 nm shells, in mM of chains. I
re-implemented the paper's approach: move the densest 0.1 nm bin along z, y
and x to the box centre, wrap, then histogram distances from the centre.
Agreement with the published curves for r up to 50 nm:

| variant | RGG3 | PLP | RNA |
|---|---|---|---|
| re-centred, frames 100+ (main analysis): relative RMS / core ratio | 0.063 / 1.03 | 0.064 / 0.99 | 0.098 / 1.04 |
| coordinates as stored, all 1445 frames: relative RMS / core ratio | 0.000 / 1.000 | 0.000 / 1.000 | 0.000 / 1.000 |

"Core ratio" is our mean concentration for r < 10 nm divided by the published
one; Pearson r is 0.999 or 1.000 in every case. The second row reproduces the
published profiles exactly. This shows that the Zenodo trajectory is the
paper's processed trajectory: the droplet is already centred in every frame,
and the paper's first 100 frames are probably already removed. Dropping
another 100 frames, as the main analysis does, therefore discards about 1 µs
more than the paper did; the profiles still agree to within 10%.

**Bridging (figure a).**

| | no hysteresis | hysteresis (off at 2.0 nm) |
|---|---|---|
| mean RNA valency B (protein chains contacted) | 13.8 | 14.2 |
| P(n ≥ 4) | 0.71 | 0.71 |
| f_bridge basic (≥ 2 protein chains) | 0.925 | 0.925 |
| f_bridge productive (bridges proteins not in direct contact) | 0.784 | 0.779 |
| persistent, k = 2 | 0.529 | 0.606 |
| persistent, k = 5 | 0.0024 | 0.0026 |

**Graphs and percolation (figure b).** No graph percolates in any frame, as
expected for a finite droplet in a large box. The largest component holds 56%
of the protein chains on average. It grows from about 27% to 60% during the
first 2.5 µs after frame 100 (mean 0.39 for frames 100 to 349, 0.60 for
frames 350 onward), so the droplet is still coarsening at the start of the
analysed window. The bridging fractions hardly change between those two
periods.

**Hubs (figure c).** In the bridged and combined graphs, RGG3 chains have more
than twice the degree of PLP chains (about 40 vs 17), which fits an RNA-RGG3
core. PLP chains have the highest betweenness in every protein-only graph and
fill 7 to 9 of the top 10 betweenness positions per frame. In the bipartite
graph, RNA chains are the hubs by degree (all top 10) and mostly by betweenness
(8 of 10).

**Small-world (combined graph).** sigma = 2.05 ± 0.19 and omega = -0.127 ±
0.022 (mean ± SD over 135 frames), so the largest component is small-world by
the usual omega window of -0.3 to 0.3.

## How to read the results

- **P(n) and f_bridge basic** count, per frame, how many protein chains each
  RNA chain touches. Here almost every RNA chain touches four or more.
- **Productive** bridges are the subset in which at least two of the protein
  partners are not in direct contact with each other, so the RNA adds a link
  the protein network does not already have.
- **Persistent over k frames** means the RNA chain keeps contact with the same
  pair of protein chains in each of the last k frames. With the assumed
  10 ns spacing, k = 2 tests about 10 ns and k = 5 about 40 ns. The fraction
  falls from 0.53 at k = 2 to 0.002 at k = 5. RNA chains in this droplet keep
  many partners at any moment but exchange them within tens of nanoseconds of
  coarse-grained time. Note that coarse-grained dynamics are faster than real
  dynamics, and the frame spacing is assumed.
- **Hysteresis** only removes flicker between frames. It raises k = 2
  persistence from 0.53 to 0.61 but does not change k = 5, because at 10 ns
  spacing most partner changes are real departures rather than brief breaks of
  a contact.
- **Largest component fraction** below 1 with no percolation means the
  network is one droplet plus chains in the dilute phase, not a
  system-spanning network.

## Output files

- `output/results.json`: settings, assumptions and time averages.
- `output/per_frame.npz`: per-frame series (bridging, graph metrics, hub
  statistics by species, small-world) and all profile variants.
- `output/fig_a_bridging.png`, `fig_b_graphs.png`, `fig_c_hubs.png`,
  `fig_d_profiles.png`.

The `output/` folder is not tracked by git.
