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
python fetch_data.py --clean  # optional: delete the downloaded files (about 1.8 GB) again
```

Measured on a 112-core workstation, CPU only:

| step | time |
|---|---|
| `fetch_data.py` (download, md5 check, extraction) | 1 min 55 s |
| `run_analysis.py --workers 48` (all 1445 frames) | 28 min (1662 s) |
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
- **Frames.** Every stored frame is analysed. The deposited DCD appears to
  be the paper's processed trajectory: the droplet is already centred in
  every frame, and density profiles from all stored frames reproduce the
  published ones exactly (below). The paper's own analysis dropped the first
  100 frames of the raw run before writing this file, so they are not
  dropped again here.
- **Plateau window.** Network metrics are reported on the window where the
  droplet has stopped growing. Rule: take a centred rolling mean over 50
  frames of the largest-component fraction of the combined graph; take the
  final value as the mean of the last 10% of frames; the window starts at
  the first frame where the rolling mean reaches 95% of the final value and
  runs to the end. Here it starts at frame 349 (3.49 µs) and covers 1096
  frames; the rolling mean falls below the threshold again in only 0.5% of
  them. The start moves by less than 30 frames (frames 334 to 364) when the
  rolling window is 100 frames, the final value is taken from the last 20%
  or 50%, or the tolerance is 10%. A stricter rule, that the rolling mean
  never falls below the threshold again, would start at frame 1398 because
  of one short dip; it is reported in `results.json` but not used.
- **Frame spacing.** Not stated in the data files. **10 ns per frame is
  assumed** (the paper's condensate runs were written every 10 ns). All times
  in the output depend on this assumption.
- **Contacts.** Two chains are in contact if any pair of their beads is within
  1.5 nm (minimum image). With hysteresis, a contact that exists stays on until
  the beads are more than 2.0 nm apart.
- **Small-world.** sigma and omega of the largest component of the combined
  graph, on every 10th analysed frame (135 frames), niter 5, nrand 10, seed 0,
  degree-preserving references.

## Results

**Density profiles (sanity check, figure d), all 1445 stored frames.**
Radial density profiles of RGG3, PLP and RNA around the box centre, 1 nm
shells, in mM of chains, against the published curves (r up to 50 nm):

| variant | RGG3 | PLP | RNA |
|---|---|---|---|
| coordinates as stored: max absolute difference | 0.003 mM | 0.0001 mM | 0.0005 mM |
| coordinates as stored: relative RMS / core ratio | 0.000 / 1.000 | 0.000 / 1.000 | 0.000 / 1.000 |
| centred again: relative RMS / core ratio | 0.036 / 0.996 | 0.020 / 0.993 | 0.061 / 1.013 |

"Core ratio" is our mean concentration for r < 10 nm divided by the
published one. With the coordinates as stored, the profiles match the
published ones to the precision of the stored single-precision coordinates.
"Centred again" applies the paper's centring (move the densest 0.1 nm bin
along z, y and x to the box centre) a second time, re-implemented here; it
moves the droplet by a few nm per frame and blurs the profiles slightly.

The network results below are for the plateau window, frames 349 to 1444
(3.49 to 14.44 µs at the assumed 10 ns per frame). `results.json` also has
every number for all frames.

**Bridging (figure a).**

| | no hysteresis | hysteresis (off at 2.0 nm) |
|---|---|---|
| mean RNA valency B (protein chains contacted) | 14.05 | 14.48 |
| P(n ≥ 4) | 0.711 | 0.712 |
| f_bridge basic (≥ 2 protein chains) | 0.924 | 0.924 |
| f_bridge productive (bridges proteins not in direct contact) | 0.783 | 0.778 |
| persistent, k = 2 | 0.527 | 0.605 |
| persistent, k = 5 | 0.0025 | 0.0026 |

Over all frames the bridging numbers differ by less than 0.01 (B 13.68
without hysteresis), so they hardly depend on the growth phase.

**Graphs and percolation (figure b).** No graph percolates in any frame, as
expected for a finite droplet in a large box. In the window the largest
component holds 60% of the protein chains (pp 0.600, bridged 0.598,
combined 0.602, bipartite 0.598); over all frames it is 54%, because it
grows from about 30% during the first 3.5 µs.

**Hubs (figure c).** In the bridged and combined graphs, RGG3 chains have
more than twice the degree of PLP chains (41 vs 18), which fits an RNA-RGG3
core. PLP chains have the highest betweenness in every protein-only graph:
they fill 9.1 (pp), 7.6 (bridged) and 7.8 (combined) of the top 10
betweenness positions per frame on average. In the bipartite graph, RNA
chains are all of the top 10 by degree and 7.9 of the top 10 by
betweenness.

**Small-world (combined graph, every 10th frame in the window, 110 frames).**
sigma = 2.12 ± 0.10 and omega = -0.134 ± 0.015 (mean ± SD), so the largest
component is small-world by the usual omega window of -0.3 to 0.3.

## How to read the results

- **P(n) and f_bridge basic** count, per frame, how many protein chains each
  RNA chain touches. Here almost every RNA chain touches four or more.
- **Productive** bridges are the subset in which at least two of the protein
  partners are not in direct contact with each other, so the RNA adds a link
  the protein network does not already have.
- **Persistent over k frames** means the RNA chain keeps contact with the same
  pair of protein chains in each of the last k frames. With the assumed
  10 ns spacing, k = 2 tests about 10 ns and k = 5 about 40 ns. The fraction
  falls from 0.53 at k = 2 to 0.0025 at k = 5. RNA chains in this droplet keep
  many partners at any moment but exchange them within tens of nanoseconds of
  coarse-grained time. Note that coarse-grained dynamics are faster than real
  dynamics, and the frame spacing is assumed.
- **Hysteresis** only removes flicker between frames. It raises k = 2
  persistence from 0.53 to 0.60 but does not change k = 5, because at 10 ns
  spacing most partner changes are real departures rather than brief breaks of
  a contact.
- **Largest component fraction** below 1 with no percolation means the
  network is one droplet plus chains in the dilute phase, not a
  system-spanning network.
- **Plateau window.** Averages over a period in which the droplet is still
  growing mix two states. Check figure b before trusting any average, and
  change the rule constants at the top of `run_analysis.py` if your system
  needs a different one.

## Output files

- `output/results.json`: settings, assumptions, the plateau rule and
  window, and time averages for the window and for all frames.
- `output/per_frame.npz`: per-frame series (bridging, graph metrics, hub
  statistics by species, small-world) and all profile variants.
- `output/fig_a_bridging.png`, `fig_b_graphs.png`, `fig_c_hubs.png`,
  `fig_d_profiles.png`.

The `output/` folder is not tracked by git.
