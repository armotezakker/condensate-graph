# Validation of small_world against published condensate networks

This page compares `condensate_graph.small_world` with published small-world
coefficients for 28 condensate systems (8 LCD systems and 20 YS sequence
variants, each at 0.90 Tc and 0.95 Tc, 216 chains per system). No data from
the source repository is included in this package; only summary numbers are
reported here.

Source of the published values:

- LCD_Network repository, https://github.com/josephresearch/LCD_Network
  (files `data/Figure2/<group>/<Tc>/<system>/SWcoeff_5kbT.dat` and
  `contact_energies.dat`).
- The associated article: Soft Matter (2025), https://doi.org/10.1039/D5SM00740B

## How the graphs were built

Each molecule is a node. Two molecules are joined by an edge when their
intermolecular interaction energy is below -5 kBT, the threshold stated in
the repository README and in the article ("All results and figures in this
work use the threshold 5kBT"). I built the graphs from the per-pair energies
in `contact_energies.dat` (kcal/mol), with an edge where E < -5 kB T and
kB = 0.0019872043 kcal/(mol K).

The published files give, for every frame, sigma, omega, the number of edges
of the whole graph and the number of edges of its largest connected
component. The whole graph is often disconnected (for example 214 of 216
nodes in the largest component), and sigma and omega are only defined for a
connected graph, so I computed both on the largest component. This choice is
inferred from the data; the sources do not state it.

With the temperatures below, my edge counts match the published counts
exactly for both the whole graph and the largest component, in all 1422
frames of all 28 systems.

## Temperatures

The threshold depends on T, and T is not given in the data files.

- **Documented (4 systems).** The simulation inputs in the repository give
  T for FUS-LCD (308.7 K at 0.90 Tc, 325.85 K at 0.95 Tc) and S4Yx30
  (283 K and 299 K).
- **Inferred (24 systems).** For each system I found the range of threshold
  energies E* for which the number of pairs with E < E* equals the published
  whole-graph edge count in every frame. That range is set by the sorted
  pair energies of each frame and intersected over all frames; I took
  T = -E*/(5 kB) at its midpoint. Every range is narrower than 0.15 K.

The whole-graph count is matched by construction for the inferred systems, so
it is not an independent test there. Three things are: a single T reproduces
every frame of a system; the largest-component edge counts (not used in the
fit) also match in every frame; and all 18 inferred YS temperatures lie
within 0.04 K of a whole number (186, 280, 895 K, for example), which is what
one expects if the inputs set T in whole kelvin. For the 4
documented systems the documented T lies inside the consistent range.

| System | T (K) | Source | Range consistent with edge counts (K) |
|---|---|---|---|
| LCD/0.90Tc/A1-LCD | 307.81 | inferred | 307.79 to 307.82 |
| LCD/0.90Tc/EWSR1-LCD | 294.30 | inferred | 294.29 to 294.31 |
| LCD/0.90Tc/FUS-LCD | 308.70 | documented | 308.69 to 308.70 |
| LCD/0.90Tc/TIA1-LCD | 333.00 | inferred | 332.98 to 333.03 |
| LCD/0.95Tc/A1-LCD | 324.87 | inferred | 324.84 to 324.90 |
| LCD/0.95Tc/EWSR1-LCD | 310.64 | inferred | 310.63 to 310.65 |
| LCD/0.95Tc/FUS-LCD | 325.85 | documented | 325.85 to 325.86 |
| LCD/0.95Tc/TIA1-LCD | 351.53 | inferred | 351.50 to 351.56 |
| YS_VAR/0.90Tc/S120Y30 | 186.00 | inferred | 185.98 to 186.02 |
| YS_VAR/0.90Tc/S12Y3 | 293.01 | inferred | 292.99 to 293.02 |
| YS_VAR/0.90Tc/S24Y6x5 | 269.98 | inferred | 269.95 to 270.01 |
| YS_VAR/0.90Tc/S4Yx30 | 283.00 | documented | 283.00 to 283.01 |
| YS_VAR/0.90Tc/S9Yx15 | 194.00 | inferred | 193.99 to 194.01 |
| YS_VAR/0.90Tc/SSYx50 | 390.99 | inferred | 390.97 to 391.00 |
| YS_VAR/0.90Tc/Sc_S4Yx30 | 280.00 | inferred | 279.99 to 280.02 |
| YS_VAR/0.90Tc/YSx75 | 504.99 | inferred | 504.99 to 505.00 |
| YS_VAR/0.90Tc/YYSx50 | 647.98 | inferred | 647.97 to 648.00 |
| YS_VAR/0.90Tc/Yx150 | 895.00 | inferred | 894.99 to 895.00 |
| YS_VAR/0.95Tc/S120Y30 | 197.00 | inferred | 196.99 to 197.01 |
| YS_VAR/0.95Tc/S12Y3 | 309.00 | inferred | 308.99 to 309.00 |
| YS_VAR/0.95Tc/S24Y6x5 | 285.00 | inferred | 284.96 to 285.04 |
| YS_VAR/0.95Tc/S4Yx30 | 299.00 | documented | 298.97 to 299.02 |
| YS_VAR/0.95Tc/S9Yx15 | 204.99 | inferred | 204.97 to 205.00 |
| YS_VAR/0.95Tc/SSYx50 | 413.02 | inferred | 412.98 to 413.06 |
| YS_VAR/0.95Tc/Sc_S4Yx30 | 294.99 | inferred | 294.96 to 295.03 |
| YS_VAR/0.95Tc/YSx75 | 533.00 | inferred | 532.95 to 533.04 |
| YS_VAR/0.95Tc/YYSx50 | 683.96 | inferred | 683.89 to 684.03 |
| YS_VAR/0.95Tc/Yx150 | 944.96 | inferred | 944.88 to 945.03 |

## Settings

niter = 5 and nrand = 10 (the networkx defaults for omega), 5 seeds (0 to 4)
per frame. The published niter, nrand and seeds are not stated. Numbers below
are means over all frames and seeds. "Within seed range" means the published
mean lies between the lowest and highest of my 5 per-seed means.

## omega

omega uses the same references in both modes, and the values are identical
between modes for every frame and seed.

| System | Theirs | Ours | Difference |
|---|---|---|---|
| LCD/0.90Tc/A1-LCD | -0.2572 | -0.2574 | -0.0002 |
| LCD/0.90Tc/EWSR1-LCD | -0.1926 | -0.1929 | -0.0003 |
| LCD/0.90Tc/FUS-LCD | -0.2261 | -0.2256 | +0.0005 |
| LCD/0.90Tc/TIA1-LCD | -0.2984 | -0.2985 | -0.0002 |
| LCD/0.95Tc/A1-LCD | -0.3190 | -0.3185 | +0.0005 |
| LCD/0.95Tc/EWSR1-LCD | -0.2226 | -0.2226 | -0.0000 |
| LCD/0.95Tc/FUS-LCD | -0.2593 | -0.2594 | -0.0001 |
| LCD/0.95Tc/TIA1-LCD | -0.3535 | -0.3538 | -0.0003 |
| YS_VAR/0.90Tc/S120Y30 | -0.3254 | -0.3254 | +0.0000 |
| YS_VAR/0.90Tc/S12Y3 | -0.2531 | -0.2536 | -0.0005 |
| YS_VAR/0.90Tc/S24Y6x5 | -0.3946 | -0.3947 | -0.0001 |
| YS_VAR/0.90Tc/S4Yx30 | -0.2329 | -0.2328 | +0.0001 |
| YS_VAR/0.90Tc/S9Yx15 | -0.2379 | -0.2378 | +0.0001 |
| YS_VAR/0.90Tc/SSYx50 | -0.2325 | -0.2324 | +0.0001 |
| YS_VAR/0.90Tc/Sc_S4Yx30 | -0.2955 | -0.2954 | +0.0000 |
| YS_VAR/0.90Tc/YSx75 | -0.2358 | -0.2356 | +0.0002 |
| YS_VAR/0.90Tc/YYSx50 | -0.2281 | -0.2280 | +0.0001 |
| YS_VAR/0.90Tc/Yx150 | -0.2140 | -0.2140 | +0.0000 |
| YS_VAR/0.95Tc/S120Y30 | -0.2526 | -0.2527 | -0.0001 |
| YS_VAR/0.95Tc/S12Y3 | -0.2910 | -0.2904 | +0.0006 |
| YS_VAR/0.95Tc/S24Y6x5 | -0.4406 | -0.4404 | +0.0002 |
| YS_VAR/0.95Tc/S4Yx30 | -0.2639 | -0.2638 | +0.0002 |
| YS_VAR/0.95Tc/S9Yx15 | -0.2868 | -0.2860 | +0.0008 |
| YS_VAR/0.95Tc/SSYx50 | -0.2533 | -0.2532 | +0.0001 |
| YS_VAR/0.95Tc/Sc_S4Yx30 | -0.3542 | -0.3544 | -0.0002 |
| YS_VAR/0.95Tc/YSx75 | -0.2722 | -0.2722 | -0.0000 |
| YS_VAR/0.95Tc/YYSx50 | -0.2643 | -0.2646 | -0.0003 |
| YS_VAR/0.95Tc/Yx150 | -0.2447 | -0.2449 | -0.0002 |

Every difference is at most 0.0008. The published mean lies inside my seed
range for 16 of 28 systems and within 4.6 seed standard deviations for all of
them, and the per-frame correlation is 0.99 to 1.00 in every system.

## sigma

| System | Theirs | Ours, degree preserving | Ours, Erdos-Renyi | Erdos-Renyi difference | Within Erdos-Renyi seed range |
|---|---|---|---|---|---|
| LCD/0.90Tc/A1-LCD | 8.130 | 7.256 | 7.770 | -0.360 (-4.4%) | no |
| LCD/0.90Tc/EWSR1-LCD | 5.800 | 5.189 | 5.940 | +0.141 (+2.4%) | no |
| LCD/0.90Tc/FUS-LCD | 7.377 | 6.652 | 7.177 | -0.200 (-2.7%) | yes |
| LCD/0.90Tc/TIA1-LCD | 9.973 | 8.956 | 9.157 | -0.815 (-8.2%) | no |
| LCD/0.95Tc/A1-LCD | 8.693 | 7.937 | 8.089 | -0.604 (-6.9%) | no |
| LCD/0.95Tc/EWSR1-LCD | 6.883 | 6.069 | 7.000 | +0.117 (+1.7%) | yes |
| LCD/0.95Tc/FUS-LCD | 8.029 | 7.227 | 7.593 | -0.436 (-5.4%) | yes |
| LCD/0.95Tc/TIA1-LCD | 7.762 | 7.321 | 6.809 | -0.952 (-12.3%) | no |
| YS_VAR/0.90Tc/S120Y30 | 4.031 | 3.656 | 4.130 | +0.098 (+2.4%) | no |
| YS_VAR/0.90Tc/S12Y3 | 8.021 | 7.097 | 7.788 | -0.233 (-2.9%) | yes |
| YS_VAR/0.90Tc/S24Y6x5 | 7.699 | 6.463 | 7.840 | +0.140 (+1.8%) | no |
| YS_VAR/0.90Tc/S4Yx30 | 7.832 | 7.153 | 7.606 | -0.226 (-2.9%) | yes |
| YS_VAR/0.90Tc/S9Yx15 | 7.800 | 6.977 | 7.566 | -0.234 (-3.0%) | yes |
| YS_VAR/0.90Tc/SSYx50 | 7.795 | 6.999 | 7.665 | -0.130 (-1.7%) | yes |
| YS_VAR/0.90Tc/Sc_S4Yx30 | 8.478 | 7.648 | 8.079 | -0.399 (-4.7%) | no |
| YS_VAR/0.90Tc/YSx75 | 7.836 | 7.054 | 7.686 | -0.151 (-1.9%) | yes |
| YS_VAR/0.90Tc/YYSx50 | 7.627 | 6.896 | 7.377 | -0.251 (-3.3%) | yes |
| YS_VAR/0.90Tc/Yx150 | 7.225 | 6.534 | 6.989 | -0.236 (-3.3%) | yes |
| YS_VAR/0.95Tc/S120Y30 | 3.842 | 3.481 | 4.027 | +0.185 (+4.8%) | no |
| YS_VAR/0.95Tc/S12Y3 | 8.301 | 7.307 | 7.700 | -0.601 (-7.2%) | no |
| YS_VAR/0.95Tc/S24Y6x5 | 6.694 | 5.629 | 6.781 | +0.087 (+1.3%) | yes |
| YS_VAR/0.95Tc/S4Yx30 | 7.628 | 7.052 | 6.903 | -0.725 (-9.5%) | yes |
| YS_VAR/0.95Tc/S9Yx15 | 7.740 | 7.123 | 7.132 | -0.608 (-7.9%) | no |
| YS_VAR/0.95Tc/SSYx50 | 7.974 | 7.348 | 7.228 | -0.746 (-9.4%) | no |
| YS_VAR/0.95Tc/Sc_S4Yx30 | 9.143 | 8.141 | 8.530 | -0.613 (-6.7%) | no |
| YS_VAR/0.95Tc/YSx75 | 7.221 | 6.796 | 6.593 | -0.628 (-8.7%) | no |
| YS_VAR/0.95Tc/YYSx50 | 7.534 | 6.913 | 6.943 | -0.592 (-7.9%) | yes |
| YS_VAR/0.95Tc/Yx150 | 7.797 | 7.079 | 7.245 | -0.552 (-7.1%) | no |

With degree-preserving references (the networkx default), my sigma is 5 to
16% lower than the published value in all 28 systems, and the published mean
is never inside my seed range. With Erdos-Renyi G(n, m) references the
difference is -12% to +5% (median -4%) and the published mean is inside my
seed range in 13 of 28 systems.

## Interpretation

omega and the edge counts match the published values, which confirms that
the graphs, the threshold, the temperatures and the choice of the largest
component are right. sigma is not reproduced with the networkx
default reference and comes much closer with Erdos-Renyi references, so I
think the published sigma used Erdos-Renyi random graphs as the reference;
the article's methods mention Erdos-Renyi random graphs. This is an
inference, not a confirmation: the analysis code that produced the published
values is not in the repository. The Erdos-Renyi agreement is also not
complete. The remaining difference is mostly negative and follows how often
the G(n, m) samples were disconnected (correlation -0.92 across systems), so
the published analysis probably treated disconnected random graphs
differently from this package, which takes the largest component of such a
sample for L_rand. In a side check on 4 systems with many disconnected samples (10 frames
each, one seed), resampling until the random graph was connected narrowed
the gap in all 4, but one (TIA1-LCD at 0.95 Tc) stayed 8% low, so the exact
published procedure remains unknown. Use `reference="erdos_renyi"` to
compare with these published values, and expect sigma to agree within about
12%, not exactly.
