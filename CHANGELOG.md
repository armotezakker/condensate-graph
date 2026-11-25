# Changelog

## 0.1.0 (2026-11-25)

First public version.

- Periodic box handling: the box is read from the trajectory or given
  explicitly (nm); analysis refuses to run without one. Orthorhombic boxes.
- Minimum-image contact search between chains, robust to coordinates that
  round onto the box edge.
- Explicit chain identification by segment, bonded fragment or chain
  lengths, with warnings when the result looks wrong (a single chain where
  many are expected, or broken fragments from missing bonds).
- Bridging metrics: RNA valency, P(n), basic, productive and persistent
  bridges, optional z-region filter.
- Hysteresis contacts and a separate protein-protein cutoff.
- Chain graphs (pp, bridged, combined, bipartite), largest component, hubs
  by degree and betweenness, and a periodic percolation test based on image
  shifts.
- Small-world coefficients sigma and omega, with degree-preserving or
  Erdős-Rényi references for sigma.
- Validation against published condensate networks (docs/validation.md) and
  a worked example on a published CALVADOS-RNA trajectory
  (examples/fig4h_multiphase).
