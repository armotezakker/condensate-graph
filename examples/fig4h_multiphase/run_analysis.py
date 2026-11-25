"""Bridging, chain graphs and density profiles for the Fig 4h multiphase droplet.

System (Yasuda et al., JCTC 2025, Fig 4h): 473 FUS-RGG3 chains (34 beads),
164 PLP chains (173 beads) and 128 polyR40 RNA chains (80 beads, two per
nucleotide) in a cubic box of 110.8 nm, CALVADOS-RNA at 293 K.

Run fetch_data.py first. Writes results.json and per_frame.npz to --out.

Usage:
    python run_analysis.py [--cache DIR] [--out DIR] [--workers N]
"""
import argparse
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DEFAULT_CACHE = Path.home() / ".cache" / "condensate-graph" / "fig4h"
PROTEIN_SEL, RNA_SEL = "not resname r", "resname r"  # RNA beads are named r; protein uses upper case
SPECIES_BY_LENGTH = {34: "RGG3", 173: "PLP"}
SPECIES = ("RGG3", "PLP", "RNA")
GRAPHS = ("pp", "bridged", "combined", "bipartite")
CUTOFF, CUTOFF_OFF, MIN_FRAMES = 1.5, 2.0, (1, 2, 5)
START = 100           # frames discarded at the start, as in the paper's analysis
FRAME_NS = 10.0       # assumed frame spacing; not stated in the data files
TOP_N = 10            # hubs counted per frame when splitting by species
CENTRE_BIN_NM = 0.1   # bins for locating the density maximum along each axis
RADIAL_BIN_NM = 1.0   # bins of the radial density profile
MM_PER_CHAIN_PER_NM3 = 1e24 / 6.02214076e23 * 1e3  # 1 chain per nm^3 = 1660.54 mM


def load(cache):
    import MDAnalysis as mda
    from condensate_graph import build_system
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="MDAnalysis")  # missing element / mass fields in the PDB
        u = mda.Universe(str(cache / "top.pdb"), str(cache / "traj.dcd"))
    return u, build_system(u, PROTEIN_SEL, RNA_SEL, by="segment")


def species_of_protein_chains(system):
    sizes = system.protein.chain_sizes()
    unknown = sorted(set(sizes.tolist()) - set(SPECIES_BY_LENGTH))
    if unknown:
        raise ValueError(f"protein chains of unexpected length {unknown}")
    return np.array([SPECIES_BY_LENGTH[s] for s in sizes.tolist()])


# ------------------------------------------------------------------ radial density profile

def centre_droplet(pos, box):
    """Shift positions so that the density maximum along z, then y, then x sits at the box centre.

    Same idea as the paper's analysis: per axis, histogram all beads, move the
    most populated bin to the middle of the box and wrap. Positions in nm.
    """
    pos = pos.copy()
    for axis in (2, 1, 0):
        length = box[axis]
        edges = np.arange(0.0, length + CENTRE_BIN_NM, CENTRE_BIN_NM)
        h, _ = np.histogram(np.mod(pos[:, axis], length), bins=edges)
        peak = 0.5 * (edges[np.argmax(h)] + edges[np.argmax(h) + 1])
        pos[:, axis] = np.mod(pos[:, axis] - peak + 0.5 * length, length)
    return pos


PROFILE_MODES = ("recentred", "as_is")


def radial_counts(frame, protein_species, protein_chain, rna_n):
    """Bead counts per radial shell around the box centre, per species.

    Two versions: "recentred" first centres the droplet with
    :func:`centre_droplet` (the paper's approach applied to this file);
    "as_is" uses the coordinates as stored (wrapped), which is right if the
    file was already centred.
    """
    box = frame.box
    allpos = np.vstack([frame.protein_positions, frame.rna_positions])
    edges = np.arange(0.0, 0.5 * box.min(), RADIAL_BIN_NM)
    bead_species = np.concatenate([protein_species[protein_chain], np.full(rna_n, "RNA")])
    out = {}
    for mode, pos in (("recentred", centre_droplet(allpos, box)), ("as_is", np.mod(allpos, box))):
        r = np.linalg.norm(pos - 0.5 * box, axis=1)
        out[mode] = {s: np.histogram(r[bead_species == s], bins=edges)[0] for s in SPECIES}
    return out, edges


def to_mM(counts, edges, beads_per_chain):
    """Chain concentration in mM from bead counts per shell (shell volume 4 pi r^2 dr, as in the paper)."""
    centres = 0.5 * (edges[1:] + edges[:-1])
    shell = 4.0 * np.pi * centres ** 2 * np.diff(edges)
    return counts / beads_per_chain / shell * MM_PER_CHAIN_PER_NM3


# ------------------------------------------------------------------ per-frame graph work (parallel)

def _hub_stats(top, species_of_node):
    """Mean value per species over all nodes, and species counts among the top TOP_N."""
    vals = {s: [] for s in SPECIES}
    for node, v in top:
        vals[species_of_node(node)].append(v)
    mean = {s: (float(np.mean(v)) if v else float("nan")) for s, v in vals.items()}
    top_counts = {s: 0 for s in SPECIES}
    for node, _ in top[:TOP_N]:
        top_counts[species_of_node(node)] += 1
    return mean, top_counts


def graph_chunk(args):
    """Graph metrics, hub statistics and radial counts for frames [a, b) of the trajectory."""
    cache, a, b, sw_frames, sw_params = args
    from condensate_graph import graph_metrics
    u, system = load(Path(cache))
    sp = species_of_protein_chains(system)
    n_nodes = system.protein.n_chains + system.rna.n_chains

    def species_of_node(node):
        return sp[node[1]] if node[0] == "P" else "RNA"

    pc, npc = system.protein.chain_of_atom, system.protein.n_chains
    rc, nrc = system.rna.chain_of_atom, system.rna.n_chains
    out = {"frame": [], "radial": {m: {s: [] for s in SPECIES} for m in PROFILE_MODES}, "graphs": {g: {} for g in GRAPHS},
           "sw": {"frame": [], "sigma": [], "omega": [], "n_nodes": []}}
    frames = list(system.iter_frames(a, b))
    res = graph_metrics(frames, pc, npc, rc, nrc, cutoff=CUTOFF, graphs=GRAPHS, top_n=n_nodes, betweenness=True)
    out["frame"] = res.frame_index.tolist()
    for g, s in res.series.items():
        d = out["graphs"][g]
        d["n_components"] = list(s.n_components)
        d["largest_fraction"] = list(s.largest_fraction)
        d["percolates"] = [p.tolist() for p in s.percolates]
        for kind, tops in (("degree", s.top_degree), ("betweenness", s.top_betweenness)):
            means, counts = zip(*(_hub_stats(t, species_of_node) for t in tops))
            d[f"mean_{kind}"] = {x: [m[x] for m in means] for x in SPECIES}
            d[f"top{TOP_N}_{kind}"] = {x: [c[x] for c in counts] for x in SPECIES}
    for f in frames:
        counts, edges = radial_counts(f, sp, pc, len(f.rna_positions))
        for m in PROFILE_MODES:
            for s in SPECIES:
                out["radial"][m][s].append(counts[m][s])
    out["radial_edges"] = edges
    chosen = [f for f in frames if f.index in sw_frames]
    if chosen:
        swr = graph_metrics(chosen, pc, npc, rc, nrc, cutoff=CUTOFF, graphs=("combined",),
                            top_n=1, betweenness=False, small_world_params=sw_params)
        s = swr.series["combined"]
        out["sw"] = {"frame": swr.frame_index.tolist(), "sigma": list(s.sigma), "omega": list(s.omega),
                     "largest_fraction": list(s.largest_fraction)}
    return out


def radial_chunk(args):
    """Radial bead counts only, for frames [a, b)."""
    cache, a, b = args
    u, system = load(Path(cache))
    sp = species_of_protein_chains(system)
    out = {m: {s: [] for s in SPECIES} for m in PROFILE_MODES}
    for f in system.iter_frames(a, b):
        counts, edges = radial_counts(f, sp, system.protein.chain_of_atom, len(f.rna_positions))
        for m in PROFILE_MODES:
            for s in SPECIES:
                out[m][s].append(counts[m][s])
    return out


def bridging_run(args):
    cache, start, stop, cutoff_off = args
    from condensate_graph import bridging_metrics
    u, system = load(Path(cache))
    r = bridging_metrics(system.iter_frames(start, stop), system.protein.chain_of_atom, system.protein.n_chains,
                         system.rna.chain_of_atom, system.rna.n_chains,
                         cutoff=CUTOFF, cutoff_off=cutoff_off, min_frames=MIN_FRAMES)
    return {
        "frame": r.frame_index.tolist(),
        "P_n": r.valency_distribution.tolist(),
        "B": r.mean_valency.tolist(),
        "f_basic": r.f_bridge_basic.tolist(),
        "f_productive": r.f_bridge_productive.tolist(),
        "f_persistent": {str(k): v.tolist() for k, v in r.f_bridge_persistent.items()},
        "means": {k: (v.tolist() if isinstance(v, np.ndarray) else
                      ({str(a): b for a, b in v.items()} if isinstance(v, dict) else v))
                  for k, v in r.means().items()},
    }


# ------------------------------------------------------------------ main

def nanmean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def agreement(ours, theirs, r, rmax):
    """Agreement of two profiles on r <= rmax: Pearson r, relative RMS and mean ratio in the core (r < 10 nm)."""
    m = (r <= rmax) & np.isfinite(ours) & np.isfinite(theirs)
    core = m & (r < 10.0)
    rel_rms = float(np.sqrt(np.mean((ours[m] - theirs[m]) ** 2)) / np.mean(theirs[m]))
    return {"pearson_r": float(np.corrcoef(ours[m], theirs[m])[0, 1]),
            "relative_rms": rel_rms,
            "core_ratio_ours_over_theirs": float(ours[core].mean() / theirs[core].mean()),
            "core_mean_ours_mM": float(ours[core].mean()), "core_mean_theirs_mM": float(theirs[core].mean()),
            "r_max_nm": rmax}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "output")
    ap.add_argument("--workers", type=int, default=max(1, min(48, (os.cpu_count() or 2) - 8)))
    ap.add_argument("--chunk", type=int, default=20, help="frames per parallel task")
    ap.add_argument("--sw-every", type=int, default=10, help="small-world on every n-th analysed frame")
    ap.add_argument("--sw-niter", type=int, default=5)
    ap.add_argument("--sw-nrand", type=int, default=10)
    ap.add_argument("--stop", type=int, default=None, help="last frame (exclusive); default all")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    u, system = load(args.cache)
    sp = species_of_protein_chains(system)
    n_frames = len(u.trajectory)
    stop = n_frames if args.stop is None else min(args.stop, n_frames)
    print(f"{u.atoms.n_atoms} beads, {n_frames} frames, box {u.dimensions[:3] / 10} nm")
    print(f"protein chains {system.protein.n_chains} ({(sp == 'RGG3').sum()} RGG3, {(sp == 'PLP').sum()} PLP), "
          f"RNA chains {system.rna.n_chains}; analysing frames {START}..{stop - 1} "
          f"(assumed {FRAME_NS:g} ns apart)", flush=True)

    sw_frames = set(range(START, stop, args.sw_every))
    sw_params = {"niter": args.sw_niter, "nrand": args.sw_nrand, "seed": 0}
    chunks = [(str(args.cache), a, min(a + args.chunk, stop), sw_frames, sw_params)
              for a in range(START, stop, args.chunk)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        bridging_jobs = {name: ex.submit(bridging_run, (str(args.cache), START, stop, off))
                         for name, off in (("plain", None), ("hysteresis", CUTOFF_OFF))}
        parts = []
        for i, part in enumerate(ex.map(graph_chunk, chunks)):
            parts.append(part)
            if (i + 1) % 10 == 0 or i + 1 == len(chunks):
                print(f"  graph chunks {i + 1}/{len(chunks)} done, {time.time() - t0:.0f} s", flush=True)
        early = list(ex.map(radial_chunk, [(str(args.cache), a, min(a + args.chunk, START))
                                           for a in range(0, START, args.chunk)]))
        bridging = {name: job.result() for name, job in bridging_jobs.items()}
    print(f"bridging done, {time.time() - t0:.0f} s", flush=True)

    frames = np.concatenate([p["frame"] for p in parts])
    graphs = {}
    for g in GRAPHS:
        graphs[g] = {k: np.concatenate([p["graphs"][g][k] for p in parts])
                     for k in ("n_components", "largest_fraction", "percolates")}
        for kind in ("degree", "betweenness"):
            for stat in (f"mean_{kind}", f"top{TOP_N}_{kind}"):
                graphs[g][stat] = {s: np.concatenate([p["graphs"][g][stat][s] for p in parts]) for s in SPECIES}
    edges = parts[0]["radial_edges"]
    r_centres = 0.5 * (edges[1:] + edges[:-1])
    beads = {"RGG3": 34, "PLP": 173, "RNA": 80}
    variants = {}  # (mode, frames) -> profile per species, in mM
    for m in PROFILE_MODES:
        late = {sp_: np.concatenate([np.array(p["radial"][m][sp_]) for p in parts]) for sp_ in SPECIES}
        allf = {sp_: np.concatenate([np.array(p[m][sp_]) for p in early] + [late[sp_]]) for sp_ in SPECIES}
        variants[f"{m}_from{START}"] = {sp_: to_mM(late[sp_].mean(axis=0), edges, beads[sp_]) for sp_ in SPECIES}
        variants[f"{m}_all_frames"] = {sp_: to_mM(allf[sp_].mean(axis=0), edges, beads[sp_]) for sp_ in SPECIES}
    sw = {k: np.concatenate([p["sw"].get(k, []) for p in parts]) for k in ("frame", "sigma", "omega", "largest_fraction")}

    # published profiles
    published, agree = {}, {v: {} for v in variants}
    for s, fname in (("RGG3", "RRP"), ("PLP", "PLP"), ("RNA", "RNA")):
        pub = np.load(args.cache / f"published_{fname}.npy", allow_pickle=False)
        if not np.allclose(pub[0], r_centres):
            raise ValueError(f"published {fname} profile uses other radial bins")
        published[s] = pub[1]
        for v, prof in variants.items():
            agree[v][s] = agreement(prof[s], pub[1], r_centres, rmax=50.0)

    summary = {
        "system": {"beads": int(u.atoms.n_atoms), "frames_in_file": n_frames, "frames_analysed": int(len(frames)),
                   "first_frame": START, "box_nm": (u.dimensions[:3] / 10).tolist(),
                   "chains": {"RGG3": int((sp == "RGG3").sum()), "PLP": int((sp == "PLP").sum()),
                              "RNA": int(system.rna.n_chains)}},
        "assumptions": {"frame_spacing_ns": FRAME_NS,
                        "note": "frame spacing is not stated in the data files; 10 ns is assumed "
                                "(the paper's condensate runs were written every 10 ns)"},
        "settings": {"cutoff_nm": CUTOFF, "cutoff_off_nm": CUTOFF_OFF, "persistence_k": list(MIN_FRAMES),
                     "small_world": dict(sw_params, every_nth_frame=args.sw_every, graph="combined",
                                         reference="degree_preserving"),
                     "radial_bin_nm": RADIAL_BIN_NM, "centring_bin_nm": CENTRE_BIN_NM},
        "bridging": {name: b["means"] for name, b in bridging.items()},
        "graphs": {g: {"mean_n_components": float(np.mean(d["n_components"])),
                       "mean_largest_fraction": float(np.mean(d["largest_fraction"])),
                       "percolating_fraction_xyz": np.mean(np.asarray(d["percolates"], float), axis=0).tolist(),
                       "mean_degree": {s: nanmean(v) for s, v in d["mean_degree"].items()},
                       "mean_betweenness": {s: nanmean(v) for s, v in d["mean_betweenness"].items()},
                       f"mean_top{TOP_N}_by_degree": {s: float(np.mean(v)) for s, v in d[f"top{TOP_N}_degree"].items()},
                       f"mean_top{TOP_N}_by_betweenness": {s: float(np.mean(v)) for s, v in d[f"top{TOP_N}_betweenness"].items()}}
                   for g, d in graphs.items()},
        "small_world_combined": {"n_frames": int(len(sw["frame"])), "sigma_mean": nanmean(sw["sigma"]),
                                 "sigma_sd": float(np.nanstd(sw["sigma"])), "omega_mean": nanmean(sw["omega"]),
                                 "omega_sd": float(np.nanstd(sw["omega"]))},
        "density_profile_agreement": agree,
        "density_profile_note": (
            f"main analysis uses the paper's centring and frames {START} onward (recentred_from{START}). "
            "The Zenodo trajectory is already centred in every frame, so it is probably the paper's "
            "processed trajectory with its first 100 frames removed; the other variants test that."),
        "runtime_s": None,
    }
    np.savez_compressed(
        args.out / "per_frame.npz", frame=frames, r_nm=r_centres,
        **{f"profile_{v}_{sp_}": prof[sp_] for v, prof in variants.items() for sp_ in SPECIES}, **{f"published_{s}": published[s] for s in SPECIES},
        **{f"{name}_{k}": np.asarray(v, float) for name, b in bridging.items()
           for k, v in b.items() if k in ("P_n", "B", "f_basic", "f_productive")},
        **{f"{name}_f_persistent_k{k}": np.asarray(v, float) for name, b in bridging.items()
           for k, v in b["f_persistent"].items()},
        bridging_frame=np.asarray(bridging["plain"]["frame"]),
        **{f"{g}_{k}": np.asarray(graphs[g][k], float) for g in GRAPHS
           for k in ("n_components", "largest_fraction", "percolates")},
        **{f"{g}_{stat}_{s}": np.asarray(graphs[g][stat][s], float) for g in GRAPHS
           for stat in ("mean_degree", "mean_betweenness", f"top{TOP_N}_degree", f"top{TOP_N}_betweenness")
           for s in SPECIES},
        sw_frame=sw["frame"], sw_sigma=sw["sigma"], sw_omega=sw["omega"],
    )
    summary["runtime_s"] = round(time.time() - t0, 1)
    with open(args.out / "results.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out / 'results.json'} and {args.out / 'per_frame.npz'} in {summary['runtime_s']} s")


if __name__ == "__main__":
    main()
