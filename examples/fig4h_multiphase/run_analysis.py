"""Bridging, chain graphs and density profiles for the Fig 4h multiphase droplet.

System (Yasuda et al., JCTC 2025, Fig 4h): 473 FUS-RGG3 chains (34 beads),
164 PLP chains (173 beads) and 128 polyR40 RNA chains (80 beads, two per
nucleotide) in a cubic box of 110.8 nm, CALVADOS-RNA at 293 K.

Every stored frame is analysed. Network metrics are then reported on the
plateau window, where the largest component of the combined graph has
stopped growing (see find_plateau). Run fetch_data.py first. Writes
results.json and per_frame.npz to --out.

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
BEADS_PER_CHAIN = {"RGG3": 34, "PLP": 173, "RNA": 80}
SPECIES = ("RGG3", "PLP", "RNA")
GRAPHS = ("pp", "bridged", "combined", "bipartite")
CUTOFF, CUTOFF_OFF, MIN_FRAMES = 1.5, 2.0, (1, 2, 5)
FRAME_NS = 10.0       # assumed frame spacing; not stated in the data files
TOP_N = 10            # hubs counted per frame when splitting by species
CENTRE_BIN_NM = 0.1   # bins for locating the density maximum along each axis
RADIAL_BIN_NM = 1.0   # bins of the radial density profile
MM_PER_CHAIN_PER_NM3 = 1e24 / 6.02214076e23 * 1e3  # 1 chain per nm^3 = 1660.54 mM
PROFILE_MODES = ("as_stored", "recentred")
# plateau rule (see find_plateau)
PLATEAU_GRAPH = "combined"
PLATEAU_ROLLING = 50          # frames in the centred rolling mean (0.5 us at 10 ns per frame)
PLATEAU_FINAL_SHARE = 0.10    # final value = mean over the last 10% of frames
PLATEAU_TOLERANCE = 0.05      # rolling mean must stay within 5% of the final value


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


def radial_counts(frame, protein_species, protein_chain, rna_n):
    """Bead counts per radial shell around the box centre, per species.

    "as_stored" uses the stored coordinates (wrapped into the box), which is
    right for this file because the droplet is already centred in every
    frame. "recentred" first applies :func:`centre_droplet` again.
    """
    box = frame.box
    allpos = np.vstack([frame.protein_positions, frame.rna_positions])
    edges = np.arange(0.0, 0.5 * box.min(), RADIAL_BIN_NM)
    bead_species = np.concatenate([protein_species[protein_chain], np.full(rna_n, "RNA")])
    out = {}
    for mode, pos in (("as_stored", np.mod(allpos, box)), ("recentred", centre_droplet(allpos, box))):
        r = np.linalg.norm(pos - 0.5 * box, axis=1)
        out[mode] = {s: np.histogram(r[bead_species == s], bins=edges)[0] for s in SPECIES}
    return out, edges


def to_mM(counts, edges, beads_per_chain):
    """Chain concentration in mM from bead counts per shell (shell volume 4 pi r^2 dr, as in the paper)."""
    centres = 0.5 * (edges[1:] + edges[:-1])
    shell = 4.0 * np.pi * centres ** 2 * np.diff(edges)
    return counts / beads_per_chain / shell * MM_PER_CHAIN_PER_NM3


def agreement(ours, theirs, r, rmax=50.0):
    """Pearson r, relative RMS difference and core (r < 10 nm) ratio on r <= rmax."""
    m = (r <= rmax) & np.isfinite(ours) & np.isfinite(theirs)
    core = m & (r < 10.0)
    return {"pearson_r": float(np.corrcoef(ours[m], theirs[m])[0, 1]),
            "relative_rms": float(np.sqrt(np.mean((ours[m] - theirs[m]) ** 2)) / np.mean(theirs[m])),
            "max_abs_diff_mM": float(np.max(np.abs(ours[m] - theirs[m]))),
            "core_ratio_ours_over_theirs": float(ours[core].mean() / theirs[core].mean()),
            "r_max_nm": rmax}


# ------------------------------------------------------------------ plateau window

def find_plateau(series):
    """Index where a growing series stops growing.

    Rule: take a centred rolling mean over PLATEAU_ROLLING frames (near the
    ends it averages over the frames that exist) and the final value as the
    mean of the last PLATEAU_FINAL_SHARE of the frames. The plateau starts at
    the first frame where the rolling mean reaches (1 - PLATEAU_TOLERANCE) x
    final value. Later dips are treated as fluctuations; how often the
    rolling mean falls below the threshold again is returned as a check.

    Returns (start index, rolling mean, final value, share of later frames
    below the threshold, strict start index). The strict start is the first
    frame from which the rolling mean never falls below the threshold again;
    it is reported for comparison only, because a single late dip moves it.
    """
    x = np.asarray(series, dtype=float)
    w = min(PLATEAU_ROLLING, len(x))
    kernel = np.ones(w)
    rolling = np.convolve(x, kernel, mode="same") / np.convolve(np.ones_like(x), kernel, mode="same")
    final = float(x[-max(1, int(round(PLATEAU_FINAL_SHARE * len(x)))):].mean())
    above = rolling >= (1 - PLATEAU_TOLERANCE) * final
    start = int(np.argmax(above)) if above.any() else len(x)
    below_later = float((~above[start:]).mean()) if start < len(x) else float("nan")
    bad = np.nonzero(~above)[0]
    strict = int(bad[-1] + 1) if len(bad) else 0
    return start, rolling, final, below_later, strict


# ------------------------------------------------------------------ per-frame work (parallel)

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
    """Graph metrics, hub statistics, radial counts and small-world for frames [a, b)."""
    cache, a, b, sw_frames, sw_params = args
    from condensate_graph import graph_metrics
    u, system = load(Path(cache))
    sp = species_of_protein_chains(system)
    n_nodes = system.protein.n_chains + system.rna.n_chains

    def species_of_node(node):
        return sp[node[1]] if node[0] == "P" else "RNA"

    pc, npc = system.protein.chain_of_atom, system.protein.n_chains
    rc, nrc = system.rna.chain_of_atom, system.rna.n_chains
    out = {"radial": {m: {s: [] for s in SPECIES} for m in PROFILE_MODES}, "graphs": {g: {} for g in GRAPHS},
           "sw": {"frame": [], "sigma": [], "omega": []}}
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
        out["sw"] = {"frame": swr.frame_index.tolist(), "sigma": list(s.sigma), "omega": list(s.omega)}
    return out


def bridging_run(args):
    cache, start, stop, cutoff_off = args
    from condensate_graph import bridging_metrics
    u, system = load(Path(cache))
    r = bridging_metrics(system.iter_frames(start, stop), system.protein.chain_of_atom, system.protein.n_chains,
                         system.rna.chain_of_atom, system.rna.n_chains,
                         cutoff=CUTOFF, cutoff_off=cutoff_off, min_frames=MIN_FRAMES)
    return {"frame": r.frame_index, "P_n": r.valency_distribution, "B": r.mean_valency,
            "f_basic": r.f_bridge_basic, "f_productive": r.f_bridge_productive,
            "f_persistent": {k: v for k, v in r.f_bridge_persistent.items()}}


# ------------------------------------------------------------------ summaries

def nanmean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def summarise(mask, bridging, graphs, sw_frame, sw_sigma, sw_omega, frames):
    """Time averages over the frames selected by mask (boolean over all analysed frames)."""
    sel = set(frames[mask].tolist())
    sw_m = np.array([f in sel for f in sw_frame], dtype=bool)
    out = {"first_frame": int(frames[mask][0]), "last_frame": int(frames[mask][-1]), "n_frames": int(mask.sum()),
           "bridging": {}, "graphs": {},
           "small_world_combined": {"n_frames": int(sw_m.sum()),
                                    "sigma_mean": nanmean(sw_sigma[sw_m]), "sigma_sd": float(np.nanstd(sw_sigma[sw_m])),
                                    "omega_mean": nanmean(sw_omega[sw_m]), "omega_sd": float(np.nanstd(sw_omega[sw_m]))}}
    for name, b in bridging.items():
        bm = np.isin(b["frame"], frames[mask])
        out["bridging"][name] = {
            "B": nanmean(b["B"][bm]),
            "P_n": [nanmean(c) for c in np.asarray(b["P_n"])[bm].T],
            "f_bridge_basic": nanmean(b["f_basic"][bm]),
            "f_bridge_productive": nanmean(b["f_productive"][bm]),
            "f_bridge_persistent": {str(k): nanmean(v[bm]) for k, v in b["f_persistent"].items()},
        }
    for g, d in graphs.items():
        out["graphs"][g] = {
            "mean_n_components": float(np.mean(d["n_components"][mask])),
            "mean_largest_fraction": float(np.mean(d["largest_fraction"][mask])),
            "percolating_fraction_xyz": np.mean(np.asarray(d["percolates"], float)[mask], axis=0).tolist(),
            "mean_degree": {s: nanmean(v[mask]) for s, v in d["mean_degree"].items()},
            "mean_betweenness": {s: nanmean(v[mask]) for s, v in d["mean_betweenness"].items()},
            f"mean_top{TOP_N}_by_degree": {s: float(np.mean(v[mask])) for s, v in d[f"top{TOP_N}_degree"].items()},
            f"mean_top{TOP_N}_by_betweenness": {s: float(np.mean(v[mask])) for s, v in d[f"top{TOP_N}_betweenness"].items()},
        }
    return out


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "output")
    ap.add_argument("--workers", type=int, default=max(1, min(48, (os.cpu_count() or 2) - 8)))
    ap.add_argument("--chunk", type=int, default=20, help="frames per parallel task")
    ap.add_argument("--sw-every", type=int, default=10, help="small-world on every n-th frame")
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
          f"RNA chains {system.rna.n_chains}; analysing frames 0..{stop - 1} "
          f"(assumed {FRAME_NS:g} ns apart)", flush=True)

    sw_frames = set(range(0, stop, args.sw_every))
    sw_params = {"niter": args.sw_niter, "nrand": args.sw_nrand, "seed": 0}
    chunks = [(str(args.cache), a, min(a + args.chunk, stop), sw_frames, sw_params)
              for a in range(0, stop, args.chunk)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        bridging_jobs = {name: ex.submit(bridging_run, (str(args.cache), 0, stop, off))
                         for name, off in (("plain", None), ("hysteresis", CUTOFF_OFF))}
        parts = []
        for i, part in enumerate(ex.map(graph_chunk, chunks)):
            parts.append(part)
            if (i + 1) % 10 == 0 or i + 1 == len(chunks):
                print(f"  graph chunks {i + 1}/{len(chunks)} done, {time.time() - t0:.0f} s", flush=True)
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
    sw_frame = np.concatenate([p["sw"]["frame"] for p in parts]).astype(int)
    sw_sigma = np.concatenate([p["sw"]["sigma"] for p in parts]).astype(float)
    sw_omega = np.concatenate([p["sw"]["omega"] for p in parts]).astype(float)

    # density profiles over all stored frames, compared with the published curves
    edges = parts[0]["radial_edges"]
    r_centres = 0.5 * (edges[1:] + edges[:-1])
    profiles, published, agree = {}, {}, {m: {} for m in PROFILE_MODES}
    for m in PROFILE_MODES:
        counts = {s: np.concatenate([np.array(p["radial"][m][s]) for p in parts]) for s in SPECIES}
        profiles[m] = {s: to_mM(counts[s].mean(axis=0), edges, BEADS_PER_CHAIN[s]) for s in SPECIES}
    for s, fname in (("RGG3", "RRP"), ("PLP", "PLP"), ("RNA", "RNA")):
        pub = np.load(args.cache / f"published_{fname}.npy", allow_pickle=False)
        if not np.allclose(pub[0], r_centres):
            raise ValueError(f"published {fname} profile uses other radial bins")
        published[s] = pub[1]
        for m in PROFILE_MODES:
            agree[m][s] = agreement(profiles[m][s], pub[1], r_centres)

    # plateau window from the combined graph
    start_idx, rolling, final, below_later, strict = find_plateau(graphs[PLATEAU_GRAPH]["largest_fraction"])
    found = start_idx < len(frames)
    if not found:
        print("no plateau found: the largest component is still growing at the end; "
              "the 'plateau window' below is only the last frame", flush=True)
    start_idx, strict = min(start_idx, len(frames) - 1), min(strict, len(frames) - 1)
    window = np.zeros(len(frames), dtype=bool)
    window[start_idx:] = True
    everything = np.ones(len(frames), dtype=bool)

    summary = {
        "system": {"beads": int(u.atoms.n_atoms), "frames_in_file": n_frames, "frames_analysed": int(len(frames)),
                   "box_nm": (u.dimensions[:3] / 10).tolist(),
                   "chains": {"RGG3": int((sp == "RGG3").sum()), "PLP": int((sp == "PLP").sum()),
                              "RNA": int(system.rna.n_chains)}},
        "assumptions": {"frame_spacing_ns": FRAME_NS,
                        "note": "frame spacing is not stated in the data files; 10 ns is assumed "
                                "(the paper's condensate runs were written every 10 ns)"},
        "settings": {"cutoff_nm": CUTOFF, "cutoff_off_nm": CUTOFF_OFF, "persistence_k": list(MIN_FRAMES),
                     "small_world": dict(sw_params, every_nth_frame=args.sw_every, graph="combined",
                                         reference="degree_preserving"),
                     "radial_bin_nm": RADIAL_BIN_NM, "centring_bin_nm": CENTRE_BIN_NM},
        "plateau": {"graph": PLATEAU_GRAPH, "found": bool(found), "first_frame": int(frames[start_idx]),
                    "first_time_us": float(frames[start_idx] * FRAME_NS / 1000.0),
                    "n_frames": int(window.sum()), "final_value": final,
                    "rule": (f"centred rolling mean over {PLATEAU_ROLLING} frames of the {PLATEAU_GRAPH} "
                             f"largest-component fraction; final value = mean of the last "
                             f"{PLATEAU_FINAL_SHARE:.0%} of frames; the window starts at the first frame where "
                             f"the rolling mean reaches {1 - PLATEAU_TOLERANCE:.0%} of the final value and "
                             "runs to the end"),
                    "share_of_window_below_threshold": below_later,
                    "strict_first_frame": int(frames[strict]),
                    "strict_note": ("first frame from which the rolling mean never falls below the threshold "
                                    "again; reported for comparison, not used, because one late dip moves it")},
        "plateau_window": summarise(window, bridging, graphs, sw_frame, sw_sigma, sw_omega, frames),
        "all_frames": summarise(everything, bridging, graphs, sw_frame, sw_sigma, sw_omega, frames),
        "density_profile_agreement": agree,
        "density_profile_note": ("all stored frames. as_stored: coordinates as in the file (the droplet is "
                                 "already centred in every frame, so the deposited DCD appears to be the "
                                 "paper's processed trajectory). recentred: the paper's centring applied "
                                 "once more."),
        "runtime_s": None,
    }
    np.savez_compressed(
        args.out / "per_frame.npz", frame=frames, r_nm=r_centres,
        plateau_first_frame=frames[start_idx], plateau_rolling=rolling, plateau_final=final,
        **{f"profile_{m}_{s}": profiles[m][s] for m in PROFILE_MODES for s in SPECIES},
        **{f"published_{s}": published[s] for s in SPECIES},
        **{f"{name}_{k}": np.asarray(b[k], float) for name, b in bridging.items()
           for k in ("P_n", "B", "f_basic", "f_productive")},
        **{f"{name}_f_persistent_k{k}": np.asarray(v, float) for name, b in bridging.items()
           for k, v in b["f_persistent"].items()},
        bridging_frame=np.asarray(bridging["plain"]["frame"]),
        **{f"{g}_{k}": np.asarray(graphs[g][k], float) for g in GRAPHS
           for k in ("n_components", "largest_fraction", "percolates")},
        **{f"{g}_{stat}_{s}": np.asarray(graphs[g][stat][s], float) for g in GRAPHS
           for stat in ("mean_degree", "mean_betweenness", f"top{TOP_N}_degree", f"top{TOP_N}_betweenness")
           for s in SPECIES},
        sw_frame=sw_frame, sw_sigma=sw_sigma, sw_omega=sw_omega,
    )
    summary["runtime_s"] = round(time.time() - t0, 1)
    with open(args.out / "results.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out / 'results.json'} and {args.out / 'per_frame.npz'} in {summary['runtime_s']} s")


if __name__ == "__main__":
    main()
