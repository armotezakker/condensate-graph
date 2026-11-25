"""Four figures from the output of run_analysis.py.

Usage:
    python make_figures.py [--out DIR]

Writes fig_a_bridging.png, fig_b_graphs.png, fig_c_hubs.png and
fig_d_profiles.png into DIR (default: ./output). Network metrics in figures
a to c use the plateau window found by run_analysis.py; figure b also shows
the full trajectory.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GRAPHS = ("pp", "bridged", "combined", "bipartite")
SPECIES = ("RGG3", "PLP", "RNA")
COLOURS = {"RGG3": "#1f77b4", "PLP": "#2ca02c", "RNA": "#d62728"}
MODE_STYLE = {"plain": dict(color="#444444", label="no hysteresis"),
              "hysteresis": dict(color="#ff7f0e", label="hysteresis (off at 2.0 nm)")}


def window_label(res):
    w = res["plateau_window"]
    ns = res["assumptions"]["frame_spacing_ns"]
    return (f"plateau window, frames {w['first_frame']}-{w['last_frame']} "
            f"({w['first_frame'] * ns / 1000:.1f}-{w['last_frame'] * ns / 1000:.1f} µs at {ns:g} ns per frame)")


def fig_bridging(d, res, mask, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(5)
    bmask = np.isin(d["bridging_frame"], d["frame"][mask])
    for i, mode in enumerate(("plain", "hysteresis")):
        p = np.nanmean(d[f"{mode}_P_n"][bmask], axis=0)
        ax[0].bar(x + (i - 0.5) * 0.38, p, width=0.38, **MODE_STYLE[mode])
    ax[0].set_xticks(x, ["0", "1", "2", "3", "≥4"])
    ax[0].set_xlabel("RNA valency n (protein chains contacted)")
    ax[0].set_ylabel("P(n)")
    ax[0].legend(frameon=False)
    names = ["basic", "productive", "persistent\nk=2", "persistent\nk=5"]
    for i, mode in enumerate(("plain", "hysteresis")):
        m = res["plateau_window"]["bridging"][mode]
        vals = [m["f_bridge_basic"], m["f_bridge_productive"],
                m["f_bridge_persistent"]["2"], m["f_bridge_persistent"]["5"]]
        bars = ax[1].bar(np.arange(4) + (i - 0.5) * 0.38, vals, width=0.38, **MODE_STYLE[mode])
        ax[1].bar_label(bars, fmt="%.3f", fontsize=7)
    ax[1].set_xticks(np.arange(4), names)
    ax[1].set_ylabel("fraction of RNA chains")
    ax[1].set_title("bridging RNA (cutoff 1.5 nm)")
    fig.suptitle(f"(a) RNA valency and bridging; {window_label(res)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_graphs(d, res, mask, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ns = res["assumptions"]["frame_spacing_ns"]
    t_us = d["frame"] * ns / 1000.0
    for g in GRAPHS:
        ax[0].plot(t_us, d[f"{g}_largest_fraction"], lw=0.6, alpha=0.8, label=g)
    ax[0].plot(t_us, d["plateau_rolling"], color="k", lw=1.5, label="combined, rolling mean (50 frames)")
    final = float(d["plateau_final"])
    ax[0].axhline(final, color="k", ls=":", lw=1, label=f"final value {final:.2f}")
    ax[0].axhline(0.95 * final, color="k", ls="--", lw=0.8, label="95% of final value")
    t0 = float(d["plateau_first_frame"]) * ns / 1000.0
    ax[0].axvspan(t0, t_us[-1], color="#cccccc", alpha=0.4, lw=0, label="plateau window")
    ax[0].set_xlabel(f"time (µs, assuming {ns:g} ns per frame)")
    ax[0].set_ylabel("protein chains in largest component (fraction)")
    ax[0].set_ylim(0, 1.05)
    ax[0].legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    perc = np.array([res["plateau_window"]["graphs"][g]["percolating_fraction_xyz"] for g in GRAPHS])
    x = np.arange(len(GRAPHS))
    for i, axis in enumerate("xyz"):
        ax[1].bar(x + (i - 1) * 0.25, perc[:, i], width=0.25, label=axis)
    ax[1].set_xticks(x, GRAPHS)
    ax[1].set_ylim(0, 1.05)
    ax[1].set_ylabel("fraction of window frames percolating")
    ax[1].legend(frameon=False, title="axis")
    if perc.max() == 0:
        ax[1].text(0.5, 0.5, "no graph percolates\nin any frame of the window", transform=ax[1].transAxes,
                   ha="center")
    fig.suptitle("(b) largest component over the full trajectory and periodic percolation in the plateau window",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_hubs(d, res, mask, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(GRAPHS))
    for k, kind in enumerate(("degree", "betweenness")):
        for i, s in enumerate(SPECIES):
            vals = [np.nanmean(d[f"{g}_mean_{kind}_{s}"][mask]) for g in GRAPHS]
            ax[k].bar(x + (i - 1) * 0.27, vals, width=0.27, color=COLOURS[s], label=s)
        ax[k].set_xticks(x, GRAPHS)
        ax[k].set_ylabel(f"mean {kind} per chain")
        ax[k].set_yscale("log" if kind == "betweenness" else "linear")
        for j, g in enumerate(GRAPHS):
            top = res["plateau_window"]["graphs"][g][f"mean_top10_by_{kind}"]
            txt = "/".join(f"{top[s]:.1f}" for s in SPECIES)
            ax[k].annotate(txt, (j, 1.0), xycoords=("data", "axes fraction"), ha="center", va="bottom", fontsize=7)
        ax[k].set_title(f"by {kind}; numbers above: top-10 hubs RGG3/PLP/RNA", fontsize=9, pad=14)
    ax[0].legend(frameon=False)
    fig.suptitle(f"(c) hub chains by species (RNA chains are nodes only in the bipartite graph); "
                 f"{window_label(res)}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_profiles(d, res, mask, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    r = d["r_nm"]
    for s in SPECIES:
        for a in ax:
            a.plot(r, d[f"published_{s}"], color=COLOURS[s], lw=3, alpha=0.3)
            a.plot(r, d[f"profile_as_stored_{s}"], color=COLOURS[s], lw=1.0, ls="--")
            a.plot(r, d[f"profile_recentred_{s}"], color=COLOURS[s], lw=0.8, ls=":")
    agree = res["density_profile_agreement"]
    for s in SPECIES:
        m, alt = agree["as_stored"][s], agree["recentred"][s]
        ax[0].plot([], [], color=COLOURS[s], label=f"{s}: rel. RMS {m['relative_rms']:.3f} / {alt['relative_rms']:.3f}")
    ax[0].plot([], [], color="k", lw=3, alpha=0.3, label="published")
    ax[0].plot([], [], color="k", lw=1, ls="--", label="coordinates as stored (first number)")
    ax[0].plot([], [], color="k", lw=0.8, ls=":", label="centred again (second number)")
    ax[0].legend(frameon=False, fontsize=7)
    ax[0].set_ylabel("concentration (mM)")
    ax[1].set_yscale("log")
    for a in ax:
        a.set_xlabel("distance from droplet centre (nm)")
    fig.suptitle(f"(d) radial density profiles, all {res['system']['frames_analysed']} stored frames, "
                 "vs published (Yasuda et al. Fig 4h)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "output")
    args = ap.parse_args()
    d = dict(np.load(args.out / "per_frame.npz", allow_pickle=False))
    res = json.load(open(args.out / "results.json"))
    mask = d["frame"] >= int(d["plateau_first_frame"])
    figs = {"fig_a_bridging.png": fig_bridging, "fig_b_graphs.png": fig_graphs,
            "fig_c_hubs.png": fig_hubs, "fig_d_profiles.png": fig_profiles}
    for name, fn in figs.items():
        fn(d, res, mask, args.out / name)
        print("wrote", args.out / name)


if __name__ == "__main__":
    main()
