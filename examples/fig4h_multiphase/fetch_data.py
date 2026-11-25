"""Download the data for the Fig 4h example into a local cache.

Two sources, both fetched at run time and never stored in this repository:

* the trajectory, from Zenodo record 10.5281/zenodo.15674310 (CC-BY-4.0),
  file md_data.zip (0.832 GB); only the md_data/fig4h/ members are extracted;
* the published radial density profiles (GPL-3.0), three small .npy files
  from github.com/KULL-Centre/_2024_yasuda_RNA at commit c421eca.

Usage:
    python fetch_data.py [--yes] [--cache DIR]
"""
import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

ZIP_URL = "https://zenodo.org/records/15674310/files/md_data.zip?download=1"
ZIP_NAME = "zenodo_15674310_md_data.zip"
ZIP_MD5 = "63d59912a024174bf049df06b86daafd"
ZIP_SIZE = 832_206_860  # bytes, as listed on Zenodo
MEMBERS = {
    "md_data/fig4h/FUSRGG3-473_PLP-164_polyR40-128_293.dcd": "traj.dcd",
    "md_data/fig4h/FUSRGG3-473_PLP-164_polyR40-128_293.pdb": "top.pdb",
    "md_data/fig4h/fastabib.fasta": "fastabib.fasta",
    "md_data/fig4h/README.txt": "README_zenodo.txt",
}
PROFILE_COMMIT = "c421eca7d803224050742f21f762950507c322ae"
PROFILE_URL = ("https://raw.githubusercontent.com/KULL-Centre/_2024_yasuda_RNA/"
               f"{PROFILE_COMMIT}/data/view2/{{species}}_FUSRGG3-473_PLP-164_polyR40-128.npy")
PROFILE_SPECIES = ("RRP", "PLP", "RNA")  # RRP = FUS-RGG3
USER_AGENT = "condensate-graph-example"
DEFAULT_CACHE = Path.home() / ".cache" / "condensate-graph" / "fig4h"


def md5sum(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url, dest):
    """Stream url to dest (via a .part file), printing progress."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as out:
        total = int(r.headers.get("Content-Length") or 0)
        done, step = 0, 50 << 20
        next_report = step
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total and done >= next_report:
                print(f"  {done / 1e6:7.0f} / {total / 1e6:.0f} MB", flush=True)
                next_report += step
    tmp.replace(dest)


def confirm(question, assume_yes):
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print(question + " [no terminal; rerun with --yes to accept]")
        return False
    return input(question + " [y/N] ").strip().lower() in ("y", "yes")


def fetch_trajectory(cache, assume_yes):
    zpath = cache / ZIP_NAME
    if zpath.exists() and md5sum(zpath) == ZIP_MD5:
        print(f"trajectory archive already in cache: {zpath}")
    else:
        if zpath.exists():
            print(f"cached archive {zpath} has the wrong md5; it will be downloaded again")
        if not confirm(f"Download {ZIP_SIZE / 1e9:.3f} GB from Zenodo to {zpath}?", assume_yes):
            sys.exit("not downloaded")
        print(f"downloading {ZIP_URL}")
        download(ZIP_URL, zpath)
        got = md5sum(zpath)
        if got != ZIP_MD5:
            zpath.unlink()
            sys.exit(f"md5 mismatch: expected {ZIP_MD5}, got {got}; the file was removed")
        print(f"md5 verified: {got}")
    with zipfile.ZipFile(zpath) as zf:
        for member, name in MEMBERS.items():
            out = cache / name
            info = zf.getinfo(member)
            if out.exists() and out.stat().st_size == info.file_size:
                continue
            print(f"extracting {member} -> {out}")
            with zf.open(member) as src, open(out, "wb") as dst:
                while True:
                    block = src.read(1 << 22)
                    if not block:
                        break
                    dst.write(block)


def fetch_profiles(cache, assume_yes):
    missing = [s for s in PROFILE_SPECIES if not (cache / f"published_{s}.npy").exists()]
    if not missing:
        print("published profiles already in cache")
        return
    if not confirm(f"Download {len(missing)} published profile files (about 1 kB each, GPL-3.0) "
                   f"from GitHub commit {PROFILE_COMMIT[:7]}?", assume_yes):
        sys.exit("not downloaded")
    for s in missing:
        url = PROFILE_URL.format(species=s)
        print(f"downloading {url}")
        download(url, cache / f"published_{s}.npy")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="download without asking")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help=f"cache folder (default {DEFAULT_CACHE})")
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    fetch_trajectory(args.cache, args.yes)
    fetch_profiles(args.cache, args.yes)
    print(f"data ready in {args.cache}")


if __name__ == "__main__":
    main()
