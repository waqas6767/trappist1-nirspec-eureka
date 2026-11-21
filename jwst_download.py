#!/usr/bin/env python
"""Download JWST NIRSpec PRISM time-series data of TRAPPIST-1 from MAST.

Programs:
  GO 1331  JWST-TST DREAMS transits of TRAPPIST-1 e (Espinoza et al. 2025)
  GO 6456  TRAPPIST-1 e/b program, Cycle 3
  GO 9256  TRAPPIST-1 e/b program, Cycle 4

Data levels (detector NRS1 only; the PRISM spectrum falls on NRS1):
  uncal     raw ramps, for a full reduction from Stage 1 (group-level 1/f correction)
  rateints  per-integration count-rate images, for a reduction from Stage 2

Files are stored as  <outdir>/<level>/<program>/<visit>/<file>.  Files whose size
already matches MAST are skipped, so the script can be stopped and restarted.

Usage
  python jwst_download.py --check                       # sizes only, nothing downloaded
  python jwst_download.py --level rateints              # download rateints
  python jwst_download.py --level uncal                 # download uncal
  python jwst_download.py --level rateints --verify     # compare local files with MAST
Options
  --programs 1331 6456 9256   programs to include (default: all three)
  --outdir PATH               where to store the data (default: ./jwst_data)
"""
import argparse
import os
import shutil
import sys

import numpy as np
from astropy.table import unique, vstack
from astroquery.mast import Observations

LEVELS = {"uncal": "UNCAL", "rateints": "RATEINTS"}


def _col(table, name):
    """Column as plain strings; masked (empty) MAST entries become ''."""
    c = table[name]
    if hasattr(c, "filled"):
        c = c.filled("")
    return [str(x) for x in c]


def find_products(programs, level):
    """Return the table of unique NRS1 PRISM products of one level for the programs."""
    obs = Observations.query_criteria(obs_collection="JWST", proposal_id=list(programs),
                                      target_name="TRAPPIST-1", dataRights="PUBLIC")
    obs = obs[np.array([("NIRSPEC" in str(i).upper()) for i in _col(obs, "instrument_name")], dtype=bool)]
    if len(obs) == 0:
        return None
    # query in chunks to keep MAST requests small
    tabs = [Observations.get_product_list(obs[i:i + 20]) for i in range(0, len(obs), 20)]
    prods = unique(vstack(tabs), keys="productFilename")
    keep = np.array([(p == LEVELS[level]) and ("_nrs1_" in f) and f.startswith("jw")
                     for p, f in zip(_col(prods, "productSubGroupDescription"),
                                     _col(prods, "productFilename"))], dtype=bool)
    prods = prods[keep]
    # PRISM time series only. Stage-3 observation ids look like
    # "jw06456-o001_t001_nirspec_clear-prism-..." (program 06456, observation 001);
    # product files look like "jw06456001001_04101_..." (program, observation, visit).
    prism = set()
    for o in _col(obs, "obs_id"):
        o = str(o).lower()
        if "prism" in o and "-o" in o:
            prog, rest = o[2:7], o.split("-o")[1]
            prism.add((prog, rest[:3]))
    prods = prods[np.array([(f[2:7], f[7:10]) in prism for f in _col(prods, "productFilename")],
                           dtype=bool)]
    return prods


def target_path(outdir, level, fname):
    prog = fname[2:7]                       # e.g. 06456
    visit = fname.split("_")[0]             # e.g. jw06456002001
    return os.path.join(outdir, level, prog, visit, fname)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only report sizes and free disk space")
    ap.add_argument("--verify", action="store_true", help="compare local files with MAST sizes")
    ap.add_argument("--level", choices=LEVELS, help="data level to download or verify")
    ap.add_argument("--programs", nargs="+", default=["1331", "6456", "9256"])
    ap.add_argument("--outdir", default="./jwst_data")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    levels = [a.level] if a.level else list(LEVELS)
    for lev in levels:
        prods = find_products(a.programs, lev)
        if prods is None or len(prods) == 0:
            print(f"{lev}: no products found"); continue
        sizes = {}
        for f, s in zip(prods["productFilename"], prods["size"]):
            sizes.setdefault(f[2:7], []).append(int(s))
        total = sum(sum(v) for v in sizes.values())
        print(f"\n{lev.upper()} (NRS1, PRISM): {len(prods)} files, {total/1e9:.1f} GB")
        for p, v in sorted(sizes.items()):
            print(f"   GO {int(p):5d}: {len(v):3d} files, {sum(v)/1e9:7.1f} GB")
        if a.check:
            continue
        if a.verify:
            ok = 0
            for f, s in zip(prods["productFilename"], prods["size"]):
                path = target_path(a.outdir, lev, f)
                good = os.path.exists(path) and os.path.getsize(path) == int(s)
                ok += good
                if not good:
                    print("   MISSING/INCOMPLETE", f)
            print(f"   {ok}/{len(prods)} files complete")
            continue
        free = shutil.disk_usage(a.outdir).free
        print(f"   free disk space at {a.outdir}: {free/1e9:.1f} GB")
        todo = [(f, int(s)) for f, s in zip(prods["productFilename"], prods["size"])
                if not (os.path.exists(target_path(a.outdir, lev, f))
                        and os.path.getsize(target_path(a.outdir, lev, f)) == int(s))]
        need = sum(s for _, s in todo)
        print(f"   still to download: {len(todo)} files, {need/1e9:.1f} GB")
        if need > free * 0.95:
            sys.exit("   Not enough free disk space. Free some space or use --outdir on a larger disk.")
        for i, (f, s) in enumerate(todo, 1):
            path = target_path(a.outdir, lev, f)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            uri = prods["dataURI"][list(prods["productFilename"]).index(f)]
            print(f"   [{i}/{len(todo)}] {f} ({s/1e9:.2f} GB)", flush=True)
            Observations.download_file(uri, local_path=path, cache=False)
            if os.path.getsize(path) != s:
                print(f"   WARNING: size mismatch for {f}; run the command again to retry")
    print("\nDone.")


if __name__ == "__main__":
    main()
