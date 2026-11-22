#!/usr/bin/env python
"""Transmission spectra of TRAPPIST-1 b and e for each visit, from the Eureka!
Stage 4 spectroscopic light curves (67 wavelength bins).

Run fit_whitelight_gp.py first: this script takes from it, for each visit,

  * the mid-transit times of b and e (held fixed here), and
  * the white-light systematics S(t) = baseline + GP / transit, evaluated at
    every integration from the maximum-probability sample of the white-light
    posterior (the "common mode").

Each spectroscopic light curve is divided by S(t), which removes the
achromatic part of the instrumental and stellar time-correlated noise
(--common-mode model). With --common-mode data, the light curve is instead
divided by the white light curve over its transit model (the classic
divide-white method), which also removes achromatic white noise shared by
all bins but adds the white-light noise to each bin.

The corrected light curve of each bin is then binned to --bin-s seconds and
fitted by least squares with
    free  - Rp/R* of each planet in the visit, quadratic limb darkening
            (Kipping q1, q2, shared by the planets, uniform in 0-1), and a
            linear baseline per continuous segment;
    fixed - mid-transit times (white light), period, a/R*, inclination
            (Agol et al. 2021).
The same integrations as in the white-light fit are used (transits of other
planets, flares and outliers removed); in addition, points more than --clip
robust sigma from the first fit of each bin are rejected. Uncertainties are
scaled by sqrt(reduced chi-square) when above 1 and by the time-averaging
beta factor of each bin.

Outputs, in --out (default <drive>/analysis/spectra):
    <visit>_spectrum.csv     one row per bin: wavelengths, Rp/R*, depth and error per planet
    <visit>_spectrum.png     the spectra of b and e in the visit
    spectra_summary.csv      all visits and bins in one table (input to combine_spectra.py)

Usage
    python fit_spectra.py --drive WORKDIR                      # all visits with a GP white-light fit
    python fit_spectra.py --drive WORKDIR --visits jw06456002001 --common-mode data
"""
import argparse
import csv
import glob
import os
import sys
import traceback
import warnings

import numpy as np
from scipy.optimize import least_squares

import fit_whitelight as W
import fit_whitelight_gp as G


def read_spectral_lcs(path):
    """Spectroscopic light curves from a Stage 4 LCData file."""
    import xarray as xr
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        t = np.asarray(ds["time"].values, float)
        F = np.asarray(ds["data"].values, float)            # (n_wave, n_time)
        E = np.asarray(ds["err"].values, float)
        M = np.asarray(ds["mask"].values, bool) if "mask" in ds else np.zeros(F.shape, bool)
        wlo = np.asarray(ds["wave_low"].values, float)
        whi = np.asarray(ds["wave_hi"].values, float)
    return t, F, E, M, wlo, whi


def white_common_mode(drive, visit, gp_dir, npoly, clip, exclude, bin_s, mode):
    """Common-mode systematics S(t) at every integration, the mask of usable
    integrations, the fixed mid-times and the white-light segments."""
    prep = G.prepare(drive, visit, npoly, clip, exclude, bin_s)
    if isinstance(prep, str):
        raise RuntimeError(prep)
    chainfile = os.path.join(gp_dir, f"{visit}_chain.npz")
    if not os.path.exists(chainfile):
        raise RuntimeError(f"no white-light GP chain ({chainfile}); run fit_whitelight_gp.py first")
    model = prep["model"]
    p_best = G.best_params(model, chainfile)
    names = model.names
    t, use = prep["t"], prep["use"]
    S, tr = model.systematics(p_best, t, prep["drift"], prep["segs"])
    if mode == "data":
        S = prep["f"] / tr
    ix = {n: i for i, n in enumerate(names)}
    t0 = {pl: p_best[ix[f"t0_{pl}"]] for pl in model.mean.pls}
    return S, use, t0, prep["segs"], prep["notes"]


def bin_arrays(t, f, e, bin_s):
    """Bin a light curve in bins of bin_s seconds that never span a gap."""
    edges = np.r_[0, np.where(np.diff(t) * 86400 > bin_s)[0] + 1, t.size]
    tb, fb, eb = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        n = max(1, int(round((t[b - 1] - t[a]) * 86400 / bin_s)))
        for ix in np.array_split(np.arange(a, b), n):
            if ix.size:
                tb.append(t[ix].mean())
                fb.append(f[ix].mean())
                eb.append(np.sqrt(np.sum(e[ix] ** 2)) / ix.size)
    return np.array(tb), np.array(fb), np.array(eb)


class BinModel:
    """Transits with fixed mid-times x linear baseline per segment."""

    def __init__(self, t, t0, segs):
        self.t, self.t0, self.pls = t, t0, sorted(t0)
        self.sel, self.tc = [], []
        for a, b in segs:
            span = max(b - a, 1e-3)
            self.sel.append((t >= a - 1e-6) & (t <= b + 1e-6))
            self.tc.append((t - 0.5 * (a + b)) / span)
        self.nseg = len(segs)

    def p0_bounds(self):
        p0 = [np.sqrt(W.PLANETS[p]["depth"]) for p in self.pls] + [0.36, 0.3] + [1.0, 0.0] * self.nseg
        lo = [0.0] * len(self.pls) + [0.0, 0.0] + [-np.inf] * (2 * self.nseg)
        hi = [0.3] * len(self.pls) + [1.0, 1.0] + [np.inf] * (2 * self.nseg)
        return np.array(p0), (np.array(lo), np.array(hi))

    def __call__(self, p):
        n = len(self.pls)
        u1, u2 = W.q_to_u(p[n], p[n + 1])
        m = np.ones_like(self.t)
        for i, pl in enumerate(self.pls):
            if p[i] > 0:
                m *= W.transit(self.t, self.t0[pl], p[i], pl, u1, u2)
        base = np.ones_like(self.t)
        for k in range(self.nseg):
            c0, c1 = p[n + 2 + 2 * k], p[n + 3 + 2 * k]
            base = np.where(self.sel[k], c0 + c1 * self.tc[k], base)
        return m * base


def fit_bin(t, f, e, t0, segs, clip, bin_s):
    """Fit one spectroscopic light curve (already divided by the common mode)."""
    ok = np.isfinite(f) & np.isfinite(e) & (e > 0)
    t, f, e = t[ok], f[ok], e[ok]
    norm = np.median(f)
    f, e = f / norm, e / norm
    for it in range(2):
        tb, fb, eb = bin_arrays(t, f, e, bin_s)
        mod = BinModel(tb, t0, segs)
        p0, bounds = mod.p0_bounds()
        sol = least_squares(lambda p: (mod(p) - fb) / eb, p0, bounds=bounds, x_scale="jac")
        if it == 0:
            # reject outliers in the unbinned data relative to the first fit
            r = f - BinModel(t, t0, segs)(sol.x)
            sig = 1.4826 * np.median(np.abs(r - np.median(r)))
            keep = np.abs(r) < clip * sig
            t, f, e = t[keep], f[keep], e[keep]
    resid = fb - mod(sol.x)
    chi2r = np.sum((resid / eb) ** 2) / max(resid.size - sol.x.size, 1)
    beta = W.beta_factor(tb, resid)
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac)
        perr = np.sqrt(np.diag(cov)) * np.sqrt(max(chi2r, 1.0)) * beta
    except np.linalg.LinAlgError:
        perr = np.full(sol.x.size, np.nan)
    return sol.x, perr, np.std(resid), beta, chi2r


def fit_visit_spectrum(drive, visit, gp_dir, outdir, a, exclude):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    path = W.find_lcdata(drive, visit)
    S, use, t0, segs, notes = white_common_mode(drive, visit, gp_dir, a.npoly, a.clip, exclude,
                                                a.bin_s, a.common_mode)
    t, F, E, M, wlo, whi = read_spectral_lcs(path)
    pls = sorted(t0)
    rows = []
    for k in range(F.shape[0]):
        good = use & ~M[k] & np.isfinite(F[k]) & np.isfinite(E[k])
        row = dict(visit=visit, program=f"GO {int(visit[2:7])}", bin=k,
                   wave_low=f"{wlo[k]:.4f}", wave_high=f"{whi[k]:.4f}",
                   wave_mid=f"{0.5*(wlo[k]+whi[k]):.4f}")
        if good.sum() < 0.5 * use.sum():
            for pl in ("b", "e"):
                row.update({f"rp_{pl}": "", f"rp_err_{pl}": "", f"depth_ppm_{pl}": "", f"depth_err_ppm_{pl}": ""})
            row.update(u1="", u2="", rms_binned_ppm="", beta="", chi2r="", note="too many masked points")
            rows.append(row)
            continue
        p, pe, rms, beta, chi2r = fit_bin(t[good], F[k][good] / S[good], E[k][good] / S[good],
                                          t0, segs, a.clip, a.bin_s)
        u1, u2 = W.q_to_u(p[len(pls)], p[len(pls) + 1])
        for pl in ("b", "e"):
            if pl in pls:
                i = pls.index(pl)
                row.update({f"rp_{pl}": f"{p[i]:.5f}", f"rp_err_{pl}": f"{pe[i]:.5f}",
                            f"depth_ppm_{pl}": f"{p[i]**2*1e6:.1f}",
                            f"depth_err_ppm_{pl}": f"{2*p[i]*pe[i]*1e6:.1f}"})
            else:
                row.update({f"rp_{pl}": "", f"rp_err_{pl}": "", f"depth_ppm_{pl}": "", f"depth_err_ppm_{pl}": ""})
        row.update(u1=f"{u1:.3f}", u2=f"{u2:.3f}", rms_binned_ppm=f"{rms*1e6:.0f}",
                   beta=f"{beta:.2f}", chi2r=f"{chi2r:.2f}",
                   note="; ".join(f"{pl}: {n}" for pl, n in notes.items()))
        rows.append(row)

    with open(os.path.join(outdir, f"{visit}_spectrum.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(len(pls), 1, figsize=(8, 3.2 * len(pls)), sharex=True, squeeze=False)
    for j, pl in enumerate(pls):
        x = np.array([float(r["wave_mid"]) for r in rows if r[f"depth_ppm_{pl}"]])
        y = np.array([float(r[f"depth_ppm_{pl}"]) for r in rows if r[f"depth_ppm_{pl}"]])
        ye = np.array([float(r[f"depth_err_ppm_{pl}"]) for r in rows if r[f"depth_ppm_{pl}"]])
        xe = np.array([[float(r["wave_mid"]) - float(r["wave_low"]) for r in rows if r[f"depth_ppm_{pl}"]],
                       [float(r["wave_high"]) - float(r["wave_mid"]) for r in rows if r[f"depth_ppm_{pl}"]]])
        ok = np.isfinite(y) & np.isfinite(ye) & (ye > 0) & (y > 0)   # skip failed bins in the plot
        ax[j, 0].errorbar(x[ok], y[ok], ye[ok], xe[:, ok], fmt="o", ms=3,
                          color="C0" if pl == "e" else "C3", lw=0.8)
        if ok.sum() > 2:
            wm = np.sum(y[ok] / ye[ok] ** 2) / np.sum(1 / ye[ok] ** 2)
            ax[j, 0].axhline(wm, color="0.5", lw=0.8, ls="--")
            ax[j, 0].set_ylim(wm - 4 * np.median(ye[ok]), wm + 4 * np.median(ye[ok]))
        ax[j, 0].set_ylabel(f"{pl}: (Rp/R*)$^2$ (ppm)")
    ax[-1, 0].set_xlabel("wavelength (µm)")
    ax[0, 0].set_title(f"{visit}  (common mode: {a.common_mode})", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{visit}_spectrum.png"), dpi=150)
    plt.close(fig)
    med = {pl: np.nanmedian([float(r[f"depth_err_ppm_{pl}"]) for r in rows if r[f"depth_ppm_{pl}"]]) for pl in pls}
    msg = f"{visit}: {len(rows)} bins; median depth error " + ", ".join(f"{pl} {v:.0f} ppm" for pl, v in med.items())
    return rows, msg


def _worker(args):
    drive, visit, gp_dir, outdir, a, exclude = args
    try:
        return fit_visit_spectrum(drive, visit, gp_dir, outdir, a, exclude)
    except Exception:
        return [], f"{visit}: FAILED\n{traceback.format_exc()}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", default=os.environ.get("TRAPPIST1_DATA", "."))
    ap.add_argument("--visits", nargs="+")
    ap.add_argument("--gp-dir", help="folder of the white-light GP fits (default <drive>/analysis/whitelight_gp)")
    ap.add_argument("--out")
    ap.add_argument("--common-mode", choices=["model", "data"], default="model")
    ap.add_argument("--npoly", type=int, default=1, help="must match the white-light GP fit (default 1)")
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--bin-s", type=float, default=30.0)
    ap.add_argument("--exclude", nargs="+", default=[], metavar="VISIT:START-END")
    ap.add_argument("--processes", type=int, default=1)
    a = ap.parse_args()
    drive = os.path.abspath(os.path.expanduser(a.drive))
    gp_dir = a.gp_dir or os.path.join(drive, "analysis", "whitelight_gp")
    outdir = a.out or os.path.join(drive, "analysis", "spectra")
    os.makedirs(outdir, exist_ok=True)
    visits = a.visits or sorted(os.path.basename(p).replace("_chain.npz", "")
                                for p in glob.glob(os.path.join(gp_dir, "jw*_chain.npz")))
    if not visits:
        sys.exit(f"No white-light GP fits found in {gp_dir}; run fit_whitelight_gp.py first")
    exclude = {k: list(v) for k, v in W.EXCLUDE.items()}
    for item in a.exclude:
        v, rng = item.split(":")
        t1, t2 = (float(x) for x in rng.split("-"))
        exclude.setdefault(v, []).append((t1, t2, "excluded on the command line"))

    jobs = [(drive, v, gp_dir, outdir, a, exclude) for v in visits]
    print(f"Fitting the spectra of {len(jobs)} visits ({a.common_mode} common mode), "
          f"{a.processes} at a time ...", flush=True)
    rows = []
    if a.processes > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.processes) as pool:
            for r, msg in pool.imap(_worker, jobs):
                print(msg, flush=True)
                rows += r
    else:
        for job in jobs:
            r, msg = _worker(job)
            print(msg, flush=True)
            rows += r
    if not rows:
        sys.exit("No spectra were fitted.")
    with open(os.path.join(outdir, "spectra_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Spectra: {outdir}")


if __name__ == "__main__":
    main()
