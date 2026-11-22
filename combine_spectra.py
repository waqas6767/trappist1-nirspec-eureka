#!/usr/bin/env python
"""Combine the per-visit transmission spectra of TRAPPIST-1 e, with and without
correction for stellar contamination using TRAPPIST-1 b.

Input: spectra_summary.csv written by fit_spectra.py. Products:

  e_uncorrected      all visits of e, combined
  b_combined         all visits of b, combined
  e_bcorr_ratio      visits with both planets: D_e(l) / eps(l), where
                     eps(l) = D_b(l) / <D_b> measures the wavelength-dependent
                     stellar contamination in the same visit, assuming b has no
                     detectable atmosphere (the close-transit method of Allen et
                     al. 2026, AJ 171, 105); then combined
  e_bcorr_diff       visits with both planets: D_e(l) x Dbar_b(l) / D_b(l), which
                     removes only the visit-to-visit changes in contamination
                     measured by b, not b's mean spectral shape; then combined

Visits are combined as weighted means after removing each visit's offset from
the global mean depth (achromatic offsets from stellar variability or from the
white-light baseline do not then add scatter); errors are inflated by
sqrt(reduced chi-square) of the visits in each bin when it exceeds 1.

For each product the script reports the chi-square of a flat line (all bins and
bins above 0.7 um) and writes a CSV and a figure. With --dreams-dir, the
spectra of the GO 1331 visits are compared bin by bin with the published DREAMS
spectra of the same visits (Espinoza et al. 2025, files JWST_PRISM_Espinoza<NNN>.txt).

Usage
    python combine_spectra.py --drive WORKDIR
    python combine_spectra.py --drive WORKDIR --dreams-dir path/to/espinoza+2025/appendix-figure1/spectra
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy.stats import chi2 as chi2_dist


def load_summary(path):
    rows = list(csv.DictReader(open(path)))
    visits = sorted({r["visit"] for r in rows})
    wl = None
    out = {}
    rejected = []
    for v in visits:
        rv = sorted([r for r in rows if r["visit"] == v], key=lambda r: int(r["bin"]))
        w = np.array([float(r["wave_mid"]) for r in rv])
        wl = w if wl is None else wl
        if w.size != wl.size or np.max(np.abs(w - wl)) > 5e-3:
            sys.exit(f"{v}: wavelength bins differ from the other visits")
        d = {}
        for pl in ("b", "e"):
            dep = np.array([float(r[f"depth_ppm_{pl}"]) if r[f"depth_ppm_{pl}"] else np.nan for r in rv])
            err = np.array([float(r[f"depth_err_ppm_{pl}"]) if r[f"depth_err_ppm_{pl}"] else np.nan for r in rv])
            rp = np.array([float(r[f"rp_{pl}"]) if r[f"rp_{pl}"] else np.nan for r in rv])
            if not np.isfinite(dep).any():
                continue
            # reject failed bin fits: radius at a bound, non-positive or non-finite errors,
            # and errors or depths far outside the rest of the visit's spectrum
            bad = ~np.isfinite(dep) | ~np.isfinite(err) | (err <= 0) | (rp <= 1e-3) | (rp >= 0.299)
            ok = ~bad
            if ok.sum() > 5:
                med_e = np.median(err[ok])
                med_d = np.median(dep[ok])
                mad = 1.4826 * np.median(np.abs(dep[ok] - med_d))
                bad |= (err > 20 * med_e) | (np.abs(dep - med_d) > 10 * max(mad, med_e))
            if bad.any():
                rejected.append(f"{v} {pl}: {int(bad.sum())} bin(s) rejected "
                                f"({', '.join(f'{w[i]:.2f}' for i in np.where(bad)[0])} um)")
            dep, err = np.where(bad, np.nan, dep), np.where(bad, np.nan, err)
            d[pl] = (dep, err)
        out[v] = d
    wlo = np.array([float(r["wave_low"]) for r in rows if r["visit"] == visits[0]])
    whi = np.array([float(r["wave_high"]) for r in rows if r["visit"] == visits[0]])
    for line in rejected:
        print("  " + line)
    return wl, wlo, whi, out


def combine(S, E, remove_offsets=True):
    """Weighted mean of spectra (n_visit x n_bin), NaN-aware. Returns mean, error,
    and the reduced chi-square of the visits about the mean (all bins)."""
    S, E = np.array(S, float), np.array(E, float)
    Wt = np.where(np.isfinite(S) & np.isfinite(E) & (E > 0), 1 / E ** 2, 0.0)
    S0 = np.where(Wt > 0, S, 0.0)
    if remove_offsets and len(S) > 1:
        g = np.sum(S0 * Wt) / np.sum(Wt)
        vm = np.sum(S0 * Wt, axis=1) / np.sum(Wt, axis=1)
        S0 = np.where(Wt > 0, S0 - (vm - g)[:, None], 0.0)
    sw = np.sum(Wt, axis=0)
    mean = np.where(sw > 0, np.sum(S0 * Wt, axis=0) / np.where(sw > 0, sw, 1), np.nan)
    err = np.where(sw > 0, 1 / np.sqrt(np.where(sw > 0, sw, 1)), np.nan)
    nvis = np.sum(Wt > 0, axis=0)
    scat = np.nan
    if len(S) > 1:
        c = np.sum(np.where(Wt > 0, (S0 - mean) ** 2 * Wt, 0.0), axis=0)
        chi2r = np.where(nvis > 1, c / np.maximum(nvis - 1, 1), 1.0)
        err = err * np.sqrt(np.maximum(chi2r, 1.0))
        scat = np.sum(c) / max(np.sum(np.maximum(nvis - 1, 0)), 1)
    return mean, err, scat


def flat_chi2(d, e):
    ok = np.isfinite(d) & np.isfinite(e)
    w = 1 / e[ok] ** 2
    lev = np.sum(d[ok] * w) / np.sum(w)
    c = np.sum((d[ok] - lev) ** 2 * w)
    dof = ok.sum() - 1
    return c, dof, chi2_dist.sf(c, dof), lev


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", default=os.environ.get("TRAPPIST1_DATA", "."))
    ap.add_argument("--spectra-dir", help="default <drive>/analysis/spectra")
    ap.add_argument("--out", help="default <drive>/analysis/combined")
    ap.add_argument("--exclude-visits", nargs="+", default=[], help="visits to leave out of all products")
    ap.add_argument("--dreams-dir", help="folder with the DREAMS per-visit spectra (JWST_PRISM_Espinoza<NNN>.txt)")
    a = ap.parse_args()
    drive = os.path.abspath(os.path.expanduser(a.drive))
    sdir = a.spectra_dir or os.path.join(drive, "analysis", "spectra")
    outdir = a.out or os.path.join(drive, "analysis", "combined")
    os.makedirs(outdir, exist_ok=True)
    wl, wlo, whi, V = load_summary(os.path.join(sdir, "spectra_summary.csv"))
    for v in a.exclude_visits:
        V.pop(v, None)

    e_vis = [v for v in V if "e" in V[v]]
    b_vis = [v for v in V if "b" in V[v]]
    eb_vis = [v for v in V if "e" in V[v] and "b" in V[v]]
    prods = {}
    m, e, s = combine([V[v]["e"][0] for v in e_vis], [V[v]["e"][1] for v in e_vis])
    prods["e_uncorrected"] = (m, e, s, e_vis)
    if b_vis:
        mb, ebb, sb = combine([V[v]["b"][0] for v in b_vis], [V[v]["b"][1] for v in b_vis])
        prods["b_combined"] = (mb, ebb, sb, b_vis)
    if eb_vis:
        R, RE, Dd, DdE = [], [], [], []
        mb2, eb2, _ = combine([V[v]["b"][0] for v in eb_vis], [V[v]["b"][1] for v in eb_vis])
        for v in eb_vis:
            De, Ee = V[v]["e"]
            Db, Eb = V[v]["b"]
            ok = np.isfinite(Db) & np.isfinite(Eb)
            wmean = np.sum(Db[ok] / Eb[ok] ** 2) / np.sum(1 / Eb[ok] ** 2)
            eps, eps_e = Db / wmean, Eb / wmean
            eps = np.where(eps > 0.2, eps, np.nan)       # b depth must be physical to correct e
            r = De / eps
            R.append(r)
            RE.append(np.abs(r) * np.sqrt((Ee / De) ** 2 + (eps_e / eps) ** 2))
            dd = De * mb2 / np.where(Db > 0.2 * wmean, Db, np.nan)
            Dd.append(dd)
            DdE.append(np.abs(dd) * np.sqrt((Ee / De) ** 2 + (Eb / Db) ** 2))
        mr, er, sr = combine(R, RE)
        prods["e_bcorr_ratio"] = (mr, er, sr, eb_vis)
        md, ed, sd = combine(Dd, DdE)
        ed = np.hypot(ed, md * eb2 / mb2)        # error of b's mean shape, common to all visits, added once
        prods["e_bcorr_diff"] = (md, ed, sd, eb_vis)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(prods), 1, figsize=(8, 2.8 * len(prods)), sharex=True, squeeze=False)
    print(f"{'product':16s} {'visits':>6s} {'visit scatter':>13s} {'flat chi2 (all)':>18s} "
          f"{'flat chi2 (>0.7um)':>20s} {'median err':>10s}")
    for ax, (name, (m, e, s, vis)) in zip(axs[:, 0], prods.items()):
        c1, d1, p1, lev = flat_chi2(m, e)
        k = wl > 0.7
        c2, d2, p2, _ = flat_chi2(m[k], e[k])
        print(f"{name:16s} {len(vis):6d} {s:13.2f} {c1:8.1f}/{d1:<3d} p={p1:.3f} {c2:9.1f}/{d2:<3d} p={p2:.3f} "
              f"{np.nanmedian(e):7.0f} ppm")
        with open(os.path.join(outdir, f"trappist1_{name}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["wave_low_um", "wave_high_um", "wave_mid_um", "depth_ppm", "err_ppm"])
            for row in zip(wlo, whi, wl, m, e):
                w.writerow([f"{row[0]:.4f}", f"{row[1]:.4f}", f"{row[2]:.4f}", f"{row[3]:.1f}", f"{row[4]:.1f}"])
        ax.errorbar(wl, m, e, [wl - wlo, whi - wl], fmt="o", ms=3, lw=0.8,
                    color="C3" if name.startswith("b") else "C0")
        ax.axhline(lev, color="0.5", ls="--", lw=0.8)
        ax.set_ylim(lev - 5 * np.nanmedian(e), lev + 5 * np.nanmedian(e))
        ax.set_ylabel("depth (ppm)")
        ax.set_title(f"{name}: {len(vis)} visits, flat-line chi2 {c1:.0f}/{d1} (p={p1:.2f})", fontsize=9)
    axs[-1, 0].set_xlabel("wavelength (µm)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "combined_spectra.png"), dpi=150)
    plt.close(fig)

    if a.dreams_dir:
        compare_dreams(a.dreams_dir, wl, V, outdir)
    print(f"Results: {outdir}")


def compare_dreams(ddir, wl, V, outdir):
    """Bin-by-bin comparison of the GO 1331 spectra with DREAMS (Espinoza et al. 2025)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vis = [v for v in sorted(V) if v.startswith("jw01331") and "e" in V[v]]
    if not vis:
        return
    fig, axs = plt.subplots(len(vis), 1, figsize=(8, 2.6 * len(vis)), sharex=True, squeeze=False)
    print("Comparison with DREAMS (Espinoza et al. 2025), per visit:")
    for ax, v in zip(axs[:, 0], vis):
        f = os.path.join(ddir, f"JWST_PRISM_Espinoza{v[7:10]}.txt")
        if not os.path.exists(f):
            continue
        dd = np.loadtxt(f)
        wd, dep, er = dd[:, 0], dd[:, 3], dd[:, 4]
        De, Ee = V[v]["e"]
        idx = np.array([np.argmin(np.abs(wd - x)) for x in wl])
        match = np.abs(wd[idx] - wl) < 5e-3
        d2, e2 = dep[idx], er[idx]
        ok = match & np.isfinite(De) & np.isfinite(d2) & np.isfinite(Ee) & np.isfinite(e2)
        diff, derr = De[ok] - d2[ok], np.hypot(Ee[ok], e2[ok])
        w = 1 / derr ** 2
        off = np.sum(diff * w) / np.sum(w)
        c = np.sum((diff - off) ** 2 * w)
        print(f"  {v}: {ok.sum()} bins, mean offset (ours - DREAMS) {off:+.0f} ppm, "
              f"chi2 of the difference after the offset {c:.1f}/{ok.sum()-1} "
              f"(p={chi2_dist.sf(c, ok.sum()-1):.3f}); median errors ours {np.median(Ee[ok]):.0f}, "
              f"DREAMS {np.median(e2[ok]):.0f} ppm")
        ax.errorbar(wl[ok] - 0.008, De[ok], Ee[ok], fmt="o", ms=3, lw=0.8, label="this work")
        ax.errorbar(wd[idx][ok] + 0.008, d2[ok], e2[ok], fmt="s", ms=3, lw=0.8, label="DREAMS (Espinoza et al. 2025)")
        ax.set_ylabel("depth (ppm)")
        ax.set_title(f"{v}: offset {off:+.0f} ppm, difference chi2 {c:.0f}/{ok.sum()-1}", fontsize=9)
        ax.legend(fontsize=7)
    axs[-1, 0].set_xlabel("wavelength (µm)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "comparison_dreams.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
