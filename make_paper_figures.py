#!/usr/bin/env python
"""Make the figures of the paper from the outputs of this pipeline.

Every figure is rebuilt from files written by the analysis scripts, so the
figures can be reproduced from the raw data by running, in order,
run_reduction.py, fit_whitelight_gp.py, fit_spectra.py, combine_spectra.py
and this script.

Figures (PDF, written to --out):
  lightcurves.pdf        white-light light curves of two visits with the GP model,
                         detrended transits and residuals
                         [needs the Stage 4 outputs and analysis/whitelight_gp/]
  radii.pdf              white-light Rp/R* of b and e for every visit
                         [analysis/whitelight_gp/whitelight_gp_fits.csv]
  oc.pdf                 transit-timing residuals of b and e  [same file]
  comparison_dreams.pdf  spectra of the four DREAMS visits compared with the
                         published reduction of Espinoza et al. (2025)
                         [analysis/spectra/spectra_summary.csv and --dreams-dir]
  combined_spectra.pdf   combined spectra of e and b  [analysis/combined/]
  model_comparison.pdf   combined spectra of e with grid models, and model
                         weights versus CO2 pressure  [--models]

Usage
    python make_paper_figures.py --drive WORKDIR --dreams-dir DIR --models DIR
    python make_paper_figures.py --drive WORKDIR --only radii oc combined
Options
    --drive       working folder (contains reduction/ and analysis/)
    --out         output folder (default <drive>/analysis/paper_figures)
    --dreams-dir  folder with JWST_PRISM_Espinoza<NNN>.txt (DREAMS per-visit spectra)
    --dreams-combined  DREAMS decontaminated combined spectrum (columns: wavelength,
                  depth, error in ppm), used for the reference model weights
    --models      folder with the model spectra binned to the 67 bins
                  (model_NNNN_jwst_fit.csv) and MC_2000_master_results_flatspec.csv
    --lc-visits   the two visits shown in lightcurves.pdf
    --only        make only some figures: lc radii oc dreams combined models
"""
import argparse
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "savefig.bbox": "tight"})
PROG_COLOR = {"GO 1331": "#d95f02", "GO 6456": "#1b9e77", "GO 9256": "#7570b3"}
PLANET_COLOR = {"b": "#c0392b", "e": "#2471a3"}
# Spitzer 4.5 um radius ratios (Agol et al. 2021), shown for reference
AGOL = {"b": (0.08590, 0.00037), "e": (0.07079, 0.00055)}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def weighted_mean(x, e):
    ok = np.isfinite(x) & np.isfinite(e) & (e > 0)
    w = 1 / e[ok] ** 2
    return np.sum(x[ok] * w) / np.sum(w), np.sum(w) ** -0.5


def bin_series(t, y, width_min):
    """Mean of y in bins of width_min minutes (bins never span a gap)."""
    edges = np.arange(t.min(), t.max() + width_min / 1440, width_min / 1440)
    idx = np.digitize(t, edges)
    keys = np.unique(idx)
    return (np.array([t[idx == k].mean() for k in keys]),
            np.array([y[idx == k].mean() for k in keys]))


def gaps_to_nan(t, y, gap_min=5.0):
    br = np.where(np.diff(t) * 1440 > gap_min)[0] + 1
    return np.insert(t, br, np.nan), np.insert(np.asarray(y, float), br, np.nan)


def visit_label(v):
    return f"{v[2:7]}-{v[7:10]}"


# ---------------------------------------------------------------------------
# 1. light curves
# ---------------------------------------------------------------------------
def fig_lightcurves(drive, gp_dir, visits, out):
    import fit_whitelight as W
    import fit_whitelight_gp as G
    fig, axs = plt.subplots(3, len(visits), figsize=(7.2, 6.4), sharex="col",
                            gridspec_kw=dict(height_ratios=[1.3, 1.3, 0.7], hspace=0.06, wspace=0.28),
                            squeeze=False)
    for j, visit in enumerate(visits):
        exclude = {k: list(v) for k, v in W.EXCLUDE.items()}
        prep = G.prepare(drive, visit, 1, 5.0, exclude, 30.0)
        if isinstance(prep, str):
            raise RuntimeError(prep)
        model = prep["model"]
        p = G.best_params(model, os.path.join(gp_dir, f"{visit}_chain.npz"))
        tb, fb = prep["tb"], prep["fb"]
        tr, base, mu = model.predict(p)
        ix = {n: i for i, n in enumerate(model.names)}
        t0 = {pl: p[ix[f"t0_{pl}"]] for pl in model.mean.pls}
        tref = min(t0.values())
        x = (tb - tref) * 24                                    # hours from the first transit
        det = (fb - mu) / base
        resid = (fb - tr * base - mu) * 1e6
        a0, a1, a2 = axs[:, j]
        a0.plot(x, fb, ".", ms=1.5, color="0.65", rasterized=True)
        xb, yb = bin_series(tb, fb, 5)
        a0.plot((xb - tref) * 24, yb, "o", ms=2.2, color="0.25")
        a0.plot(*gaps_to_nan(x, tr * base + mu), "-", color="#e67e22", lw=1.1, zorder=5, label="transit $\\times$ baseline + GP")
        a0.plot(*gaps_to_nan(x, base), "--", color="0.35", lw=0.8, zorder=5, label="baseline")
        for lo, hi, why in prep["windows"]:
            for a in (a0, a1, a2):
                a.axvspan((lo - tref) * 24, (hi - tref) * 24, color="#f5b7b1", alpha=0.5, lw=0)
        for lab, mid, fw, dp in prep["dips"]:
            if lab == "other":
                hw = (0.8 * fw + 0.25) / 24
                for a in (a0, a1, a2):
                    a.axvspan((mid - hw - tref) * 24, (mid + hw - tref) * 24, color="0.9", lw=0)
        a1.plot(x, det, ".", ms=1.5, color="0.65", rasterized=True)
        xb, yb = bin_series(tb, det, 5)
        a1.plot((xb - tref) * 24, yb, "o", ms=2.2, color="0.25")
        for pl in model.mean.pls:
            # transit model of each planet on its own, for the labels and colors
            u1, u2 = W.q_to_u(p[ix["q1"]], p[ix["q2"]])
            m = W.transit(tb, t0[pl], p[ix[f"rp_{pl}"]], pl, u1, u2)
            sel = m < 1
            a1.plot(*gaps_to_nan(x[sel], m[sel]), "-", color=PLANET_COLOR[pl], lw=1.4, zorder=5)
            a1.text((t0[pl] - tref) * 24, m.min() - 0.0007, pl,
                    color=PLANET_COLOR[pl], ha="center", va="top", fontsize=10, fontweight="bold")
        a2.plot(x, resid, ".", ms=1.5, color="0.65", rasterized=True)
        xb, yb = bin_series(tb, resid, 5)
        a2.plot((xb - tref) * 24, yb, "o", ms=2.2, color="0.25")
        a2.axhline(0, color="#e67e22", lw=0.9, zorder=5)
        a0.set_title(f"{visit} (GO {int(visit[2:7])})", fontsize=9)
        a2.set_xlabel("time from first transit (h)")
        a2.set_ylim(-700, 700)
        lo1 = min(det.min() - 0.0025, 0.990)
        a1.set_ylim(lo1, 1.0015)
        if j == 0:
            a0.set_ylabel("normalized flux")
            a1.set_ylabel("detrended flux")
            a2.set_ylabel("residual (ppm)")
            a0.legend(loc="lower left", frameon=False)
    fig.savefig(os.path.join(out, "lightcurves.pdf"), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. radii and 3. timing
# ---------------------------------------------------------------------------
def fig_radii(rows, out):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(wspace=0.3))
    for ax, pl in zip(axs, "be"):
        r = [x for x in rows if x["planet"] == pl]
        rp = np.array([float(x["rp"]) for x in r])
        er = 0.5 * (np.array([float(x["rp_err_lo"]) for x in r]) + np.array([float(x["rp_err_hi"]) for x in r]))
        full = np.array([not x["note"] for x in r])
        m, me = weighted_mean(rp[full], er[full])
        ref, rerr = AGOL[pl]
        ax.axhspan(ref - rerr, ref + rerr, color="0.88", lw=0, label="Agol et al. (2021), 4.5 µm")
        ax.axhspan(m - me, m + me, color=PLANET_COLOR[pl], alpha=0.2, lw=0)
        ax.axhline(m, color=PLANET_COLOR[pl], lw=0.9, label="weighted mean (this work)")
        for i, x in enumerate(r):
            c = PROG_COLOR[x["program"]]
            ax.errorbar(i, float(x["rp"]), [[float(x["rp_err_lo"])], [float(x["rp_err_hi"])]],
                        fmt="s" if x["note"] else "o", ms=4, color=c, mfc="white" if x["note"] else c, lw=1)
        ax.set_xticks(range(len(r)))
        ax.set_xticklabels([visit_label(x["visit"]) for x in r], rotation=75, fontsize=6.5)
        ax.set_ylabel(r"$R_p/R_\star$")
        ax.set_title(f"TRAPPIST-1 {pl}", fontsize=9)
    for prog, c in PROG_COLOR.items():
        axs[1].plot([], [], "o", color=c, label=prog)
    axs[0].legend(loc="lower left", frameon=False)
    axs[1].legend(loc="lower left", frameon=False, ncol=2)
    fig.savefig(os.path.join(out, "radii.pdf"))
    plt.close(fig)


def fig_oc(rows, out):
    from astropy.time import Time
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 4.4), sharex=True, gridspec_kw=dict(hspace=0.08))
    for ax, pl in zip(axs, "be"):
        r = [x for x in rows if x["planet"] == pl]
        t = np.array([float(x["t0_bmjd"]) for x in r])
        e = np.array([max(float(x["t0_err_min"]), 0.01) for x in r]) / 1440
        P0 = {"b": 1.51085, "e": 6.0998}[pl]
        n = np.round((t - t[np.argmin(np.abs(t - np.median(t)))]) / P0)
        A = np.vstack([np.ones_like(n), n]).T
        w = 1 / e ** 2
        coef = np.linalg.solve(A.T @ (w[:, None] * A), A.T @ (w * t))
        oc = (t - A @ coef) * 1440
        yr = Time(t, format="mjd").decimalyear
        for x, yy, ee, rr in zip(yr, oc, e * 1440, r):
            ax.errorbar(x, yy, ee, fmt="o", ms=3.5, color=PROG_COLOR[rr["program"]], lw=1)
        ax.axhline(0, color="0.5", lw=0.7, ls="--")
        ax.set_ylabel(f"O$-$C, {pl} (min)")
    for prog, c in PROG_COLOR.items():
        axs[1].plot([], [], "o", color=c, label=prog)
    axs[1].legend(frameon=False, loc="upper left")
    axs[1].set_xlabel("year")
    fig.savefig(os.path.join(out, "oc.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. comparison with DREAMS
# ---------------------------------------------------------------------------
def fig_dreams(spec_dir, dreams_dir, out):
    from scipy.stats import chi2 as chi2_dist
    visits = ["jw01331001001", "jw01331002001", "jw01331003001", "jw01331104001"]
    from combine_spectra import load_summary
    # same reading and screening of failed bins as combine_spectra.py
    w, _, _, V = load_summary(os.path.join(spec_dir, "spectra_summary.csv"))
    fig = plt.figure(figsize=(7.2, 7.6))
    outer = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)
    for k, v in enumerate(visits):
        inner = outer[k // 2, k % 2].subgridspec(2, 1, height_ratios=[2.2, 1], hspace=0.05)
        a0 = fig.add_subplot(inner[0])
        a1 = fig.add_subplot(inner[1], sharex=a0)
        if v not in V or "e" not in V[v]:
            print(f"  comparison_dreams: no spectrum of e for {v}, panel left empty")
            a0.text(0.5, 0.5, f"{v}: no spectrum", transform=a0.transAxes, ha="center")
            continue
        d, e = V[v]["e"]
        dd = np.loadtxt(os.path.join(dreams_dir, f"JWST_PRISM_Espinoza{v[7:10]}.txt"))
        idx = np.array([np.argmin(np.abs(dd[:, 0] - x)) for x in w])
        match = np.abs(dd[idx, 0] - w) < 5e-3
        d2, e2 = np.where(match, dd[idx, 3], np.nan), np.where(match, dd[idx, 4], np.nan)
        ok = np.isfinite(d) & np.isfinite(d2) & np.isfinite(e) & np.isfinite(e2)
        diff, derr = d - d2, np.hypot(e, e2)
        off, _ = weighted_mean(diff[ok], derr[ok])
        c2 = np.sum(((diff - off)[ok] / derr[ok]) ** 2)
        dof = ok.sum() - 1
        a0.errorbar(w[ok] + 0.012, d2[ok] + off, e2[ok], fmt="s", ms=2.5, lw=0.7, color="#e67e22",
                    label="Espinoza et al. (2025)")
        a0.errorbar(w[ok] - 0.012, d[ok], e[ok], fmt="o", ms=2.5, lw=0.7, color="#2471a3", label="this work")
        lev, _ = weighted_mean(d[ok], e[ok])
        a0.set_ylim(lev - 1300, lev + 1300)
        a0.text(0.97, 0.06, f"offset {off:+.0f} ppm\n$\\chi^2$ = {c2:.1f} / {dof} (p = {chi2_dist.sf(c2, dof):.2g})",
                transform=a0.transAxes, ha="right", va="bottom", fontsize=7)
        a0.set_title(f"DREAMS visit {v[7:10]} ({v})", fontsize=8)
        a1.errorbar(w[ok], (diff - off)[ok] / derr[ok], 1, fmt="o", ms=2, lw=0.5, color="0.3")
        a1.axhline(0, color="0.5", lw=0.7)
        a1.set_ylim(-5, 5)
        plt.setp(a0.get_xticklabels(), visible=False)
        a1.set_xlim(0.55, 5.35)
        if k % 2 == 0:
            a0.set_ylabel("depth (ppm)")
            a1.set_ylabel("$\\Delta/\\sigma$")
        if k >= 2:
            a1.set_xlabel("wavelength (µm)")
        if k == 0:
            a0.legend(loc="upper right", frameon=False)
    fig.savefig(os.path.join(out, "comparison_dreams.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. combined spectra
# ---------------------------------------------------------------------------
def read_combined(path):
    d = np.genfromtxt(path, delimiter=",", skip_header=1)
    return d[:, 2], d[:, 0], d[:, 1], d[:, 3], d[:, 4]


def fig_combined(comb_dir, out):
    panels = [("e_uncorrected", "TRAPPIST-1 e, 17 visits, uncorrected", PLANET_COLOR["e"]),
              ("b_combined", "TRAPPIST-1 b, 14 visits", PLANET_COLOR["b"]),
              ("e_bcorr_ratio", "e, ratio correction with b (13 visits)", "0.45"),
              ("e_bcorr_diff", "e, differential correction with b (13 visits)", "#1b9e77")]
    fig, axs = plt.subplots(4, 1, figsize=(3.5, 7.6), sharex=True, gridspec_kw=dict(hspace=0.07))
    for ax, (name, label, col) in zip(axs, panels):
        w, lo, hi, d, e = read_combined(os.path.join(comb_dir, f"trappist1_{name}.csv"))
        ok = np.isfinite(d) & np.isfinite(e)
        lev, _ = weighted_mean(d[ok], e[ok])
        ax.errorbar(w[ok], d[ok] - lev, e[ok], [w[ok] - lo[ok], hi[ok] - w[ok]], fmt="o", ms=2.2, lw=0.7, color=col)
        ax.axhline(0, color="0.5", lw=0.7, ls="--")
        ax.set_ylim(-260, 260)
        ax.set_ylabel("depth $-$ mean (ppm)")
        ax.text(0.03, 0.92, label, transform=ax.transAxes, fontsize=7, va="top")
    axs[-1].set_xlim(0.55, 5.35)
    axs[-1].set_xlabel("wavelength (µm)")
    fig.savefig(os.path.join(out, "combined_spectra.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. model comparison
# ---------------------------------------------------------------------------
def grid_chi2(models_dir, w, d, e):
    """Chi-square of every grid model against a spectrum, with a free offset."""
    master = read_csv(os.path.join(models_dir, "MC_2000_master_results_flatspec.csv"))
    pco2 = {int(r["model_id"]): float(r["P_CO2"]) for r in master}
    ok = np.isfinite(d) & np.isfinite(e)
    wt = np.where(ok, 1 / np.where(ok, e, 1) ** 2, 0.0)
    res = []
    for f in sorted(glob.glob(os.path.join(models_dir, "model_*_jwst_fit.csv"))):
        mid = int(os.path.basename(f).split("_")[1])
        t = read_csv(f)
        m = np.array([float(x["model_depth_ppm"]) for x in t])
        if m.size != d.size:
            continue
        r = np.where(ok, d - m, 0.0)
        off = np.sum(r * wt) / np.sum(wt)
        res.append((mid, np.sum((r - off) ** 2 * wt), off, pco2.get(mid, np.nan), m))
    r0 = np.where(ok, d, 0.0)
    flat = np.sum((r0 - np.sum(r0 * wt) / np.sum(wt)) ** 2 * wt)
    return res, flat


def fig_models(comb_dir, models_dir, dreams_combined, out):
    panels = [("e_uncorrected", "17 visits, uncorrected"), ("e_bcorr_diff", "13 visits, differential b-correction")]
    fig, axs = plt.subplots(3, 1, figsize=(3.5, 8.0), gridspec_kw=dict(height_ratios=[1, 1, 0.85], hspace=0.38))
    bins = np.logspace(-3, 0, 19)
    for ax, (name, label) in zip(axs[:2], panels):
        w, lo, hi, d, e = read_combined(os.path.join(comb_dir, f"trappist1_{name}.csv"))
        ok = np.isfinite(d) & np.isfinite(e)
        lev, _ = weighted_mean(d[ok], e[ok])
        res, flat = grid_chi2(models_dir, w, d, e)
        res.sort(key=lambda r: r[1])
        best = res[0]
        rich = [r for r in res if 0.4 < r[3] < 0.6][0]
        ax.errorbar(w[ok], d[ok] - lev, e[ok], [w[ok] - lo[ok], hi[ok] - w[ok]], fmt="o", ms=2.2, lw=0.7,
                    color="k", label=label)
        for r, col, txt in ((best, "#2471a3", "best model"), (rich, "#c0392b", "CO$_2$-rich model")):
            ax.plot(w, r[4] + r[2] - lev, "-", color=col, lw=1.2,
                    label=f"{txt}: CO$_2$ {r[3]:.3f} bar, $\\Delta\\chi^2_{{\\rm flat}}$ = {r[1] - flat:+.1f}")
        ax.axhline(0, color="0.6", lw=0.7, ls="--")
        ax.set_ylim(-260, 300)
        ax.set_xlim(0.55, 5.35)
        ax.set_ylabel("depth $-$ mean (ppm)")
        ax.set_xlabel("wavelength (µm)")
        ax.legend(loc="upper left", frameon=False, fontsize=6.5)
        c = np.array([r[1] for r in res])
        wgt = np.exp(-0.5 * (c - c.min()))
        axs[2].hist([r[3] for r in res], bins=bins, weights=wgt / wgt.sum(), histtype="step", lw=1.4,
                    color="k" if name == "e_uncorrected" else "#1b9e77", label=f"this work, {label.split(', ')[1]}")
    if dreams_combined:
        dc = np.loadtxt(dreams_combined)
        res, _ = grid_chi2(models_dir, dc[:, 0], dc[:, 1], dc[:, 2])
        c = np.array([r[1] for r in res])
        wgt = np.exp(-0.5 * (c - c.min()))
        axs[2].hist([r[3] for r in res], bins=bins, weights=wgt / wgt.sum(), histtype="step", lw=1.4, ls="--",
                    color="#e67e22", label="DREAMS combined spectrum")
    axs[2].set_xscale("log")
    axs[2].set_xlabel("CO$_2$ partial pressure (bar)")
    axs[2].set_ylabel("model weight")
    axs[2].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.savefig(os.path.join(out, "model_comparison.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", default=os.environ.get("TRAPPIST1_DATA", "."))
    ap.add_argument("--out")
    ap.add_argument("--dreams-dir")
    ap.add_argument("--dreams-combined")
    ap.add_argument("--models")
    ap.add_argument("--lc-visits", nargs=2, default=["jw09256006001", "jw06456001001"])
    ap.add_argument("--only", nargs="+", choices=["lc", "radii", "oc", "dreams", "combined", "models"])
    a = ap.parse_args()
    drive = os.path.abspath(os.path.expanduser(a.drive))
    an = os.path.join(drive, "analysis")
    out = a.out or os.path.join(an, "paper_figures")
    os.makedirs(out, exist_ok=True)
    todo = a.only or ["lc", "radii", "oc", "dreams", "combined", "models"]
    wl_csv = os.path.join(an, "whitelight_gp", "whitelight_gp_fits.csv")
    if "lc" in todo:
        fig_lightcurves(drive, os.path.join(an, "whitelight_gp"), a.lc_visits, out)
        print("lightcurves.pdf")
    if "radii" in todo or "oc" in todo:
        rows = read_csv(wl_csv)
        if "radii" in todo:
            fig_radii(rows, out)
            print("radii.pdf")
        if "oc" in todo:
            fig_oc(rows, out)
            print("oc.pdf")
    if "dreams" in todo:
        if not a.dreams_dir:
            print("comparison_dreams.pdf skipped: give --dreams-dir")
        else:
            fig_dreams(os.path.join(an, "spectra"), a.dreams_dir, out)
            print("comparison_dreams.pdf")
    if "combined" in todo:
        fig_combined(os.path.join(an, "combined"), out)
        print("combined_spectra.pdf")
    if "models" in todo:
        if not a.models:
            print("model_comparison.pdf skipped: give --models")
        else:
            fig_models(os.path.join(an, "combined"), a.models, a.dreams_combined, out)
            print("model_comparison.pdf")
    print(f"Figures: {out}")


if __name__ == "__main__":
    main()
