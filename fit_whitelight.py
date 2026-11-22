#!/usr/bin/env python
"""Fit the white-light transits of TRAPPIST-1 b and e in the Eureka! Stage 4 outputs.

For every reduced visit (marked S4_DONE by run_reduction.py), this script

  1. reads the white light curve (flux_white, err_white, mask_white) and the
     trace position and width recorded by Eureka! (centroid_y, centroid_sy,
     centroid_x) from  reduction/<visit>/Stage4/S4_*/ap*_bg*/S4_*_LCData.h5;
  2. finds the transits in the visit and identifies TRAPPIST-1 b and e from
     their predicted times (linear ephemerides with a tolerance for transit-timing
     variations) and their duration and depth. Transits of the other TRAPPIST-1
     planets are masked out of the fit;
  3. fits all b and e transits of the visit jointly with batman:
       free  - mid-transit time and Rp/R* of each planet, quadratic limb darkening
               (Kipping 2013 q1, q2, shared by the planets), a baseline polynomial
               in time for each continuous segment of the light curve (order
               --npoly, or 1 for segments shorter than 2.5 h) and linear terms in
               the trace position and width;
       fixed - period, a/R* and inclination (Agol et al. 2021, PSJ 2, 1, Table 5);
     transits of other planets, flares and other events listed in EXCLUDE (or
     given with --exclude) are left out, which splits the light curve into
     segments; points deviating by more than --clip robust sigma from the first
     fit are removed and the fit is repeated;
  4. estimates correlated noise with the time-averaging method (beta factor,
     bins of 5-30 min) and inflates the parameter uncertainties by beta.

Outputs, in --out (default <drive>/analysis/whitelight):
    whitelight_fits.csv            one row per planet and visit
    <visit>_whitelight.png         light curve, detrended transit(s) and residuals
    summary_rp.png                 Rp/R* of b and e for all visits

Usage
    python fit_whitelight.py --drive WORKDIR                      # all reduced visits
    python fit_whitelight.py --drive WORKDIR --visits jw01331001001
    python fit_whitelight.py --drive WORKDIR --npoly 1            # linear baselines
    python fit_whitelight.py --drive WORKDIR --exclude jw06456002001:60843.80-60843.81
Requires numpy, scipy, matplotlib, xarray, h5netcdf and batman-package (all
installed with Eureka!).
"""
import argparse
import csv
import glob
import os
import sys
import traceback

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import least_squares

# Fixed orbital parameters: Agol et al. (2021), PSJ 2, 1, Table 5 (photodynamical fit).
# rp_ref is the Spitzer 4.5 um radius ratio, shown in the summary figure for reference.
PLANETS = {
    "b": dict(per=1.510826, a=20.843, inc=89.728, rp_ref=0.08590, rp_ref_err=0.00037,
              dur_h=36.06 / 60, depth=0.00738),
    "e": dict(per=6.101013, a=52.855, inc=89.793, rp_ref=0.07079, rp_ref_err=0.00055,
              dur_h=55.76 / 60, depth=0.00501),
}
# Linear ephemerides (BMJD_TDB, days) used to recognise b and e among the dips
# found in a light curve, with the tolerance (minutes) allowed for transit-timing
# variations. b: fitted to 14 transits in GO 6456/9256 (residuals < 1.5 min).
# e: fitted to 13 white-light mid-times from GO 1331, 6456 and 9256 (2023-2025),
# residuals within +/-30 min. Shape alone cannot separate b from c, or e from d,
# whose transits have similar durations, hence ephemerides plus depth windows.
EPHEM = {"b": dict(T0=60496.37309, P=1.510849, tol_min=30.0),
         "e": dict(T0=60502.24253, P=6.099797, tol_min=45.0)}
# Time windows (BMJD_TDB) excluded from the fit, with the reason. Add events
# found on inspection here, or pass them with --exclude VISIT:START-END.
EXCLUDE = {
    "jw06456001001": [(60496.1505, 60496.2620,
                       "flare starting during the egress of e, with its decay")],
    "jw09256007001": [(60977.9550, 60978.1000,
                       "transit of e overlapping a long transit-like event at the start of the visit; not modelled"),
                      (60978.1690, 60978.1850, "small flare")],
}
# Loose shape windows (full width at half depth in hours, depth as a fraction)
# that a dip must also satisfy.
SHAPE = {"b": dict(fwhm=(0.35, 0.75), depth=(0.0040, 0.0120)),
         "e": dict(fwhm=(0.60, 1.10), depth=(0.0042, 0.0080))}   # d is ~0.0037 deep


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
# Folder of the Eureka! outputs inside the working folder; set TRAPPIST1_REDUCTION to
# analyse an alternative reduction (e.g. reduction_nosat written with run_reduction.py --tag nosat).
REDUCTION = os.environ.get("TRAPPIST1_REDUCTION", "reduction")


def find_lcdata(drive, visit):
    """Newest Stage 4 LCData file of a visit, or None."""
    files = glob.glob(os.path.join(drive, REDUCTION, visit, "Stage4", "S4_*",
                                   "ap*_bg*", "S4_*_LCData.h5"))
    return max(files, key=os.path.getmtime) if files else None


def read_whitelight(path):
    """Return time (BMJD_TDB), normalised flux and error, good-point mask and
    a dict of standardised decorrelation vectors (trace position and width)."""
    import xarray as xr
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        t = np.asarray(ds["time"].values, float)
        f = np.asarray(ds["flux_white"].values, float)
        e = np.asarray(ds["err_white"].values, float)
        bad = np.asarray(ds["mask_white"].values, bool) if "mask_white" in ds else np.zeros(t.size, bool)
        vecs = {}
        for name in ("centroid_y", "centroid_sy", "centroid_x"):
            if name in ds and ds[name].dims == ("time",):
                vecs[name] = np.asarray(ds[name].values, float)
    good = (~bad) & np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    norm = np.median(f[good])
    f, e = f / norm, e / norm
    drift = {}
    for name, v in vecs.items():
        ok = good & np.isfinite(v)
        s = np.std(v[ok]) if ok.sum() > 10 else 0.0
        if s > 1e-6 * max(1.0, abs(np.median(v[ok]))):   # skip vectors that do not vary
            drift[name] = np.where(np.isfinite(v), (v - np.median(v[ok])) / s, 0.0)
    return t, f, e, good, drift


# ---------------------------------------------------------------------------
# Finding and identifying the transits
# ---------------------------------------------------------------------------
def find_dips(t, flux, min_depth=0.0015, min_dur_h=0.25):
    """Locate transit-like dips. A 5-minute running median is divided by a
    quadratic baseline fitted to the out-of-dip points. Returns a list of
    (mid_time, fwhm_hours, depth) tuples."""
    dt = np.median(np.diff(t)) * 86400.0
    sm = median_filter(flux, size=int(max(5, round(300 / dt))), mode="nearest")
    base = np.full_like(sm, np.median(sm))
    for _ in range(3):
        m = sm > base - min_depth
        base = np.polyval(np.polyfit(t[m] - t[0], sm[m], 2), t - t[0])
    r = sm / base
    dips = []
    below = np.diff(np.r_[0, (r < 1 - min_depth).astype(int), 0])
    for a, b in zip(np.where(below == 1)[0], np.where(below == -1)[0]):
        if (t[b - 1] - t[a]) * 24 < min_dur_h:
            continue
        seg = np.sort(r[a:b])
        depth = 1 - np.median(seg[: max(3, len(seg) // 3)])
        half = np.diff(np.r_[0, (r < 1 - depth / 2).astype(int), 0])
        for x, y in zip(np.where(half == 1)[0], np.where(half == -1)[0]):
            if x <= b and y >= a:
                dips.append(((t[x] + t[y - 1]) / 2, (t[y - 1] - t[x]) * 24, depth))
                break
    return dips


def drop_excluded(t0_guess, windows, max_frac=0.5):
    """Remove planets whose transit (mid-time +/- half duration) lies more than
    max_frac inside the exclusion windows; return (kept, dropped)."""
    kept, dropped = {}, []
    for pl, t0 in t0_guess.items():
        half = PLANETS[pl]["dur_h"] / 48
        a0, b0 = t0 - half, t0 + half
        cover = sum(max(0.0, min(b0, b) - max(a0, a)) for a, b, _ in windows)
        if cover > max_frac * (b0 - a0):
            dropped.append(pl)
        else:
            kept[pl] = t0
    return kept, dropped


def identify(dips):
    """Label each dip 'b', 'e' or 'other'. A dip is assigned to a planet when its
    mid-time is within the planet's tolerance of a predicted transit and its
    shape is compatible; if several dips qualify, the one nearest the prediction
    is kept."""
    labels = ["other"] * len(dips)
    for p, eph in EPHEM.items():
        best, dist = None, np.inf
        for i, (mid, fw, dp) in enumerate(dips):
            n = np.round((mid - eph["T0"]) / eph["P"])
            off = abs(mid - (eph["T0"] + n * eph["P"])) * 1440
            sh = SHAPE[p]
            if (off < eph["tol_min"] and sh["fwhm"][0] <= fw <= sh["fwhm"][1]
                    and sh["depth"][0] <= dp <= sh["depth"][1] and labels[i] == "other"
                    and off < dist):
                best, dist = i, off
        if best is not None:
            labels[best] = p
    return [(lab, mid, fw, dp) for lab, (mid, fw, dp) in zip(labels, dips)]


# ---------------------------------------------------------------------------
# Model and fit
# ---------------------------------------------------------------------------
def transit(t, t0, rp, planet, u1, u2):
    import batman
    pp = PLANETS[planet]
    prm = batman.TransitParams()
    prm.t0, prm.per, prm.rp, prm.a, prm.inc = t0, pp["per"], rp, pp["a"], pp["inc"]
    prm.ecc, prm.w, prm.limb_dark, prm.u = 0.0, 90.0, "quadratic", [u1, u2]
    return batman.TransitModel(prm, t).light_curve(prm)


def q_to_u(q1, q2):
    """Kipping (2013) parameters to quadratic limb-darkening coefficients."""
    s = np.sqrt(q1)
    return 2 * s * q2, s * (1 - 2 * q2)


def segments(t, use, gap_min=20.0):
    """Time windows (start, end) of the continuous stretches of used points,
    split wherever more than gap_min minutes are excluded (masked transits of
    other planets, flares, user exclusions)."""
    tu = t[use]
    br = np.where(np.diff(tu) * 1440 > gap_min)[0]
    starts = np.r_[tu[0], tu[br + 1]]
    ends = np.r_[tu[br], tu[-1]]
    return list(zip(starts, ends))


class VisitModel:
    """Transits of the planets in `t0_guess` times a baseline. The baseline has
    its own polynomial in each continuous segment (order npoly, or 1 for
    segments shorter than 2.5 h) plus linear terms in the drift vectors."""

    def __init__(self, t, t0_guess, drift, npoly, segs):
        self.t = t
        self.pls = sorted(t0_guess)
        self.t0_guess = t0_guess
        self.dnames = sorted(drift)
        self.D = np.array([drift[k] for k in self.dnames]) if drift else np.zeros((0, t.size))
        self.segs = []
        for a, b in segs:
            span = max(b - a, 1e-3)
            order = npoly if span * 24 >= 2.5 else min(npoly, 1)
            sel = (t >= a - 1e-6) & (t <= b + 1e-6)
            self.segs.append(dict(sel=sel, tc=(t - 0.5 * (a + b)) / span, order=order))
        self.ncoef = sum(sg["order"] + 1 for sg in self.segs)
        self.names = ([f"t0_{p}" for p in self.pls] + [f"rp_{p}" for p in self.pls] +
                      ["q1", "q2"] +
                      [f"s{k}_c{i}" for k, sg in enumerate(self.segs) for i in range(sg["order"] + 1)] +
                      [f"d_{k}" for k in self.dnames])

    def p0_bounds(self):
        coef0 = []
        for sg in self.segs:
            coef0 += [1.0] + [0.0] * sg["order"]
        p0 = ([self.t0_guess[p] for p in self.pls] +
              [np.sqrt(PLANETS[p]["depth"]) for p in self.pls] + [0.36, 0.3] +
              coef0 + [0.0] * len(self.dnames))
        nfree = self.ncoef + len(self.dnames)
        lo = ([self.t0_guess[p] - 0.03 for p in self.pls] + [0.02] * len(self.pls) +
              [0.0, 0.0] + [-np.inf] * nfree)
        hi = ([self.t0_guess[p] + 0.03 for p in self.pls] + [0.2] * len(self.pls) +
              [1.0, 1.0] + [np.inf] * nfree)
        return np.array(p0), (np.array(lo), np.array(hi))

    def split(self, p):
        n = len(self.pls)
        t0 = dict(zip(self.pls, p[:n]))
        rp = dict(zip(self.pls, p[n:2 * n]))
        q1, q2 = p[2 * n:2 * n + 2]
        c = p[2 * n + 2:2 * n + 2 + self.ncoef]
        d = p[2 * n + 2 + self.ncoef:]
        return t0, rp, q1, q2, c, d

    def components(self, p):
        """Return (transit model, baseline) at the model's times."""
        t0, rp, q1, q2, c, d = self.split(p)
        u1, u2 = q_to_u(q1, q2)
        tr = np.ones_like(self.t)
        for pl in self.pls:
            tr *= transit(self.t, t0[pl], rp[pl], pl, u1, u2)
        base = np.ones_like(self.t)
        i = 0
        for sg in self.segs:
            k = sg["order"] + 1
            base = np.where(sg["sel"], np.polyval(c[i:i + k][::-1], sg["tc"]), base)
            i += k
        if len(d):
            base = base + d @ self.D
        return tr, base

    def __call__(self, p):
        tr, base = self.components(p)
        return tr * base


def fit(t, f, e, use, t0_guess, drift, npoly, clip):
    """Least-squares fit with one round of outlier rejection. Returns the
    model object, best-fit parameters, covariance-based errors, the final
    mask and the residuals of the used points."""
    use = use.copy()
    segs = segments(t, use)
    for it in range(2):
        tt, ff, ee = t[use], f[use], e[use]
        dd = {k: v[use] for k, v in drift.items()}
        mod = VisitModel(tt, t0_guess, dd, npoly, segs)
        p0, bounds = mod.p0_bounds()
        sol = least_squares(lambda p: (mod(p) - ff) / ee, p0, bounds=bounds, x_scale="jac")
        if it == 0:
            # remove outliers relative to the first fit, then refit
            full = VisitModel(t, t0_guess, drift, npoly, segs)
            r = f - full(sol.x)
            sig = 1.4826 * np.median(np.abs(r[use] - np.median(r[use])))
            use &= np.abs(r) < clip * sig
    resid = ff - mod(sol.x)
    chi2r = np.sum((resid / ee) ** 2) / (resid.size - sol.x.size)
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * max(chi2r, 1.0)
        perr = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        perr = np.full(sol.x.size, np.nan)
    return mod, sol.x, perr, use, resid


def beta_factor(t, resid, bins_min=(5, 7.5, 10, 15, 20, 30)):
    """Time-averaging red-noise factor: median ratio of the binned residual
    scatter to the white-noise expectation over the given bin sizes (minutes)."""
    dt = np.median(np.diff(t)) * 1440.0
    s1 = np.std(resid)
    ratios = []
    for b in bins_min:
        n = int(round(b / dt))
        m = resid.size // n
        if n < 2 or m < 8:
            continue
        sb = np.std(resid[: m * n].reshape(m, n).mean(1))
        ratios.append(sb / (s1 / np.sqrt(n) * np.sqrt(m / (m - 1))))
    return max(1.0, float(np.median(ratios))) if ratios else 1.0


def gaps_to_nan(t, y, gap_min=5.0):
    """Insert NaN where consecutive times are more than gap_min apart, so that
    lines are not drawn across masked intervals."""
    br = np.where(np.diff(t) * 1440 > gap_min)[0] + 1
    return np.insert(t, br, np.nan), np.insert(np.asarray(y, float), br, np.nan)


def bin_series(t, y, width_min):
    edges = np.arange(t.min(), t.max() + width_min / 1440, width_min / 1440)
    idx = np.digitize(t, edges)
    tb = np.array([t[idx == i].mean() for i in np.unique(idx)])
    yb = np.array([y[idx == i].mean() for i in np.unique(idx)])
    return tb, yb


# ---------------------------------------------------------------------------
# One visit
# ---------------------------------------------------------------------------
def fit_visit(drive, visit, outdir, npoly, clip, exclude):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = find_lcdata(drive, visit)
    if path is None:
        print(f"{visit}: no Stage 4 LCData file found, skipped")
        return []
    t, f, e, good, drift = read_whitelight(path)
    dips = identify(find_dips(t[good], f[good]))
    t0_guess = {lab: mid for lab, mid, fw, dp in dips if lab in PLANETS}
    if not t0_guess:
        print(f"{visit}: no transit of b or e identified (dips: "
              + ", ".join(f"{fw:.2f} h/{dp*1e6:.0f} ppm" for _, _, fw, dp in dips) + ")")
        return []
    use = good.copy()
    for lab, mid, fw, dp in dips:                    # mask transits of other planets
        if lab == "other":
            use &= np.abs(t - mid) * 24 > 0.8 * fw + 0.25
    windows = exclude.get(visit, [])
    t0_guess, dropped = drop_excluded(t0_guess, windows)
    for pl in dropped:
        print(f"{visit}  {pl}: transit mostly inside an excluded window, not fitted")
    if not t0_guess:
        return []
    notes = {}
    for a, b, why in windows:                        # flares and other excluded events
        use &= ~((t >= a) & (t <= b))
        for pl, t0g in t0_guess.items():
            half = PLANETS[pl]["dur_h"] / 48
            if a <= t0g + half and b >= t0g - half:
                notes[pl] = f"part of transit excluded: {why}"
    for a, b, why in windows:
        print(f"{visit}  excluded {a:.4f}-{b:.4f}: {why}")
    n_before = int(use.sum())
    mod, p, perr, use, resid = fit(t, f, e, use, t0_guess, drift, npoly, clip)
    beta = beta_factor(t[use], resid)
    t0, rp, q1, q2, c, d = mod.split(p)
    t0e, rpe, *_ = mod.split(perr * beta)
    u1, u2 = q_to_u(q1, q2)
    rms = np.std(resid) * 1e6
    nclip = n_before - int(use.sum())
    prog = f"GO {int(visit[2:7])}"
    rows = []
    for pl in mod.pls:
        rows.append(dict(visit=visit, program=prog, planet=pl,
                         t0_bmjd=f"{t0[pl]:.6f}", t0_err_min=f"{t0e[pl]*1440:.2f}",
                         rp=f"{rp[pl]:.5f}", rp_err=f"{rpe[pl]:.5f}",
                         depth_ppm=f"{rp[pl]**2*1e6:.0f}",
                         depth_err_ppm=f"{2*rp[pl]*rpe[pl]*1e6:.0f}",
                         u1=f"{u1:.3f}", u2=f"{u2:.3f}", rms_ppm=f"{rms:.0f}",
                         beta=f"{beta:.2f}", n_used=int(use.sum()), n_clipped=nclip,
                         npoly=npoly, segments=len(mod.segs),
                         decorrelation="+".join(mod.dnames) or "none",
                         note=notes.get(pl, ""),
                         lcdata=os.path.relpath(path, drive)))
        print(f"{visit}  {pl}: Rp/R* = {rp[pl]:.5f} +/- {rpe[pl]:.5f}  "
              f"(depth {rp[pl]**2*1e6:.0f} +/- {2*rp[pl]*rpe[pl]*1e6:.0f} ppm), "
              f"t0 = {t0[pl]:.5f} +/- {t0e[pl]*1440:.1f} min")
    others = [d_ for d_ in dips if d_[0] == "other"]
    print(f"{visit}  rms {rms:.0f} ppm, beta {beta:.2f}, {nclip} points clipped"
          + (f", {len(others)} other transit(s) masked" if others else ""))

    # figure: raw light curve with model, detrended transit, residuals
    tt = t[use]
    tr, base = mod.components(p)
    x0 = np.floor(tt.min())
    fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 2, 1]))
    tb, fb = bin_series(tt, f[use], 2)
    ax[0].plot(tt - x0, f[use], ",", color="0.7")
    ax[0].plot(tb - x0, fb, "o", ms=3, color="C0", label="2-min bins")
    ax[0].plot(*gaps_to_nan(tt - x0, tr * base), "-", color="C3", lw=1, label="model")
    for lab, mid, fw, dp in others:
        ax[0].axvspan(mid - x0 - (0.8 * fw + 0.25) / 24, mid - x0 + (0.8 * fw + 0.25) / 24,
                      color="0.9", zorder=0)
    for a, b, why in windows:
        ax[0].axvspan(a - x0, b - x0, color="C3", alpha=0.12, zorder=0)
        ax[0].text(0.5 * (a + b) - x0, ax[0].get_ylim()[1], "excluded", ha="center",
                   va="top", fontsize=7, color="C3")
    ax[0].set_ylabel("normalised flux")
    ax[0].legend(loc="lower left", fontsize=8)
    det = f[use] / base
    tb, db = bin_series(tt, det, 2)
    ax[1].plot(tt - x0, det, ",", color="0.7")
    ax[1].plot(tb - x0, db, "o", ms=3, color="C0")
    ax[1].plot(*gaps_to_nan(tt - x0, tr), "-", color="C3", lw=1)
    for pl in mod.pls:
        ax[1].text(t0[pl] - x0, 1 - rp[pl] ** 2 - 0.0012, pl, ha="center", fontsize=11)
    ax[1].set_ylabel("detrended flux")
    tb, rb = bin_series(tt, resid * 1e6, 2)
    ax[2].plot(tt - x0, resid * 1e6, ",", color="0.7")
    ax[2].plot(tb - x0, rb, "o", ms=3, color="C0")
    ax[2].axhline(0, color="C3", lw=1)
    ax[2].set_ylabel("residual (ppm)")
    ax[2].set_xlabel(f"time (BMJD_TDB - {x0:.0f})")
    ax[0].set_title(f"{visit} ({prog})  rms {rms:.0f} ppm, beta {beta:.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{visit}_whitelight.png"), dpi=150)
    plt.close(fig)
    return rows


def summary_plot(rows, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, pl in zip(axs, ("b", "e")):
        r = [x for x in rows if x["planet"] == pl]
        if not r:
            ax.set_visible(False)
            continue
        ref, referr = PLANETS[pl]["rp_ref"], PLANETS[pl]["rp_ref_err"]
        ax.axhspan(ref - referr, ref + referr, color="0.85", label="Agol et al. 2021 (Spitzer 4.5 µm)")
        for i, x in enumerate(r):
            col = "C1" if x["program"] == "GO 1331" else "C0"
            ax.errorbar(i, float(x["rp"]), float(x["rp_err"]), fmt="o", color=col)
        ax.set_xticks(range(len(r)))
        ax.set_xticklabels([x["visit"][2:] for x in r], rotation=60, fontsize=7)
        ax.set_ylabel("Rp/R* (white light, 0.62-5.29 µm)")
        ax.set_title(f"TRAPPIST-1 {pl}  (orange: GO 1331, blue: GO 6456/9256)", fontsize=9)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "summary_rp.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", default=os.environ.get("TRAPPIST1_DATA", "."))
    ap.add_argument("--visits", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--npoly", type=int, default=2,
                    help="order of the baseline polynomial in each segment longer than 2.5 h (default 2)")
    ap.add_argument("--exclude", nargs="+", default=[], metavar="VISIT:START-END",
                    help="extra time windows (BMJD_TDB) to exclude, e.g. jw06456001001:60496.15-60496.26")
    ap.add_argument("--clip", type=float, default=5.0, help="outlier rejection threshold in robust sigma (default 5)")
    a = ap.parse_args()
    drive = os.path.abspath(os.path.expanduser(a.drive))
    outdir = a.out or os.path.join(drive, "analysis", "whitelight")
    os.makedirs(outdir, exist_ok=True)

    done = sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(drive, REDUCTION, "jw*", "S4_DONE")))
    visits = [v for v in (a.visits or done)]
    if not visits:
        sys.exit(f"No reduced visits (S4_DONE) found under {drive}/reduction")
    exclude = {k: list(v) for k, v in EXCLUDE.items()}
    for item in a.exclude:
        v, rng = item.split(":")
        t1, t2 = (float(x) for x in rng.split("-"))
        exclude.setdefault(v, []).append((t1, t2, "excluded on the command line"))
    rows = []
    for v in visits:
        try:
            rows += fit_visit(drive, v, outdir, a.npoly, a.clip, exclude)
        except Exception:
            print(f"{v}: fit FAILED")
            traceback.print_exc()
    if not rows:
        sys.exit("No fits succeeded.")
    csvfile = os.path.join(outdir, "whitelight_fits.csv")
    with open(csvfile, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary_plot(rows, outdir)
    for pl in ("b", "e"):
        r = [x for x in rows if x["planet"] == pl]
        if len(r) > 1:
            rp = np.array([float(x["rp"]) for x in r])
            er = np.array([float(x["rp_err"]) for x in r])
            w_ = 1 / er**2
            mean = np.sum(rp * w_) / np.sum(w_)
            chi2 = np.sum((rp - mean) ** 2 * w_)
            print(f"{pl}: {len(r)} transits, weighted mean Rp/R* = {mean:.5f} +/- "
                  f"{np.sum(w_)**-0.5:.5f}, chi2 = {chi2:.1f} for {len(r)-1} dof")
    print(f"Results: {csvfile}\nFigures: {outdir}")


if __name__ == "__main__":
    main()
