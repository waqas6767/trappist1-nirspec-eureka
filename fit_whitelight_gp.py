#!/usr/bin/env python
"""White-light transit fits of TRAPPIST-1 b and e with a Gaussian-process noise
model and MCMC sampling.

This is the final-quality version of fit_whitelight.py. It uses the same input
(Eureka! Stage 4 white light curves), the same identification of b and e, the
same exclusion windows (EXCLUDE in fit_whitelight.py, or --exclude) and the same
fixed orbital parameters (Agol et al. 2021). It differs in the noise treatment:

  * The light curve is binned to --bin-s seconds (default 30 s, i.e. about 20
    integrations), which leaves ingress and egress (3-4 min) well resolved and
    makes the likelihood fast.
  * Time-correlated noise (stellar granulation, small flares, slow trends) is
    modelled with a celerite2 Matern-3/2 Gaussian process (amplitude sigma_gp,
    time scale rho_gp) plus a white-noise jitter term added to the
    uncertainties, instead of inflating the errors with a beta factor.
  * The deterministic baseline is a low-order polynomial for each continuous
    segment (--npoly, default 1) plus linear terms in the trace position and
    width; the GP absorbs the remaining curvature.
  * All parameters are sampled with emcee, so the transit depths are
    marginalised over the baseline and noise parameters. Reported values are
    posterior medians with 16th-84th percentile intervals.

Free parameters and priors (uniform):
    t0 of each planet       guess +/- 0.03 d
    Rp/R* of each planet    0.02 - 0.2
    q1, q2 (Kipping 2013)   0 - 1, shared by the planets
    baseline coefficients   unbounded
    drift coefficients      unbounded
    ln sigma_gp             ln(1e-6) - ln(1e-2)
    ln rho_gp [d]           ln(1 min) - ln(1 d)
    ln jitter               ln(1e-6) - ln(1e-2)

Outputs, in --out (default <drive>/analysis/whitelight_gp):
    whitelight_gp_fits.csv          one row per planet and visit
    <visit>_gp.png                  light curve with GP, detrended transit(s), residuals
    <visit>_chain.npz               flattened posterior samples and parameter names
    summary_rp_gp.png               Rp/R* of b and e for all visits

Usage
    python fit_whitelight_gp.py --drive WORKDIR                       # all reduced visits
    python fit_whitelight_gp.py --drive WORKDIR --visits jw01331001001 --steps 4000
    python fit_whitelight_gp.py --drive WORKDIR --processes 8         # visits in parallel
Requires the packages installed with Eureka! (batman-package, celerite2, emcee).
"""
import argparse
import csv
import glob
import logging
import os
import sys
import traceback
import warnings

import numpy as np
from scipy.optimize import minimize

import fit_whitelight as W

LN_BOUNDS = {"ln_sigma": (np.log(1e-6), np.log(1e-2)),
             "ln_rho": (np.log(1 / 1440), np.log(1.0)),
             "ln_jit": (np.log(1e-6), np.log(1e-2))}


def bin_lightcurve(t, f, e, use, drift, bin_s):
    """Average the used points in bins of bin_s seconds; bins never span a gap."""
    tu, fu, eu = t[use], f[use], e[use]
    du = {k: v[use] for k, v in drift.items()}
    edges = np.r_[0, np.where(np.diff(tu) * 86400 > bin_s)[0] + 1, tu.size]
    idx = []
    for a, b in zip(edges[:-1], edges[1:]):          # continuous stretches
        n = max(1, int(round((tu[b - 1] - tu[a]) * 86400 / bin_s)))
        idx += [ix for ix in np.array_split(np.arange(a, b), n) if ix.size]
    tb = np.array([tu[ix].mean() for ix in idx])
    fb = np.array([fu[ix].mean() for ix in idx])
    eb = np.array([np.sqrt(np.sum(eu[ix] ** 2)) / ix.size for ix in idx])
    db = {k: np.array([v[ix].mean() for ix in idx]) for k, v in du.items()}
    return tb, fb, eb, db


class GPModel:
    """Transit x baseline mean model with a Matern-3/2 GP and jitter."""

    def __init__(self, t, f, e, t0_guess, drift, npoly, segs):
        import celerite2
        self.celerite2 = celerite2
        self.t, self.f, self.e = t, f, e
        self.npoly = npoly
        self.mean = W.VisitModel(t, t0_guess, drift, npoly, segs)
        self.nmean = len(self.mean.names)
        self.names = self.mean.names + ["ln_sigma", "ln_rho", "ln_jit"]
        p0, (lo, hi) = self.mean.p0_bounds()
        self.lo = np.r_[lo, [LN_BOUNDS[k][0] for k in ("ln_sigma", "ln_rho", "ln_jit")]]
        self.hi = np.r_[hi, [LN_BOUNDS[k][1] for k in ("ln_sigma", "ln_rho", "ln_jit")]]

    def gp(self, p):
        ln_sigma, ln_rho, ln_jit = p[self.nmean:]
        term = self.celerite2.terms.Matern32Term(sigma=np.exp(ln_sigma), rho=np.exp(ln_rho))
        gp = self.celerite2.GaussianProcess(term, mean=0.0)
        gp.compute(self.t, diag=self.e ** 2 + np.exp(2 * ln_jit), quiet=True)
        return gp

    def log_prob(self, p):
        if np.any(p < self.lo) or np.any(p > self.hi):
            return -np.inf
        r = self.f - self.mean(p[:self.nmean])
        try:
            ll = self.gp(p).log_likelihood(r)
        except Exception:
            return -np.inf
        return ll if np.isfinite(ll) else -np.inf

    def systematics(self, p, t_full, drift_full, segs):
        """Multiplicative systematics S(t) = baseline + GP / transit at any times,
        such that the white light curve is transit(t) x S(t). The GP is
        conditioned on the binned residuals and predicted at t_full."""
        full = W.VisitModel(t_full, self.mean.t0_guess, drift_full, self.npoly, segs)
        pm = p[:self.nmean]
        tr_full, base_full = full.components(pm)
        r = self.f - self.mean(pm)
        mu_full = self.gp(p).predict(r, t=t_full, return_cov=False)
        return base_full + mu_full / tr_full, tr_full

    def predict(self, p):
        """Mean model, transit, baseline and GP prediction at the data times."""
        tr, base = self.mean.components(p[:self.nmean])
        r = self.f - tr * base
        mu = self.gp(p).predict(r, t=self.t, return_cov=False)
        return tr, base, mu


def best_params(model, chainfile):
    """Maximum-probability parameters of a saved white-light chain.

    Chains are stored in double precision. Chains written in single precision
    (float32) by earlier versions cannot hold the mid-transit times to better
    than 0.0039 d (5.6 min) at MJD 60000; for those the maximum-probability
    sample is refined by a local optimization of the posterior, which restores
    the full precision."""
    ch = np.load(chainfile)
    names = [str(x) for x in ch["names"]]
    if names != model.names:
        raise RuntimeError(f"{chainfile}: parameters do not match the current setup; rerun fit_whitelight_gp.py")
    p = ch["chain"][np.argmax(ch["log_prob"])].astype(float)
    if ch["chain"].dtype == np.float32:
        p = np.clip(p, model.lo + 1e-12, model.hi - 1e-12)
        for method, opts in (("Nelder-Mead", dict(maxiter=20000, maxfev=20000, xatol=1e-9, fatol=1e-4)),
                             ("Powell", dict(maxiter=20000, xtol=1e-9, ftol=1e-8))):
            r = minimize(lambda x: -model.log_prob(x), p, method=method, options=opts)
            if np.isfinite(r.fun) and -r.fun >= model.log_prob(p):
                p = r.x
    return p


def prepare(drive, visit, npoly, clip, exclude, bin_s):
    """Read a visit, identify the transits, apply masks and exclusions, run the
    least-squares fit (outlier rejection, starting point) and bin the white light
    curve. Returns a dict, or an error message string."""
    path = W.find_lcdata(drive, visit)
    if path is None:
        return f"{visit}: no Stage 4 LCData file found, skipped"
    t, f, e, good, drift = W.read_whitelight(path)
    dips = W.identify(W.find_dips(t[good], f[good]))
    t0_guess = {lab: mid for lab, mid, fw, dp in dips if lab in W.PLANETS}
    if not t0_guess:
        return f"{visit}: no transit of b or e identified"
    use = good.copy()
    for lab, mid, fw, dp in dips:
        if lab == "other":
            use &= np.abs(t - mid) * 24 > 0.8 * fw + 0.25
    windows = exclude.get(visit, [])
    t0_guess, dropped = W.drop_excluded(t0_guess, windows)
    if not t0_guess:
        return f"{visit}: all transits inside excluded windows"
    notes = {}
    for a, b, why in windows:
        use &= ~((t >= a) & (t <= b))
        for pl, t0g in t0_guess.items():
            half = W.PLANETS[pl]["dur_h"] / 48
            if a <= t0g + half and b >= t0g - half:
                notes[pl] = f"part of transit excluded: {why}"
    segs = W.segments(t, use)          # the same segments as used inside W.fit
    _, p_ls, _, use, _ = W.fit(t, f, e, use, t0_guess, drift, npoly, clip)
    tb, fb, eb, db = bin_lightcurve(t, f, e, use, drift, bin_s)
    model = GPModel(tb, fb, eb, t0_guess, db, npoly, segs)
    return dict(path=path, t=t, f=f, e=e, good=good, drift=drift, dips=dips, t0_guess=t0_guess,
                use=use, windows=windows, notes=notes, segs=segs, tb=tb, fb=fb, eb=eb, db=db,
                model=model, p_ls=p_ls)


def fit_visit_gp(drive, visit, outdir, npoly, clip, exclude, bin_s, walkers, steps, burn, seed):
    # convergence is reported in the output line, so emcee's and scipy's own warnings are silenced
    logging.getLogger("emcee").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import emcee

    prep = prepare(drive, visit, npoly, clip, exclude, bin_s)
    if isinstance(prep, str):
        return [], prep
    path, dips, windows, notes, segs = (prep[k] for k in ("path", "dips", "windows", "notes", "segs"))
    tb, fb, model, p_ls = prep["tb"], prep["fb"], prep["model"], prep["p_ls"]
    resid_ls = fb - model.mean(p_ls)
    p_start = np.r_[p_ls, np.log(max(np.std(resid_ls) * 0.5, 2e-5)), np.log(10 / 1440),
                    np.log(max(np.std(resid_ls) * 0.3, 2e-5))]
    p_start = np.clip(p_start, model.lo + 1e-9, model.hi - 1e-9)
    opt = minimize(lambda p: -model.log_prob(p), p_start, method="Nelder-Mead",
                   options=dict(maxiter=20000, maxfev=20000, xatol=1e-8, fatol=1e-3))
    opt = minimize(lambda p: -model.log_prob(p), opt.x, method="Powell",
                   options=dict(maxiter=20000, xtol=1e-8, ftol=1e-6))
    p_map = opt.x if np.isfinite(-opt.fun) else p_start

    # 3. MCMC
    ndim = p_map.size
    nw = max(walkers, 2 * ndim + 2)
    rng = np.random.default_rng(seed)
    scale = np.maximum(np.abs(p_map) * 1e-5, 1e-6)
    for i, n in enumerate(model.names):
        if n.startswith("ln_"):
            scale[i] = 0.05
        elif n.startswith(("q1", "q2")):
            scale[i] = 0.01
        elif n.startswith("t0_"):
            scale[i] = 1e-4
        elif n.startswith("rp_"):
            scale[i] = 1e-4
    pos = p_map + scale * rng.standard_normal((nw, ndim))
    pos = np.clip(pos, model.lo + 1e-9, model.hi - 1e-9)
    sampler = emcee.EnsembleSampler(nw, ndim, model.log_prob)
    sampler.run_mcmc(pos, steps, progress=False)
    chain = sampler.get_chain(discard=burn, flat=True)
    lnp = sampler.get_log_prob(discard=burn, flat=True)
    try:
        tau = float(np.max(sampler.get_autocorr_time(discard=burn, quiet=True)))
    except Exception:
        tau = np.nan
    acc = float(np.mean(sampler.acceptance_fraction))
    np.savez_compressed(os.path.join(outdir, f"{visit}_chain.npz"), chain=chain,
                        log_prob=lnp, names=np.array(model.names))

    pct = np.percentile(chain, [16, 50, 84], axis=0)
    med = pct[1]
    lo_err, hi_err = med - pct[0], pct[2] - med
    p_best = chain[np.argmax(lnp)]
    ix = {n: i for i, n in enumerate(model.names)}
    q1s, q2s = chain[:, ix["q1"]], chain[:, ix["q2"]]
    u1s, u2s = W.q_to_u(q1s, q2s)

    tr, base, mu = model.predict(p_best)
    resid = fb - tr * base - mu
    rms_bin = np.std(resid) * 1e6
    sig_gp = np.exp(chain[:, ix["ln_sigma"]]) * 1e6
    rho_gp = np.exp(chain[:, ix["ln_rho"]]) * 1440
    jit = np.exp(chain[:, ix["ln_jit"]]) * 1e6
    prog = f"GO {int(visit[2:7])}"
    rows, lines = [], []
    for pl in model.mean.pls:
        rps = chain[:, ix[f"rp_{pl}"]]
        dps = rps ** 2 * 1e6
        t0s = chain[:, ix[f"t0_{pl}"]]
        dlo, dmed, dhi = np.percentile(dps, [16, 50, 84])
        rows.append(dict(
            visit=visit, program=prog, planet=pl,
            t0_bmjd=f"{np.median(t0s):.6f}", t0_err_min=f"{np.std(t0s)*1440:.2f}",
            rp=f"{med[ix['rp_'+pl]]:.5f}", rp_err_lo=f"{lo_err[ix['rp_'+pl]]:.5f}",
            rp_err_hi=f"{hi_err[ix['rp_'+pl]]:.5f}",
            depth_ppm=f"{dmed:.0f}", depth_err_lo_ppm=f"{dmed-dlo:.0f}", depth_err_hi_ppm=f"{dhi-dmed:.0f}",
            u1=f"{np.median(u1s):.3f}", u2=f"{np.median(u2s):.3f}",
            sigma_gp_ppm=f"{np.median(sig_gp):.0f}", rho_gp_min=f"{np.median(rho_gp):.1f}",
            jitter_ppm=f"{np.median(jit):.0f}", rms_binned_ppm=f"{rms_bin:.0f}", bin_s=bin_s,
            n_bins=tb.size, npoly=npoly, segments=len(segs),
            decorrelation="+".join(model.mean.dnames) or "none",
            walkers=nw, steps=steps, burn=burn, autocorr_steps=f"{tau:.0f}",
            acceptance=f"{acc:.2f}", note=notes.get(pl, ""),
            lcdata=os.path.relpath(path, drive)))
        lines.append(f"{visit}  {pl}: Rp/R* = {med[ix['rp_'+pl]]:.5f} "
                     f"-{lo_err[ix['rp_'+pl]]:.5f} +{hi_err[ix['rp_'+pl]]:.5f}  "
                     f"(depth {dmed:.0f} -{dmed-dlo:.0f} +{dhi-dmed:.0f} ppm), "
                     f"t0 = {np.median(t0s):.5f} +/- {np.std(t0s)*1440:.1f} min")
    lines.append(f"{visit}  GP: sigma {np.median(sig_gp):.0f} ppm, rho {np.median(rho_gp):.1f} min, "
                 f"jitter {np.median(jit):.0f} ppm; binned rms {rms_bin:.0f} ppm; "
                 f"autocorr {tau:.0f} steps of {steps - burn} kept, acceptance {acc:.2f}"
                 + ("  [chain may be too short]" if np.isfinite(tau) and (steps - burn) < 30 * tau else ""))

    # figure
    x0 = np.floor(tb.min())
    fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 2, 1]))
    ax[0].plot(tb - x0, fb, ".", ms=2, color="0.5", label=f"{bin_s:.0f}-s bins")
    ax[0].plot(*W.gaps_to_nan(tb - x0, tr * base + mu), "-", color="C3", lw=1,
               label="transit x baseline + GP")
    ax[0].plot(*W.gaps_to_nan(tb - x0, base), "--", color="C2", lw=0.8, label="baseline")
    for a, b, why in windows:
        ax[0].axvspan(a - x0, b - x0, color="C3", alpha=0.12, zorder=0)
    for lab, mid, fw, dp in dips:
        if lab == "other":
            hw = (0.8 * fw + 0.25) / 24
            ax[0].axvspan(mid - x0 - hw, mid - x0 + hw, color="0.9", zorder=0)
    ax[0].set_ylabel("normalised flux")
    ax[0].legend(loc="lower left", fontsize=7)
    det = (fb - mu) / base
    tb2, db2 = W.bin_series(tb, det, 2)
    ax[1].plot(tb - x0, det, ".", ms=2, color="0.6")
    ax[1].plot(tb2 - x0, db2, "o", ms=3, color="C0", label="2-min bins")
    ax[1].plot(*W.gaps_to_nan(tb - x0, tr), "-", color="C3", lw=1)
    for pl in model.mean.pls:
        ax[1].text(med[ix["t0_" + pl]] - x0, 1 - med[ix["rp_" + pl]] ** 2 - 0.0012, pl,
                   ha="center", fontsize=11)
    ax[1].set_ylabel("flux / baseline - GP")
    ax[1].legend(loc="lower left", fontsize=7)
    tb3, rb3 = W.bin_series(tb, resid * 1e6, 2)
    ax[2].plot(tb - x0, resid * 1e6, ".", ms=2, color="0.6")
    ax[2].plot(tb3 - x0, rb3, "o", ms=3, color="C0")
    ax[2].axhline(0, color="C3", lw=1)
    ax[2].set_ylabel("residual (ppm)")
    ax[2].set_xlabel(f"time (BMJD_TDB - {x0:.0f})")
    ax[0].set_title(f"{visit} ({prog})  GP sigma {np.median(sig_gp):.0f} ppm, "
                    f"rho {np.median(rho_gp):.0f} min", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{visit}_gp.png"), dpi=150)
    plt.close(fig)
    return rows, "\n".join(lines)


def _worker(args):
    try:
        return fit_visit_gp(*args)
    except Exception:
        return [], f"{args[1]}: fit FAILED\n{traceback.format_exc()}"


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
        ref, referr = W.PLANETS[pl]["rp_ref"], W.PLANETS[pl]["rp_ref_err"]
        ax.axhspan(ref - referr, ref + referr, color="0.85", label="Agol et al. 2021 (Spitzer 4.5 µm)")
        for i, x in enumerate(r):
            col = "C1" if x["program"] == "GO 1331" else "C0"
            mk = "s" if x["note"] else "o"
            ax.errorbar(i, float(x["rp"]), [[float(x["rp_err_lo"])], [float(x["rp_err_hi"])]],
                        fmt=mk, color=col, mfc="white" if x["note"] else col)
        ax.set_xticks(range(len(r)))
        ax.set_xticklabels([x["visit"][2:] for x in r], rotation=60, fontsize=7)
        ax.set_ylabel("Rp/R* (white light, GP + MCMC)")
        ax.set_title(f"TRAPPIST-1 {pl}  (orange: GO 1331, blue: GO 6456/9256; open: partial transit)",
                     fontsize=8)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "summary_rp_gp.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", default=os.environ.get("TRAPPIST1_DATA", "."))
    ap.add_argument("--visits", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--npoly", type=int, default=1, help="baseline polynomial order per segment (default 1)")
    ap.add_argument("--clip", type=float, default=5.0, help="outlier threshold in robust sigma (default 5)")
    ap.add_argument("--exclude", nargs="+", default=[], metavar="VISIT:START-END")
    ap.add_argument("--bin-s", type=float, default=30.0, help="bin width in seconds (default 30)")
    ap.add_argument("--walkers", type=int, default=48)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--burn", type=int, default=2000)
    ap.add_argument("--processes", type=int, default=1, help="visits fitted in parallel (default 1)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    drive = os.path.abspath(os.path.expanduser(a.drive))
    outdir = a.out or os.path.join(drive, "analysis", "whitelight_gp")
    os.makedirs(outdir, exist_ok=True)

    done = sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(drive, W.REDUCTION, "jw*", "S4_DONE")))
    visits = a.visits or done
    if not visits:
        sys.exit(f"No reduced visits (S4_DONE) found under {drive}/{W.REDUCTION}")
    exclude = {k: list(v) for k, v in W.EXCLUDE.items()}
    for item in a.exclude:
        v, rng = item.split(":")
        t1, t2 = (float(x) for x in rng.split("-"))
        exclude.setdefault(v, []).append((t1, t2, "excluded on the command line"))

    jobs = [(drive, v, outdir, a.npoly, a.clip, exclude, a.bin_s, a.walkers, a.steps, a.burn,
             a.seed + k) for k, v in enumerate(visits)]
    print(f"Fitting {len(jobs)} visits with GP + MCMC ({a.walkers} walkers, {a.steps} steps, "
          f"burn-in {a.burn}), {a.processes} at a time ...", flush=True)
    rows = []
    if a.processes > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.processes) as pool:
            for r, text in pool.imap(_worker, jobs):
                print(text, flush=True)
                rows += r
    else:
        for job in jobs:
            r, text = _worker(job)
            print(text, flush=True)
            rows += r
    if not rows:
        sys.exit("No fits succeeded.")
    csvfile = os.path.join(outdir, "whitelight_gp_fits.csv")
    if a.visits and os.path.exists(csvfile):
        # keep the earlier results of visits not refitted in this run
        refit = {v for v in visits}
        old = [r for r in csv.DictReader(open(csvfile)) if r["visit"] not in refit]
        if old and set(old[0]) == set(rows[0]):
            rows = sorted(old + rows, key=lambda r: (r["visit"], r["planet"]))
    with open(csvfile, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary_plot(rows, outdir)
    for pl in ("b", "e"):
        r = [x for x in rows if x["planet"] == pl]
        if len(r) > 1:
            for label, sel in (("all transits", r), ("complete transits", [x for x in r if not x["note"]])):
                if len(sel) < 2:
                    continue
                rp = np.array([float(x["rp"]) for x in sel])
                er = np.array([0.5 * (float(x["rp_err_lo"]) + float(x["rp_err_hi"])) for x in sel])
                w_ = 1 / er ** 2
                mean = np.sum(rp * w_) / np.sum(w_)
                chi2 = np.sum((rp - mean) ** 2 * w_)
                print(f"{pl} ({label}): {len(sel)} transits, weighted mean Rp/R* = {mean:.5f} +/- "
                      f"{np.sum(w_)**-0.5:.5f}, chi2 = {chi2:.1f} for {len(sel)-1} dof")
    print(f"Results: {csvfile}\nFigures: {outdir}")


if __name__ == "__main__":
    main()
