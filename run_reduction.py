#!/usr/bin/env python
"""Reduce JWST NIRSpec PRISM time series of TRAPPIST-1 with Eureka! (v1.4), visit by visit.

Expected layout under the data folder (--drive), as written by jwst_download.py:

    <drive>/jwst_data/uncal/<program>/<visit>/*_nrs1_uncal.fits
    <drive>/jwst_data/rateints/<program>/<visit>/*_nrs1_rateints.fits

For every visit this script writes Eureka! control files from the templates in
./templates and runs Stages 1-4:

    Stage 1  uncal    -> rateints  (group-level 1/f subtraction, saturation flagging)
    Stage 2  rateints -> calints
    Stage 3  calints  -> 1-D spectra (optimal extraction)
    Stage 4  1-D spectra -> white and spectroscopic light curves (67 DREAMS bins)

Outputs go to <drive>/reduction/<visit>/Stage1 ... Stage4; the control files are
kept in <drive>/reduction/<visit>/ecf and the log in <drive>/reduction/logs/<visit>.log.
A finished visit is marked with S4_DONE and skipped later (use --redo to force);
a visit that crashed gets FAILED (it is retried on the next run).

With --jobs N, N visits are reduced at the same time, each in its own process
(run at low priority, so other users of a shared server are not slowed down).
A new visit is only started when enough memory is free (--min-free-gb).

With --from-rateints, the reduction starts from the MAST rateints files at Stage 2
and writes to <drive>/reduction_rateints/<visit>; this is a cross-check of Stage 1.

Usage
    python run_reduction.py --drive DIR --list                    # visits found
    python run_reduction.py --drive DIR --status                  # progress table
    python run_reduction.py --drive DIR --prefetch-crds           # download all reference files once
    python run_reduction.py --drive DIR --visits jw01331001001    # one visit (test)
    python run_reduction.py --drive DIR --jobs 6                  # all visits, 6 at a time
Options
    --drive PATH       data folder containing jwst_data/ (default: $TRAPPIST1_DATA, else the current folder)
    --programs ...     restrict to programs, e.g. --programs 06456 09256
    --jobs N           visits reduced in parallel (default 1)
    --s1-ncpu N        cores per visit for the Stage 1 1/f subtraction (default 1)
    --max-cores X      cores per visit for jwst ramp fitting: none, quarter, half, all (default none)
    --s3-ncpu N        cores per visit for Stage 3 extraction (default 2)
    --min-free-gb G    start a new visit only when this much memory is free (default 12)
    --offline          use only the local CRDS cache (after --prefetch-crds); no network needed
    --redo             reduce again even if S4_DONE exists
    --start-stage N    begin at stage N (2-4), reusing the newest outputs of the earlier stages
    --tag NAME         write to reduction_NAME/ (to compare alternative settings)
    --set S1:KEY=VAL   override a control-file setting, e.g. --set S1:update_sat_flags=False
"""
import argparse
import glob
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = os.path.join(HERE, "templates")
BINS = os.path.join(HERE, "dreams_bins.txt")


def find_visits(drive, level, programs=None):
    """Return {visit: (program, folder relative to drive)} for one data level."""
    out = {}
    for d in sorted(glob.glob(os.path.join(drive, "jwst_data", level, "*", "jw*"))):
        if not glob.glob(os.path.join(d, f"*_{level}.fits")):
            continue
        prog, visit = d.split(os.sep)[-2], d.split(os.sep)[-1]
        if programs and prog not in programs:
            continue
        out[visit] = (prog, os.path.relpath(d, drive))
    return out


def mem_available_gb():
    """Free memory (MemAvailable) in GB, read from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return float("inf")


def apply_overrides(text, overrides):
    """Replace the value of each 'key' line of a control file (overrides: {key: value})."""
    lines = text.splitlines()
    for key, val in overrides.items():
        for i, line in enumerate(lines):
            if line.split() and line.split()[0] == key:
                lines[i] = f"{key:<20s}{val}    # set with --set"
                break
        else:
            lines.append(f"{key:<20s}{val}    # set with --set")
    return "\n".join(lines) + "\n"


def write_ecf(stage, visit, ecfdir, topdir, inputdir, outputdir, cores, overrides=None):
    """Fill one template and write it as S<stage>_<visit>.ecf in ecfdir."""
    text = open(os.path.join(TEMPLATES, f"S{stage}_trappist1.ecf")).read()
    repl = {"{TOPDIR}": topdir, "{INPUTDIR}": inputdir, "{OUTPUTDIR}": outputdir,
            "{BINS}": BINS, "{MAXCORES}": cores["max_cores"],
            "{S1_NCPU}": str(cores["s1_ncpu"]), "{S3_NCPU}": str(cores["s3_ncpu"]),
            "{S3_MAXMEM}": f"{cores['s3_maxmem']:.3f}"}
    for k, v in repl.items():
        text = text.replace(k, v)
    if overrides and overrides.get(stage):
        text = apply_overrides(text, overrides[stage])
    path = os.path.join(ecfdir, f"S{stage}_{visit}.ecf")
    with open(path, "w") as f:
        f.write(text)
    return path


def reduce_visit(visit, src_rel, drive, from_rateints, cores, base_name="reduction", overrides=None,
                 start_stage=1):
    """Run Eureka! Stages 1-4 (or 2-4) for one visit in this process."""
    import matplotlib
    matplotlib.use("Agg")
    import eureka.lib.plots
    import eureka.S1_detector_processing.s1_process as s1
    import eureka.S2_calibrations.s2_calibrate as s2
    import eureka.S3_data_reduction.s3_reduce as s3
    import eureka.S4_generate_lightcurves.s4_genLC as s4
    eureka.lib.plots.set_rc(style="eureka", usetex=False, filetype=".png")

    base = os.path.join(base_name, visit)
    ecfdir = os.path.join(drive, base, "ecf")
    os.makedirs(ecfdir, exist_ok=True)
    topdir = os.path.join(drive, "")
    # Stage n reads the output folder of Stage n-1; when the stages run in one
    # process, Eureka! also passes the exact folder on through the meta object.
    dirs = {1: (src_rel, f"{base}/Stage1"),
            2: (src_rel if from_rateints else f"{base}/Stage1", f"{base}/Stage2"),
            3: (f"{base}/Stage2", f"{base}/Stage3"),
            4: (f"{base}/Stage3", f"{base}/Stage4")}
    for st, (i, o) in dirs.items():
        write_ecf(st, visit, ecfdir, topdir, i, o, cores, overrides)

    t0 = time.time()
    meta = None              # with meta None, Eureka! finds the newest output of the previous stage
    if not from_rateints and start_stage <= 1:
        meta = s1.rampfitJWST(visit, ecf_path=ecfdir + os.sep)
        print(f"  Stage 1 done ({(time.time()-t0)/60:.0f} min)", flush=True)
    if start_stage <= 2:
        meta = s2.calibrateJWST(visit, ecf_path=ecfdir + os.sep, s1_meta=meta)
        print(f"  Stage 2 done ({(time.time()-t0)/60:.0f} min)", flush=True)
    if start_stage <= 3:
        spec, meta = s3.reduce(visit, ecf_path=ecfdir + os.sep, s2_meta=meta)
        print(f"  Stage 3 done ({(time.time()-t0)/60:.0f} min)", flush=True)
    spec, lc, meta = s4.genlc(visit, ecf_path=ecfdir + os.sep, s3_meta=meta)
    print(f"  Stage 4 done ({(time.time()-t0)/60:.0f} min)", flush=True)
    open(os.path.join(drive, base, "S4_DONE"), "w").write(time.ctime() + "\n")


def visit_state(drive, base, visit):
    d = os.path.join(drive, base, visit)
    if os.path.exists(os.path.join(d, "S4_DONE")):
        return "done"
    if os.path.exists(os.path.join(d, "FAILED")):
        return "FAILED"
    if os.path.exists(os.path.join(d, "RUNNING")):
        return "running"
    return "waiting"


def last_stage(drive, base, visit):
    log = os.path.join(drive, base, "logs", f"{visit}.log")
    if not os.path.exists(log):
        return ""
    done = [l.strip() for l in open(log, errors="replace") if "Stage" in l and "done (" in l]
    return done[-1] if done else "Stage 1 running"


def prefetch_crds(drive, visits, level):
    """Download every CRDS reference file the visits will need (one file per visit)."""
    files = []
    for v, (p, rel) in visits.items():
        # science exposures only (activity 04101); the target-acquisition images
        # (02101) use a different mode and would select the wrong reference files
        segs = sorted(f for f in glob.glob(os.path.join(drive, rel, f"*_{level}.fits"))
                      if "_02101_" not in os.path.basename(f))
        if segs:
            files.append(segs[0])
    print(f"Fetching CRDS reference files for {len(files)} visits into {os.environ['CRDS_PATH']} ...", flush=True)
    cmd = [sys.executable, "-m", "crds.bestrefs", "--jwst", "--sync-references=1", "--files"] + files
    r = subprocess.run(cmd)
    print("CRDS prefetch", "finished." if r.returncode == 0 else f"exited with code {r.returncode}.")
    return r.returncode


def run_parallel(a, visits, base):
    """Reduce visits N at a time, each in a separate low-priority child process."""
    logdir = os.path.join(a.drive, base, "logs")
    os.makedirs(logdir, exist_ok=True)
    todo = [v for v in visits if a.redo or visit_state(a.drive, base, v) != "done"]
    skipped = len(visits) - len(todo)
    print(f"{len(todo)} visits to reduce ({skipped} already done), {a.jobs} at a time. "
          f"Logs: {logdir}", flush=True)
    running = {}
    nfail = 0
    while todo or running:
        for v, p in list(running.items()):
            if p.poll() is not None:
                del running[v]
                ok = p.returncode == 0 and visit_state(a.drive, base, v) == "done"
                nfail += 0 if ok else 1
                vdir = os.path.join(a.drive, base, v)
                if not ok:                          # also covers a child killed by the system (e.g. out of memory)
                    open(os.path.join(vdir, "FAILED"), "w").write(f"exit code {p.returncode}, {time.ctime()}\n")
                if os.path.exists(os.path.join(vdir, "RUNNING")):
                    os.remove(os.path.join(vdir, "RUNNING"))
                print(f"{time.strftime('%H:%M')}  {v} {'finished' if ok else 'FAILED (see log)'}"
                      f"  [{len(running)} running, {len(todo)} waiting]", flush=True)
        while todo and len(running) < a.jobs:
            free = mem_available_gb()
            if running and free < a.min_free_gb:
                break                               # wait for memory before starting another
            v = todo.pop(0)
            cmd = ["nice", "-n", "10", sys.executable, os.path.abspath(__file__),
                   "--drive", a.drive, "--visits", v, "--child",
                   "--s1-ncpu", str(a.s1_ncpu), "--s3-ncpu", str(a.s3_ncpu),
                   "--max-cores", a.max_cores, "--jobs", str(a.jobs)]
            if a.from_rateints:
                cmd.append("--from-rateints")
            if a.offline:
                cmd.append("--offline")
            if a.tag:
                cmd += ["--tag", a.tag]
            if a.start_stage > 1:
                cmd += ["--start-stage", str(a.start_stage)]
            if a.set:
                cmd += ["--set"] + a.set
            log = open(os.path.join(logdir, f"{v}.log"), "w")
            running[v] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            print(f"{time.strftime('%H:%M')}  {v} started  (free memory {free:.0f} GB)", flush=True)
            time.sleep(60)                          # let it load data before judging free memory
        time.sleep(20)
    print(f"All done: {len(visits) - nfail} reduced, {nfail} failed. "
          f"Check with --status.", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", default=os.environ.get("TRAPPIST1_DATA", "."))
    ap.add_argument("--visits", nargs="+")
    ap.add_argument("--programs", nargs="+")
    ap.add_argument("--from-rateints", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--prefetch-crds", action="store_true")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--s1-ncpu", type=int, default=1)
    ap.add_argument("--s3-ncpu", type=int, default=2)
    ap.add_argument("--max-cores", default="none", choices=["none", "quarter", "half", "all"])
    ap.add_argument("--min-free-gb", type=float, default=12.0)
    ap.add_argument("--start-stage", type=int, default=1, choices=[1, 2, 3, 4],
                    help="first stage to run; earlier stages must already be done (default 1)")
    ap.add_argument("--tag", help="write to reduction_<tag>/ instead of reduction/ (for alternative settings)")
    ap.add_argument("--set", nargs="+", default=[], metavar="S<n>:KEY=VALUE",
                    help="override control-file settings, e.g. S1:update_sat_flags=False")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.drive = os.path.abspath(os.path.expanduser(a.drive))

    if not os.environ.get("CRDS_PATH") or not os.environ.get("CRDS_SERVER_URL"):
        sys.exit("Set CRDS_PATH and CRDS_SERVER_URL first (see README.md, section Installation).")
    if a.offline:
        os.environ["CRDS_MODE"] = "local"          # use the cache only, never contact the server
    level = "rateints" if a.from_rateints else "uncal"
    base = "reduction_rateints" if a.from_rateints else "reduction"
    if a.tag:
        base = f"{base}_{a.tag}"
    overrides = {}
    for item in a.set:
        st, kv = item.split(":", 1)
        key, val = kv.split("=", 1)
        overrides.setdefault(int(st.lstrip("Ss")), {})[key] = val
    visits = find_visits(a.drive, level, a.programs)
    if a.visits:
        missing = [v for v in a.visits if v not in visits]
        if missing:
            print("Not found on the drive:", " ".join(missing))
        visits = {v: visits[v] for v in a.visits if v in visits}

    if a.list or not visits:
        print(f"{len(visits)} visits with {level} files under {a.drive}:")
        for v, (p, rel) in visits.items():
            n = len(glob.glob(os.path.join(a.drive, rel, f"*_{level}.fits")))
            print(f"  {v}  (GO {int(p)}, {n} segment files)")
        return
    if a.status:
        states = {v: visit_state(a.drive, base, v) for v in visits}
        for v, s in states.items():
            print(f"  {v}  GO {int(visits[v][0]):<5d} {s:8s} {last_stage(a.drive, base, v) if s == 'running' else ''}")
        counts = {s: list(states.values()).count(s) for s in ("done", "running", "waiting", "FAILED")}
        print("  " + ", ".join(f"{k}: {n}" for k, n in counts.items()) +
              f"   | free memory {mem_available_gb():.0f} GB")
        return
    if a.prefetch_crds:
        sys.exit(prefetch_crds(a.drive, visits, level))

    cores = {"max_cores": a.max_cores, "s1_ncpu": a.s1_ncpu, "s3_ncpu": a.s3_ncpu,
             "s3_maxmem": min(0.5, 0.8 / max(a.jobs, 1))}
    if a.jobs > 1 and not a.child:
        run_parallel(a, visits, base)
        return

    for k, (visit, (prog, rel)) in enumerate(visits.items(), 1):
        vdir = os.path.join(a.drive, base, visit)
        if os.path.exists(os.path.join(vdir, "S4_DONE")) and not a.redo:
            print(f"[{k}/{len(visits)}] {visit}: already reduced, skipped")
            continue
        os.makedirs(vdir, exist_ok=True)
        for m in ("FAILED", "S4_DONE"):
            if os.path.exists(os.path.join(vdir, m)):
                os.remove(os.path.join(vdir, m))
        open(os.path.join(vdir, "RUNNING"), "w").write(f"{os.getpid()} {time.ctime()}\n")
        print(f"[{k}/{len(visits)}] {visit} (GO {int(prog)}) started {time.ctime()}", flush=True)
        try:
            reduce_visit(visit, rel, a.drive, a.from_rateints, cores, base, overrides, a.start_stage)
        except Exception:
            print(f"  {visit} FAILED:")
            traceback.print_exc()
            open(os.path.join(vdir, "FAILED"), "w").write(time.ctime() + "\n")
            if a.child:
                sys.exit(1)
            print("  continuing with the next visit", flush=True)
        finally:
            if os.path.exists(os.path.join(vdir, "RUNNING")):
                os.remove(os.path.join(vdir, "RUNNING"))


if __name__ == "__main__":
    main()
