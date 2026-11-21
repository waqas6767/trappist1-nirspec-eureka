# TRAPPIST-1 NIRSpec PRISM transits: Eureka! reduction

Reduction of the JWST NIRSpec PRISM time-series observations of TRAPPIST-1 from the raw
`uncal` ramps to white-light and spectroscopic light curves, using
[Eureka!](https://eurekadocs.readthedocs.io) v1.4 and the STScI `jwst` pipeline.

| Program | Description |
|---|---|
| GO 1331 | JWST-TST DREAMS transits of TRAPPIST-1 e (Espinoza et al. 2025) |
| GO 6456 | TRAPPIST-1 e and b transits, Cycle 3 |
| GO 9256 | TRAPPIST-1 e and b transits, Cycle 4 |

All visits use NIRSpec PRISM/CLEAR in the SUB512 subarray with the NRSRAPID readout; the
spectrum falls on detector NRS1. Spectroscopic light curves are built in the 67 wavelength
bins of Espinoza et al. (2025), so the results can be compared bin by bin with the DREAMS
spectrum.

Author: Muhammad Waqas ([ORCID 0009-0002-1134-2666](https://orcid.org/0009-0002-1134-2666)).
This repository accompanies a paper in preparation.

## Contents

| File | Purpose |
|---|---|
| `jwst_download.py` | Finds and downloads the `uncal` or `rateints` files (NRS1, PRISM) of the three programs from MAST |
| `run_reduction.py` | Runs Eureka! Stages 1–4 for each visit, one at a time or several in parallel |
| `start_tmux.sh` | Starts `run_reduction.py` in a tmux window, so a reduction on a remote server continues after logout |
| `templates/S1_trappist1.ecf` … `S4_trappist1.ecf` | Eureka! control-file templates; `run_reduction.py` fills in the paths and core settings for each visit |
| `dreams_bins.txt`, `dreams_edges.npy` | The 67 wavelength bins (0.6185–5.2935 µm) of Espinoza et al. (2025) |

## Installation

Eureka! and `jwst` are installed in their own conda environment. The versions below are the
ones used for this work.

```bash
conda create -y -p ./envs/eureka -c conda-forge python=3.11 "numpy<2" "pyarrow<20" pip git
conda activate ./envs/eureka
python -m pip install "eureka-bang[jwst] @ git+https://github.com/kevin218/Eureka.git@v1.4" "numpy<2" "pyarrow<20"
python -c "import eureka, jwst; print('Eureka!', eureka.__version__, '| jwst', jwst.__version__)"
# Eureka! 1.4 | jwst 2.0.1
```

`pyarrow` is pinned below 20 because newer releases require NumPy 2, which `jwst` 2.0.1 does
not support. On older Linux systems (e.g. Ubuntu 18.04), installing the compiled packages
from conda-forge, as above, avoids glibc incompatibilities of some pip wheels.

The pipeline downloads calibration reference files from CRDS. Choose a cache folder on a
local disk with a few GB free and set:

```bash
export CRDS_PATH=$HOME/crds_cache
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
```

These settings can be kept in a file `env.sh` in the working folder; `start_tmux.sh`
reads it automatically.

## Folder layout

```
WORKDIR/
├── jwst_data/uncal/<program>/<visit>/*_nrs1_uncal.fits        # written by jwst_download.py
├── jwst_data/rateints/<program>/<visit>/*_nrs1_rateints.fits
├── reduction/<visit>/Stage1 … Stage4                          # Eureka! outputs and figures
├── reduction/<visit>/ecf/                                     # control files used for the visit
├── reduction/logs/<visit>.log                                 # one log per visit
└── trappist1-nirspec-eureka/                                  # this repository
```

## 1. Download the data

```bash
python jwst_download.py --outdir WORKDIR/jwst_data --check            # sizes only
python jwst_download.py --outdir WORKDIR/jwst_data --level uncal      # raw ramps
python jwst_download.py --outdir WORKDIR/jwst_data --level rateints   # optional, for a Stage 2 cross-check
python jwst_download.py --outdir WORKDIR/jwst_data --level uncal --verify
```

Files already present with the size listed by MAST are skipped, so an interrupted download
can be restarted with the same command.

## 2. Reduce

```bash
python run_reduction.py --drive WORKDIR --list                 # visits found, with segment counts
python run_reduction.py --drive WORKDIR --prefetch-crds        # fetch all reference files once
python run_reduction.py --drive WORKDIR --visits jw01331001001 # one visit, to check the settings
python run_reduction.py --drive WORKDIR --jobs 6 --s1-ncpu 4 --s3-ncpu 6 --offline
python run_reduction.py --drive WORKDIR --status               # progress table
```

The steps for each visit are:

| Stage | Input → output | Main settings |
|---|---|---|
| 1 | `uncal` → `rateints` | jump threshold 15, saturation flags updated and expanded to the previous group, group-level background (1/f) subtraction with the curved trace masked |
| 2 | `rateints` → `calints` | wavelength assignment; flat field, photometric calibration and 1-D extraction of `jwst` skipped |
| 3 | `calints` → 1-D spectra | curvature correction, background from rows 8 px from the trace, optimal extraction (half-width 4 px), x-window 10–505 |
| 4 | 1-D spectra → light curves | white light curve (0.6185–5.2935 µm) and 67 spectroscopic channels, 7σ outlier clipping, drift recorded for detrending |

Check the Stage 3 figures of the first visit (`reduction/<visit>/Stage3/figs/`) and adjust
`ywindow`, `xwindow`, `spec_hw` and `bg_hw` in `templates/S3_trappist1.ecf` if the trace or
the background region is not where expected.

A visit whose Stage 4 finished is marked `S4_DONE` and is skipped later (`--redo` reduces it
again); a visit that failed is marked `FAILED` and is retried on the next run.
`--from-rateints` starts from the MAST `rateints` files at Stage 2 and writes to
`reduction_rateints/`, as an independent check of Stage 1.

### Options of `run_reduction.py`

| Option | Meaning | Default |
|---|---|---|
| `--drive` | working folder containing `jwst_data/` | `$TRAPPIST1_DATA`, else `.` |
| `--programs` | restrict to programs, e.g. `06456 09256` | all |
| `--visits` | restrict to visits, e.g. `jw06456002001` | all |
| `--jobs` | visits reduced at the same time, each in its own process | 1 |
| `--s1-ncpu` | cores per visit for the Stage 1 background subtraction | 1 |
| `--max-cores` | cores per visit for `jwst` ramp fitting (`none`, `quarter`, `half`, `all`) | `none` |
| `--s3-ncpu` | cores per visit for Stage 3 | 2 |
| `--min-free-gb` | a new visit starts only when this much memory is free | 12 |
| `--offline` | use only the local CRDS cache (`CRDS_MODE=local`), after `--prefetch-crds` | off |

### Memory and cores

Stage 1 is the most memory-intensive step: one visit with `--s1-ncpu 1` and
`--max-cores none` uses about 6 GB. Each additional core used by `jwst` ramp fitting
(`--max-cores`) holds its own copy of the data, so on machines with less than ~16 GB keep
the defaults and `--jobs 1`. In parallel mode, Stage 3 `max_memory` is set to 0.8/jobs,
child processes run at low priority (`nice 10`) so other users of a shared machine are not
slowed down, and a new visit is started only when `--min-free-gb` of memory is available.

### Running on a remote server

```bash
bash start_tmux.sh --visits jw01331001001 --s1-ncpu 4 --s3-ncpu 8   # test visit
bash start_tmux.sh --jobs 6 --s1-ncpu 4 --s3-ncpu 6 --offline       # all visits
```

The script opens a tmux window named `reduce`, sources `WORKDIR/env.sh` if present, and
uses `WORKDIR/envs/eureka/bin/python` if that environment exists (set `PY` to use another
interpreter). Detach with `Ctrl+B` then `d`; the reduction continues after logout.
Running the script again resumes where the previous run stopped.

## References

- Bell, T. J., et al. 2022, *Eureka!: An End-to-End Pipeline for JWST Time-Series Observations*, JOSS, 7, 4503
- Espinoza, N., et al. 2025, ApJL, JWST-TST DREAMS transmission spectrum of TRAPPIST-1 e (GO 1331)
- Bushouse, H., et al., *JWST Calibration Pipeline*, Zenodo
