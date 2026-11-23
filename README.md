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

Authors: Muhammad Waqas ([ORCID 0009-0002-1134-2666](https://orcid.org/0009-0002-1134-2666))
and Taufiq Hidayat ([ORCID 0009-0008-8045-6054](https://orcid.org/0009-0008-8045-6054)),
Institut Teknologi Bandung.
This repository accompanies a paper in preparation by M. Waqas and T. Hidayat.

## Contents

| File | Purpose |
|---|---|
| `jwst_download.py` | Finds and downloads the `uncal` or `rateints` files (NRS1, PRISM) of the three programs from MAST |
| `run_reduction.py` | Runs Eureka! Stages 1–4 for each visit, one at a time or several in parallel |
| `start_tmux.sh` | Starts `run_reduction.py` in a tmux window, so a reduction on a remote server continues after logout |
| `fit_whitelight_gp.py` | The same fits with a Gaussian-process noise model (celerite2, Matern-3/2) and MCMC sampling (emcee); the version used for final results |
| `fit_spectra.py` | Per-visit transmission spectra of b and e in the 67 bins: common-mode correction from the white-light GP fit, then a transit fit per bin |
| `combine_spectra.py` | Combines the visits of e with and without the correction for stellar contamination using b, and compares the GO 1331 visits with DREAMS |
| `fit_whitelight.py` | Fits the white-light transits of TRAPPIST-1 b and e in every reduced visit (batman, joint fit with baseline and drift detrending) |
| `make_paper_figures.py` | Rebuilds every figure of the paper from the outputs of the scripts above |
| `templates/S1_trappist1.ecf` … `S4_trappist1.ecf` | Eureka! control-file templates; `run_reduction.py` fills in the paths and core settings for each visit |
| `results/` | White-light fits and combined transmission spectra from all 18 visits (see `results/README.md`) |
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
`reduction_rateints/`, as an independent check of Stage 1. To test other settings
without touching the main reduction, use `--tag NAME` (outputs go to `reduction_NAME/`)
with `--set S1:KEY=VALUE` overrides, e.g.
`--tag nosat --set S1:update_sat_flags=False S1:expand_prev_group=False`; the analysis
scripts read an alternative reduction when `TRAPPIST1_REDUCTION=reduction_NAME` is set.

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

## 3. White-light transit fits

```bash
python fit_whitelight.py --drive WORKDIR                   # all visits marked S4_DONE
python fit_whitelight.py --drive WORKDIR --visits jw01331001001 --npoly 2
```

For each visit the script finds the transits, identifies TRAPPIST-1 b and e from
linear ephemerides (with a tolerance for transit-timing variations) and their shape,
masks transits of the other planets, and fits all b and e transits jointly:

| Parameter | Treatment |
|---|---|
| Mid-transit time, Rp/R* | free, per planet |
| Quadratic limb darkening | free, shared by the planets (Kipping 2013 q1, q2) |
| Baseline | a polynomial in time for each continuous segment of the light curve (`--npoly`, default 2; order 1 for segments shorter than 2.5 h) plus linear terms in the trace position and width recorded by Eureka! |
| Period, a/R*, inclination | fixed to Agol et al. (2021), Table 5 |

Transits of other planets and events such as flares are left out of the fit; the
known events are listed in `EXCLUDE` at the top of the script, and further windows
can be given with `--exclude VISIT:START-END` (BMJD_TDB). Excluded stretches split
the light curve into segments, each with its own baseline.
Points more than `--clip` (default 5) robust sigma from the first fit are removed
and the fit is repeated. Uncertainties come from the covariance matrix, scaled by
the reduced chi-square when it exceeds 1 and by the time-averaging red-noise factor
beta (bins of 5–30 min). Results are written to `WORKDIR/analysis/whitelight/`:
`whitelight_fits.csv`, one figure per visit and `summary_rp.png`.

### Gaussian-process fits

```bash
python fit_whitelight_gp.py --drive WORKDIR --processes 8      # visits fitted in parallel
```

`fit_whitelight_gp.py` uses the same input, planet identification, exclusion windows
and fixed orbital parameters, but models time-correlated noise (stellar granulation,
small flares, slow trends) with a celerite2 Matern-3/2 Gaussian process plus a jitter
term, on light curves binned to 30 s. The baseline is a linear polynomial per segment
plus the trace-drift terms; all parameters are sampled with emcee (48 walkers, 8,000
steps, 2,000 discarded), so the depths are marginalised over the baseline and noise
model. On synthetic visits with 150 ppm correlated noise (15 min time scale) on top of
371 ppm white noise, it recovers the injected radius ratios with a reduced chi-square of
about 1 across visits. Results go to `WORKDIR/analysis/whitelight_gp/`.

## 4. Transmission spectra

```bash
python fit_spectra.py --drive WORKDIR --processes 8
python combine_spectra.py --drive WORKDIR --dreams-dir path/to/espinoza+2025/appendix-figure1/spectra
```

`fit_spectra.py` divides each spectroscopic light curve by the white-light systematics of
its visit (baseline + Gaussian process, from the maximum-probability sample of the
`fit_whitelight_gp.py` posterior), which removes the achromatic instrumental and stellar
noise, and fits each bin by least squares with the mid-transit times fixed, Rp/R* of each
planet and quadratic limb darkening free, and a linear baseline per segment. Errors are
scaled by sqrt(reduced chi-square) and by the time-averaging beta factor of each bin.
`--common-mode data` uses the classic divide-white correction instead. The per-visit
spectra can carry a small achromatic offset relative to the white-light depth; absolute
depths should be taken from the white-light fits, and the offsets are removed when visits
are combined.

The chains of `fit_whitelight_gp.py` are stored in double precision. Chains written by
versions before November 2026 are single precision, which holds the mid-transit times
only to 0.0039 d (5.6 min); `fit_spectra.py` and `make_paper_figures.py` refine the
maximum-probability sample of such chains by a local optimization of the posterior before
use, so they need not be rerun. On synthetic visits the refined parameters reproduce the
spectra from double-precision chains to 0.01 sigma per bin.

`combine_spectra.py` writes four products: `e_uncorrected` (all visits of e),
`b_combined`, `e_bcorr_ratio` (each visit of e divided by eps = D_b / <D_b> from the b
transit of the same visit, following the close-transit method of Allen et al. 2026) and
`e_bcorr_diff` (each visit of e multiplied by the mean b spectrum over that visit's b
spectrum, removing only the visit-to-visit changes in contamination). Visits are combined
after removing each visit's achromatic offset, with errors inflated by the visit-to-visit
scatter. Errors of b and e from the same visit are propagated as independent, which is
conservative because both are divided by the same common mode. On synthetic visits with
stellar contamination that differs between visits, the correction reduces the
visit-to-visit scatter of e from 3.3 to 0.6 times the expected value.

## 5. Figures of the paper

```bash
python make_paper_figures.py --drive WORKDIR \
    --dreams-dir path/to/espinoza+2025/appendix-figure1/spectra \
    --dreams-combined path/to/tspectra_decontaminated_corrected_flat_JWST-CLR-R10000.txt \
    --models path/to/model_fits_all
```

writes `lightcurves.pdf`, `radii.pdf`, `oc.pdf`, `comparison_dreams.pdf`,
`combined_spectra.pdf` and `model_comparison.pdf` to `WORKDIR/analysis/paper_figures/`.
`--only` selects figures (`lc radii oc dreams combined models`); figures whose inputs are
not given are skipped. The light-curve figure needs the Stage 4 outputs and the GP
chains; the others need only the CSV files written by the analysis scripts.
`--models` is a folder with the model spectra binned to the 67 bins
(`model_NNNN_jwst_fit.csv`, column `model_depth_ppm`) and the table of model
parameters `MC_2000_master_results_flatspec.csv` (column `P_CO2`).

## Visit-specific settings

All visits are reduced with the templates above, except `jw09256010001`, where about 45 integrations
contain no valid spectrum. For that visit Stages 3 and 4 were rerun with
`--start-stage 3 --set S3:record_ypos=False` and `--start-stage 4 --set S4:mad_sigma=None`, which skip
the per-integration Gaussian fit of the trace position and the bad-column search; both steps fail on
fully empty integrations. Time windows excluded from the light-curve fits (a flare in `jw06456001001`
and an overlapping transit-like event in `jw09256007001`) are listed in `EXCLUDE` in `fit_whitelight.py`.

## References

- Agol, E., et al. 2021, *Refining the Transit-timing and Photometric Analysis of TRAPPIST-1*, PSJ, 2, 1
- Foreman-Mackey, D., et al. 2017, *Fast and Scalable Gaussian Process Modeling with Applications to Astronomical Time Series* (celerite), AJ, 154, 220
- Foreman-Mackey, D., et al. 2013, *emcee: The MCMC Hammer*, PASP, 125, 306
- Kipping, D. M. 2013, *Efficient, uninformative sampling of limb darkening coefficients for two-parameter laws*, MNRAS, 435, 2152
- Kreidberg, L. 2015, *batman: BAsic Transit Model cAlculatioN in Python*, PASP, 127, 1161
- Allen, N. H., Espinoza, N., et al. 2026, *JWST TRAPPIST-1 e/b Program: Motivation and First Observations*, AJ, 171, 105
- Espinoza, N., et al. 2025, *JWST-TST DREAMS: NIRSpec/PRISM Transmission Spectroscopy of the Habitable Zone Planet TRAPPIST-1 e*, arXiv:2509.05414
- Bell, T. J., et al. 2022, *Eureka!: An End-to-End Pipeline for JWST Time-Series Observations*, JOSS, 7, 4503
- Bushouse, H., et al., *JWST Calibration Pipeline*, Zenodo
