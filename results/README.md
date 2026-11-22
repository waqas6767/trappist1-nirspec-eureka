# Results

Products of the analysis of the 18 NIRSpec/PRISM visits of TRAPPIST-1 (GO 1331, 6456, 9256),
produced with the scripts in this repository.

| File | Contents |
|---|---|
| `whitelight_gp_fits.csv` | White-light GP + MCMC fit of every transit: mid-time (BMJD_TDB), Rp/R*, depth, limb darkening, GP amplitude and time scale, jitter, chain diagnostics |
| `trappist1_e_uncorrected.csv` | Transmission spectrum of TRAPPIST-1 e combined over 17 visits |
| `trappist1_b_combined.csv` | Transmission spectrum of TRAPPIST-1 b combined over 14 visits |
| `trappist1_e_bcorr_ratio.csv` | Spectrum of e divided by the normalized spectrum of b of the same visit (13 visits) |
| `trappist1_e_bcorr_diff.csv` | Spectrum of e with only the visit-to-visit changes measured by b removed (13 visits) |

Spectra: columns `wave_low_um`, `wave_high_um`, `wave_mid_um`, `depth_ppm`, `err_ppm`, in the 67 bins
of Espinoza et al. (2025). Each visit's achromatic offset is removed before combining, so the absolute
level of the spectra is arbitrary to within ~100 ppm; use the white-light depths for absolute values.
Uncertainties include the visit-to-visit scatter. Bins without a value (`nan`) had no valid fit in any visit.

Visits with special treatment: in `jw06456001001` a flare during the egress of e is excluded; in
`jw09256007001` the transit of e overlaps a transit-like event and only b is fitted; in `jw09256010001`
about 45 empty integrations required disabling the per-integration trace-position fit (Stage 3,
`record_ypos=False`) and the bad-column search (Stage 4, `mad_sigma=None`).
