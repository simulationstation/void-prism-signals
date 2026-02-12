# void-prism-signals

Void-prism signal pipeline extracted into a standalone repository.

This repo contains the scripts and minimal Python module set needed to:

1. Build tomographic `theta` maps from ACT + SDSS kSZx inputs.
2. Measure a z-binned / radius-binned void-prism `E_G` observable with jackknife covariance.
3. Score posterior-predictive MG embeddings against an internal GR baseline.

## Repository Layout

- `scripts/build_theta_maps_tomo_from_act_dr6_sdss_kszx.py`
- `scripts/build_void_prism_eg_measurement.py`
- `scripts/measure_void_prism_eg_suite_jackknife.py`
- `scripts/run_void_prism_eg_joint_test.py`
- `scripts/run_void_prism_fast_signal_battery.py`
- `src/entropy_horizon_recon/` minimal transitive module set for the above scripts
- `artifacts/ancillary/void/` bundled example result artifacts

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e .[healpix]
```

Optional (only for `build_theta_maps_tomo_from_act_dr6_sdss_kszx.py`):

```bash
pip install -e .[theta_builder]
# plus kszx (install separately if not on your index)
```

## Pipeline Usage

### 1) Build theta tomographic maps (optional heavy stage)

```bash
PYTHONPATH=src python scripts/build_theta_maps_tomo_from_act_dr6_sdss_kszx.py \
  --out-dir outputs/theta_tomo_kszx
```

### 2) Measure void-prism suite and joint covariance

```bash
PYTHONPATH=src python scripts/measure_void_prism_eg_suite_jackknife.py \
  --void-csv data/processed/void_prism/boss_dr12_voids_mao2017.csv \
  --theta-fits /path/to/theta_bin0.fits,/path/to/theta_bin1.fits,/path/to/theta_bin2.fits,/path/to/theta_bin3.fits \
  --theta-mask-fits /path/to/mask_bin0.fits,/path/to/mask_bin1.fits,/path/to/mask_bin2.fits,/path/to/mask_bin3.fits \
  --planck \
  --z-edges 0.2,0.36,0.48,0.56,0.67 \
  --bin-edges 0,20,50,100,200,400,700 \
  --jackknife-nside 8 \
  --out-base outputs/void_prism_suite
```

### 3) Joint MG-vs-GR scoring

`run_void_prism_eg_joint_test.py` requires explicit `--embedding` selection.

```bash
PYTHONPATH=src python scripts/run_void_prism_eg_joint_test.py \
  --run-dir /path/to/M0_start101 \
  --run-dir /path/to/M0_start202 \
  --suite-json artifacts/ancillary/void/void_prism_three_source_suite_joint.json \
  --embedding minimal \
  --embedding slip_allowed \
  --embedding screening_allowed \
  --eta0 1.12 --eta1 -0.18 \
  --env-proxy 1.0 --env-alpha 0.25 \
  --muP-highz 1.05 \
  --fit-amplitude \
  --out outputs/void_prism_joint
```

### 4) Fast signal battery (efficient diagnostic tests)

This runs:
- block-permutation nulls (misalignment null),
- block-sign nulls (coherent-sign null),
- leave-one-block-out robustness,
- z/radius split consistency.

```bash
PYTHONPATH=src python scripts/run_void_prism_fast_signal_battery.py \
  --run-dir /path/to/M0_start101 \
  --run-dir /path/to/M0_start202 \
  --run-dir /path/to/M0_start303 \
  --run-dir /path/to/M0_start404 \
  --run-dir /path/to/M0_start505 \
  --suite-json artifacts/ancillary/void/void_prism_three_source_suite_joint.json \
  --embedding minimal \
  --embedding slip_allowed \
  --embedding screening_allowed \
  --fit-amplitude \
  --max-draws 256 \
  --n-perm 200 \
  --n-sign 200 \
  --out outputs/void_prism_signal_battery
```

## Bundled Results

### Legacy 5-seed minimal-embedding run

Artifact:
- `artifacts/ancillary/void/void_prism_three_source_results.json`

Summary (`delta_lpd_vs_gr`):
- Values: `[+0.0116, +0.0198, +0.0249, +0.0127, +0.0221]`
- Mean: `+0.0182`
- Min/Max: `+0.0116 / +0.0249`

Interpretation: all seeds are same-sign positive, but near-tie scale.

### Explicit multi-embedding 5x3 run (Feb 12, 2026 UTC)

Artifact:
- `artifacts/ancillary/void/void_prism_joint_explicit_multiembed_20260212_001503UTC/tables/results.json`

Settings:
- `fit_amplitude=true`
- `report_both_amplitudes=true`
- `eta0=1.12`, `eta1=-0.18`
- `env_proxy=1.0`, `env_alpha=0.25`
- `muP_highz=1.05`
- `max_draws=2048`

Mean `delta_lpd_vs_gr` by embedding:
- `minimal`: `+0.04233` (range `+0.04015` to `+0.04350`)
- `slip_allowed`: `+0.04139` (range `+0.03914` to `+0.04260`)
- `screening_allowed`: `+0.04898` (range `+0.04685` to `+0.05010`)

### Fast battery run (Feb 12, 2026 UTC)

Run output:
- `artifacts/ancillary/void/void_prism_signal_battery_20260212_003206UTC/tables/battery_results.json`

Key outcomes:
- All embeddings show positive observed deltas in all 5 seeds (`positive_delta_fraction=1.0`).
- Split consistency is positive in all available splits (`split_all_positive_fraction=1.0`).
- Leave-one-block-out is **not** all-positive (`loo_all_positive_fraction=0.0`), indicating block-level fragility.
- Fast null tests are not extreme:
  - permutation null mean upper-tail p-values: `~0.27-0.30`
  - sign null mean upper-tail p-values: `~0.53-0.56`

Interpretation:
- Directional consistency exists, but these efficient nulls do **not** yet indicate a strongly isolated signal.

### Fast battery rerun with placebo + leave-two-out (Feb 12, 2026 UTC)

Run output:
- `artifacts/ancillary/void/void_prism_signal_battery_l2o_placebo_20260212_rerun/tables/battery_results.json`

Settings:
- `fit_amplitude=true`
- `max_draws=256`
- `n_perm=500`, `n_sign=500`, `n_placebo=500`

Key outcomes:
- `positive_delta_fraction=1.0` remains true for all embeddings.
- Data-side placebo null p-values remain non-extreme:
  - `minimal`: `~0.307`
  - `slip_allowed`: `~0.302`
  - `screening_allowed`: `~0.293`
- `L2O all-positive fraction = 0.0` for all embeddings.
- In every seed/embedding, the same dropped pair is worst:
  - `zbin0_small_z0.200-0.360` + `zbin1_small_z0.360-0.480`

Interpretation:
- The sign is stable, but the signal still does not cleanly reject placebo/null constructions and remains sensitive to specific low-z/small-Rv blocks.

## Notes

- This repository preserves the original module namespace (`entropy_horizon_recon`) to minimize code drift from the source pipeline.
- The GR baseline remains explicit and is used as the internal reference for `delta_lpd_vs_gr`.
- The no-amplitude score channel can show numerically huge values depending on scale mismatch; the primary reported channel in bundled runs is amplitude-fitted (`--fit-amplitude`).

## License

MIT (see `LICENSE`).
