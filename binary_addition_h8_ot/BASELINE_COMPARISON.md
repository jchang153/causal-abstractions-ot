# Baseline Comparison (`h=8`)

This note summarizes the non-OT comparison methods included in this package.

## Included comparators

- `das/`
  - anchor-prefix DAS on the same `structured_26` benchmark family used for the frozen `h=8` OT package
- `mib_baselines/`
  - MIB-style baselines on the same `structured_26` benchmark family:
    - `full_vector`
    - `dbm`
    - `dbm_pca`
    - `dbm_sae`

## Main numbers

Anchored OT (`h=8`):
- carry subset combined: `0.8643`
- internal-carry mean (`C1,C2,C3`): `0.8190`
- measured runtime: `275.7s`

Anchor-prefix DAS (`h=8`):
- carry subset combined: `0.9886`
- internal-carry mean: `0.9848`
- `C3 = 0.9597`
- measured runtime: `237.9s`

MIB-style baselines (`h=8`, structured_26):
- `full_vector`
  - carry subset combined: `0.8181`
  - internal-carry mean: `0.7575`
  - `C3 = 0.7287`
  - runtime: `12.5s`
- `dbm`
  - carry subset combined: `0.8605`
  - internal-carry mean: `0.8105`
  - `C3 = 0.8003`
  - runtime: `41.3s`
- `dbm_pca`
  - carry subset combined: `0.9099`
  - internal-carry mean: `0.8798`
  - `C3 = 0.8434`
  - runtime: `42.6s`
- `dbm_sae`
  - carry subset combined: `0.8473`
  - internal-carry mean: `0.7964`
  - `C3 = 0.7429`
  - runtime: `48.8s`

## Interpretation

- anchored OT is the strongest OT variant in this package
- DAS is still substantially stronger than OT on the same repaired benchmark
- among the MIB-style baselines, `DBM+PCA` is the strongest comparator on `h=8`
- `Full Vector` roughly matches the regular shared-bank OT result, while `DBM` and especially `DBM+PCA` outperform it

## Where to look

- `das/README.md`
- `mib_baselines/README.md`
- `das/best_result_compact.json`
- `mib_baselines/best_result_compact.json`
