# Baseline Comparison (`h=8`)

This note summarizes the non-OT comparison methods included in this package.

## Included comparators

- `shared_ot_guided_das/`
  - 3-seed shared-OT-guided DAS on the same `structured_26` benchmark family used for the frozen `h=8` OT package
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

Shared-OT-guided DAS (`h=8`, 3 seeds):
- internal-carry mean: `0.9775`
- `C3 = 0.9522`
- mean runtime: `84.7s`
- mean speedup vs matched full DAS: `3.21x`

Shared-OT-guided DAS (`h=16`, 3 seeds):
- internal-carry mean: `0.9795`
- `C3 = 0.9565`
- mean runtime: `174.9s`
- mean speedup vs matched full DAS: `6.18x`

Matched full DAS on `C1,C2,C3` (`h=8`, 3 seeds):
- internal-carry mean: `0.9746`
- `C3 = 0.9508`
- mean runtime: `271.8s`

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

- anchored OT is the strongest standalone OT variant in this package
- shared-OT-guided DAS is the strongest new hybrid result: it stays close to matched full DAS while running much faster
- the `h=16` scaling run strengthens the runtime story: the mean speedup grows from `3.21x` to `6.18x`
- DAS is still substantially stronger than OT when OT is used as the final intervention method
- among the MIB-style baselines, `DBM+PCA` is the strongest non-DAS comparator on `h=8`
- `Full Vector` roughly matches the regular shared-bank OT result, while `DBM` and especially `DBM+PCA` outperform it

## Where to look

- `shared_ot_guided_das/README.md`
- `shared_ot_guided_das/RUN_LOG.md`
- `shared_ot_guided_das/three_seed_summary_compact.json`
- `shared_ot_guided_das_h16/README.md`
- `shared_ot_guided_das_h16/RUN_LOG.md`
- `shared_ot_guided_das_h16/three_seed_summary_compact.json`
- `das/README.md`
- `mib_baselines/README.md`
- `das/best_result_compact.json`
- `mib_baselines/best_result_compact.json`
