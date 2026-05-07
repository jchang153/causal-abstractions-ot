# Run Log

This log records the exported 3-seed `h=8` shared-OT-guided DAS run.

## Seed-level summary

| Seed | Best mask | Guided internal-carry | Guided C3 | Full DAS best r | Full DAS internal-carry | Full DAS C3 | Hybrid runtime (s) | Full DAS runtime (s) | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | StepMask | 0.9772 | 0.9323 | 8 | 0.9591 | 0.9114 | 119.78 | 280.64 | 2.34x |
| 1 | StepMask | 0.9807 | 0.9577 | 8 | 0.9868 | 0.9680 | 66.03 | 237.92 | 3.60x |
| 2 | S90 | 0.9747 | 0.9665 | 8 | 0.9780 | 0.9730 | 68.21 | 296.94 | 4.35x |

## Aggregate

- guided internal-carry mean: `0.9775`
- guided `C3`: `0.9522`
- full DAS internal-carry mean: `0.9746`
- full DAS `C3`: `0.9508`
- mean hybrid runtime: `84.67s`
- mean full DAS runtime: `271.83s`
- mean speedup: `3.21x`

## Shared OT discovery details

### Seed 0

- `C1`: best OT row fit at r=2, epsilon=0.01, top_k=2, lambda=2.0, sites h_0[0,1] + h_0[4,5]; dominant timestep `h_0` with stability `1.0`
- `C2`: best OT row fit at r=8, epsilon=0.01, top_k=1, lambda=1.0, site h_1[0,1,2,3,4,5,6,7]; dominant timestep `h_1` with stability `1.0`
- `C3`: best OT row fit at r=8, epsilon=0.01, top_k=1, lambda=1.0, site h_2[0,1,2,3,4,5,6,7]; dominant timestep `h_2` with stability `1.0`
- best guided mask mode: `StepMask`
- hybrid runtime: `119.78s`; full DAS runtime: `280.64s`

### Seed 1

- `C1`: best OT row fit at r=1, epsilon=0.01, top_k=2, lambda=2.0, sites h_0[1] + h_0[6]; dominant timestep `h_0` with stability `1.0`
- `C2`: best OT row fit at r=8, epsilon=0.01, top_k=1, lambda=1.0, site h_1[0,1,2,3,4,5,6,7]; dominant timestep `h_1` with stability `1.0`
- `C3`: best OT row fit at r=8, epsilon=0.01, top_k=1, lambda=1.0, site h_2[0,1,2,3,4,5,6,7]; dominant timestep `h_2` with stability `1.0`
- best guided mask mode: `StepMask`
- hybrid runtime: `66.03s`; full DAS runtime: `237.92s`

### Seed 2

- `C1`: best OT row fit at r=8, epsilon=0.03, top_k=1, lambda=1.0, site h_0[0,1,2,3,4,5,6,7]; dominant timestep `h_0` with stability `1.0`
- `C2`: best OT row fit at r=8, epsilon=0.01, top_k=1, lambda=1.0, site h_1[0,1,2,3,4,5,6,7]; dominant timestep `h_1` with stability `0.5`
- `C3`: best OT row fit at r=8, epsilon=0.03, top_k=1, lambda=1.0, site h_2[0,1,2,3,4,5,6,7]; dominant timestep `h_2` with stability `1.0`
- best guided mask mode: `S90`
- hybrid runtime: `68.21s`; full DAS runtime: `296.94s`

## Observations

- `C2` and `C3` keep choosing coarse `r=8` discovery handles across all three seeds.
- Adding `r=1` and `r=2` mainly affected `C1`; the pooled support still collapsed to the correct timestep story `C1 -> h_0`, `C2 -> h_1`, `C3 -> h_2`.
- Compression inside the timestep is weak. `StepMask` won in two seeds, `S90` in one, and `S80` never won.
- The hybrid is faster in all three seeds and does not lose aggregate internal-carry accuracy on average.
