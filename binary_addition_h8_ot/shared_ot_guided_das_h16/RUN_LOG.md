# Run Log

This log records the exported 3-seed `h=16` shared-OT-guided DAS scaling run.

## Seed-level summary

| Seed | Best mask | Guided internal-carry | Guided C3 | Full DAS best r | Full DAS internal-carry | Full DAS C3 | Hybrid runtime (s) | Full DAS runtime (s) | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | StepMask | 0.9815 | 0.9586 | 16 | 0.9890 | 0.9722 | 169.59 | 1085.89 | 6.40x |
| 1 | StepMask | 0.9862 | 0.9697 | 16 | 0.9861 | 0.9694 | 175.18 | 1087.84 | 6.21x |
| 2 | StepMask | 0.9709 | 0.9414 | 16 | 0.9726 | 0.9426 | 179.79 | 1067.87 | 5.94x |

## Aggregate

- guided internal-carry mean: `0.9795`
- guided `C3`: `0.9565`
- full DAS internal-carry mean: `0.9826`
- full DAS `C3`: `0.9614`
- mean hybrid runtime: `174.85s`
- mean full DAS runtime: `1080.53s`
- mean speedup: `6.18x`

## Shared OT discovery details

### Seed 0

- `C1`: best OT row fit at r=8, epsilon=0.03, top_k=2, lambda=2.0, sites h_0[8..15] + h_0[0..7]; dominant timestep `h_0` with stability `1.0`
- `C2`: best OT row fit at r=16, epsilon=0.01, top_k=1, lambda=1.0, site h_1[0..15]; dominant timestep `h_1` with stability `0.6666666666666666`
- `C3`: best OT row fit at r=16, epsilon=0.01, top_k=1, lambda=1.0, site h_2[0..15]; dominant timestep `h_2` with stability `1.0`
- best guided mask mode: `StepMask`
- hybrid runtime: `169.59s`; full DAS runtime: `1085.89s`

### Seed 1

- `C1`: best OT row fit at r=16, epsilon=0.03, top_k=1, lambda=1.0, site h_0[0..15]; dominant timestep `h_0` with stability `1.0`
- `C2`: best OT row fit at r=16, epsilon=0.01, top_k=1, lambda=1.0, site h_1[0..15]; dominant timestep `h_1` with stability `1.0`
- `C3`: best OT row fit at r=16, epsilon=0.01, top_k=1, lambda=1.0, site h_2[0..15]; dominant timestep `h_2` with stability `0.6666666666666666`
- best guided mask mode: `StepMask`
- hybrid runtime: `175.18s`; full DAS runtime: `1087.84s`

### Seed 2

- `C1`: best OT row fit at r=4, epsilon=0.01, top_k=1, lambda=2.0, site h_0[4,5,6,7]; dominant timestep `h_0` with stability `1.0`
- `C2`: best OT row fit at r=16, epsilon=0.01, top_k=1, lambda=1.0, site h_1[0..15]; dominant timestep `h_1` with stability `1.0`
- `C3`: best OT row fit at r=16, epsilon=0.01, top_k=1, lambda=1.0, site h_2[0..15]; dominant timestep `h_2` with stability `1.0`
- best guided mask mode: `StepMask`
- hybrid runtime: `179.79s`; full DAS runtime: `1067.87s`

## Observations

- `C2` and `C3` keep choosing coarse full-state discovery handles at `r=16` across all three seeds.
- The pooled support still collapses to the same timestep story `C1 -> h_0`, `C2 -> h_1`, `C3 -> h_2`.
- `StepMask` wins in all three seeds at `h=16`, which is even cleaner than the `h=8` run.
- The runtime gap widens substantially in favor of the hybrid while accuracy remains close to full DAS.
