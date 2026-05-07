# Shared OT-Guided DAS (`h=16`, 3 seeds)

This folder packages the `h=16` scaling follow-up for the **shared OT -> support extraction -> restricted DAS** pipeline on the repaired recurrent 4-bit GRU addition benchmark.

The role of this run is different from the `h=8` result:
- `h=8` is the main proof-of-concept benchmark
- `h=16` is the scaling follow-up asking whether the runtime gap in favor of OT-guided DAS gets larger while accuracy stays close to full DAS

## Benchmark setting

- backbone: 4-step `GRUCell` adder
- hidden size: `16`
- factual outputs: `S0,S1,S2,S3,C4`
- factual exact accuracy: `1.0`
- abstract rows for discovery: all endogenous bits `C1,C2,C3,C4,S0,S1,S2,S3`
- rows for the hybrid comparison: `C1,C2,C3`
- source policy: `structured_26_top3carry_c2x5_c3x7_no_random`
- split: `128 / 64 / 64` fit / calibration / test bases

## Shared OT discovery stage

Fixed choices:
- output-effect signatures
- all-endogenous abstract rows
- signature normalization on
- fit-signature mode `all`
- fit-family profile `all`
- fit-stratify mode `none`
- cost metric `sq_l2`

Swept OT axes:
- resolutions: `16,8,4,2,1`
- epsilon: `0.01,0.03`
- `top_k`: `1,2`
- `lambda`: `0.5,1,2`
- calibration selection: `combined:0.0`

## Support extraction

Same extraction rule as the `h=8` hybrid:
- keep trials within `98%` of the best row-level calibration score
- cap at `12` retained trials per row
- compute row-dominant conditional mass
- pool first to timestep evidence, then to coordinate evidence inside the dominant timestep

Exported masks:
- `StepMask`
- `S80`
- `S90`

## Restricted DAS stage

Fixed choices:
- fit-bank mode: `anchored_prefix`
- rows: `C1,C2,C3`
- mask modes: `StepMask,S80,S90`
- selection rule: `combined`

Swept DAS axes:
- lambda: `0.25,0.5,1,2,4,8`
- subspace dimension: `1,2,4`
- learning rate: `0.01,0.003`

## Matched full DAS baseline

Rows:
- `C1,C2,C3`

Swept axes:
- resolutions: `16,8,4,2,1`
- lambda: `0.25,0.5,1,2,4,8`
- subspace dimension: `1,2,4`
- learning rate: `0.01,0.003`

## Aggregate 3-seed result

Shared-OT-guided DAS:
- internal-carry mean: `0.9795`
- `C3`: `0.9565`
- end-to-end runtime mean: `174.85s`

Matched full DAS:
- internal-carry mean: `0.9826`
- `C3`: `0.9614`
- end-to-end runtime mean: `1080.53s`

Speedup:
- shared-OT-guided DAS is `6.18x` faster end to end on average

## Interpretation

This is the scaling result we wanted:
- accuracy remains close to matched full DAS
- the runtime advantage becomes much larger than at `h=8`
- the hybrid stays a coarse timestep-support story rather than a sparse coordinate-mask story

The main comparison to the `h=8` proof-of-concept is:
- `h=8`: mean speedup `3.21x`
- `h=16`: mean speedup `6.18x`

## Files to read

- `README.md`
- `RUN_LOG.md`
- `three_seed_summary_compact.json`
- `three_seed_summary_full.json`

## Re-running

Use:

```bash
python scripts/run_h16_shared_ot_guided_das_multiseed.py
```

The wrapper reuses these checkpoints if present:
- `results/checkpoints/gru_h16_seed0.pt`
- `results/checkpoints/gru_h16_seed1.pt`
- `results/checkpoints/gru_h16_seed2.pt`
