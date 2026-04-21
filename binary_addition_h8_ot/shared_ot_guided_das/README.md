# Shared OT-Guided DAS (`h=8`, 3 seeds)

This folder packages the current **shared OT -> support extraction -> restricted DAS** result on the repaired recurrent 4-bit GRU addition benchmark.

This is the main hybrid claim from the current benchmark:
- use **shared OT** only as a cheap support-discovery stage
- extract row-dominant support from the raw OT couplings
- run DAS only inside that support
- compare against a matched full DAS baseline on the same internal-carry rows `C1,C2,C3`

## Benchmark setting

- backbone: 4-step `GRUCell` adder
- hidden size: `8`
- factual outputs: `S0,S1,S2,S3,C4`
- factual exact accuracy: `1.0`
- abstract rows for discovery: all endogenous bits `C1,C2,C3,C4,S0,S1,S2,S3`
- rows for the hybrid comparison: `C1,C2,C3`
- source policy: `structured_26_top3carry_c2x5_c3x7_no_random`
- split: `128 / 64 / 64` fit / calibration / test bases
- candidate neural sites for OT discovery:
  - grouped hidden states `h_0..h_3`
  - scalar output logits `logit[0..4]`

## Shared OT discovery stage

The OT front end is deliberately frozen and small.

Fixed choices:
- output-effect signatures
- all-endogenous abstract rows
- signature normalization on
- fit-signature mode `all`
- fit-family profile `all`
- fit-stratify mode `none`
- cost metric `sq_l2`

Swept OT axes:
- resolutions: `8,4,2,1`
- epsilon: `0.01,0.03`
- `top_k`: `1,2`
- `lambda`: `0.5,1,2`
- calibration selection: `combined:0.0`

## Support extraction

Support is extracted from raw OT couplings using row-dominant evidence pooled across near-best trials.

Procedure:
- keep trials within `98%` of the best row-level calibration score
- cap at `12` retained trials per row
- compute row-dominant conditional mass
- aggregate first to timestep evidence, then to coordinate evidence inside the dominant timestep

Exported masks:
- `StepMask`: full dominant timestep
- `S80`: smallest coordinate subset covering `80%` of pooled evidence
- `S90`: same for `90%`

## Restricted DAS stage

Restricted DAS is run only on masks emitted by the OT support extractor.

Fixed choices:
- fit-bank mode: `anchored_prefix`
- rows: `C1,C2,C3`
- mask modes: `StepMask,S80,S90`
- selection rule: `combined`

Swept DAS axes:
- lambda: `0.25,0.5,1,2,4,8`
- subspace dimension: `1,2,4`
- learning rate: `0.01,0.003`

Mask selection is done by calibration, not by test hindsight.

## Matched full DAS baseline

The comparator is a row-matched full DAS baseline on the same rows `C1,C2,C3`.

Swept axes:
- resolutions: `8,4,2,1`
- lambda: `0.25,0.5,1,2,4,8`
- subspace dimension: `1,2,4`
- learning rate: `0.01,0.003`

## Aggregate 3-seed result

Shared-OT-guided DAS:
- internal-carry mean: `0.9775`
- `C3`: `0.9522`
- end-to-end runtime mean: `84.67s`

Matched full DAS:
- internal-carry mean: `0.9746`
- `C3`: `0.9508`
- end-to-end runtime mean: `271.83s`

Speedup:
- shared-OT-guided DAS is `3.21x` faster end to end on average

## Interpretation

Main takeaways:
- shared OT reliably recovers the correct coarse timestep support
- adding `r=2` and `r=1` does not change the support story for the hard middle carries
- restricted DAS inside OT-discovered support is competitive with matched full DAS
- the runtime advantage is substantial enough to justify the hybrid framing

This supports the current thesis for the GRU addition benchmark:
- OT is better used as a **support-discovery stage** than as the final direct intervention method
- DAS remains the stronger final intervention primitive inside the discovered support

## Files to read

- `README.md`
  - method specification and benchmark setup
- `RUN_LOG.md`
  - seed-by-seed support, accuracy, and runtime log
- `three_seed_summary_compact.json`
  - compact machine-readable summary of the full 3-seed run
- `three_seed_summary_full.json`
  - fuller machine-readable artifact for the complete run

## Re-running

Use:

```bash
python scripts/run_h8_shared_ot_guided_das_multiseed.py
```

The wrapper reuses these checkpoints if present:
- `results/checkpoints/gru_h8_seed0.pt`
- `results/checkpoints/gru_h8_seed1.pt`
- `results/checkpoints/gru_h8_seed2.pt`

If they do not exist, the underlying runners will train and save them.
