# Best Shared-Bank OT (`h=8`)

This folder documents the strongest **pre-anchoring** OT pipeline for the recurrent 4-bit binary addition benchmark.

## Setting

- backbone: 4-step GRU adder
- hidden size: `8`
- factual outputs: `S0,S1,S2,S3,C4`
- abstract rows: `C1,C2,C3,C4,S0,S1,S2,S3`
- site family:
  - grouped hidden states at resolutions `8,4,2,1`
  - scalar output logits `logit[0..4]`
- method: `OT`
- source policy: `structured_26_top3carry_c2x5_c3x7_no_random`
- fit / calibration / test bases: `128 / 64 / 64`
- signature normalization: on
- fit signature mode: `all`
- fit stratification: none
- fit-family profile: `all`
- cost metric: squared `L2`

## Fit/calibration bank construction

### Factual base split
- fit bases: `128`
- calibration bases: `64`
- test bases: `64`

### Source families per base
The shared fit bank uses the same `26` structured source families for every abstract row:
- operand-local bit flips:
  - `flip_A0..flip_A3`
  - `flip_B0..flip_B3`
- carry-targeted sources:
  - `target_C1` x `3`
  - `target_C2` x `5`
  - `target_C3` x `7`
  - `target_C4` x `3`

There is **no random source family** in this final best shared-bank run.

### Abstract and neural effect signatures
This stays inside the original output-effect methodology:
- abstract side: final 5-bit output probability delta under abstract intervention
- neural side: final 5-bit output probability delta after site swap

The deterministic SCM yields degenerate `0/1` probability vectors on the abstract side; the GRU yields Bernoulli probabilities on the neural side.

### Aggregation
- signatures are averaged within each source family first
- family means are concatenated into one row/site signature
- concatenated signatures are normalized before cost computation

## Sweep that produced the best result

Swept axes:
- resolutions: `8,4,2,1`
- OT epsilons: `0.003,0.01,0.03,0.1,0.3`
- `top_k`: `1,2,4,8`
- `lambda`: `0.25,0.5,1,2,4,8`
- selection profiles:
  - `combined:0.0`
  - `sensitivity_only:0.0`

Exact wrapper:

```bash
python scripts/run_h8_regular_shared_ot.py
```

## Best result

Source artifact copied here:
- `best_result_full_summary.json`

Compact best-result summary:
- `best_result_compact.json`

Main numbers:
- carry subset:
  - sensitivity: `0.7353`
  - invariance: `0.9010`
  - combined: `0.8181`
- internal-carry mean (`C1,C2,C3`): `0.7575`

Row-level best handles:
- `C1 -> h_0[0,1,2,3,4,5,6,7]`
- `C2 -> h_1[0,1,2,3,4,5,6,7]`
- `C3 -> h_2[0,1,2,3,4,5,6,7]`
- `C4 -> logit[4]`

Row-level combined scores:
- `C1 = 0.7693`
- `C2 = 0.7745`
- `C3 = 0.7287`
- `C4 = 1.0000`

## Why this is the pre-anchoring baseline

This is the strongest OT result obtained **before** moving to anchored-bank fitting:
- one shared fit bank for all rows
- one shared transport problem over all rows and all candidate sites
- no row-specific anchored bank construction
