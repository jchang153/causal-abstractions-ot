# Best Anchored OT (`h=8`)

This folder documents the strongest **anchored-bank OT** result for the recurrent 4-bit binary addition benchmark.

## Setting

- same backbone and source-bank foundation as the shared-bank run:
  - GRU hidden size `8`
  - abstract rows `C1,C2,C3,C4,S0,S1,S2,S3`
  - grouped hidden-state + output-logit site family
  - source policy `structured_26_top3carry_c2x5_c3x7_no_random`
  - fit / calibration / test bases `128 / 64 / 64`
- method: anchored-bank `OT`
- normalization: on
- fit signature mode: `all`
- fit stratification: `row_counterfactual`
- fit-family profile: `all`
- cost metric: squared `L2`

## What anchoring changes

Instead of one shared fit bank for every abstract row, this runner fits **one OT plan per anchor row**:
1. choose an anchor row `Z_i`
2. build an anchor-filtered fit bank using families most relevant to that row
3. fit a full OT plan over all abstract rows x all neural sites using that anchor bank
4. calibrate and keep only the anchor row result
5. repeat for all rows and aggregate the per-anchor winners

This keeps a full row-vs-site OT plan inside each run, but the fit bank is conditioned on the anchor row.

## Anchor-bank construction used by the best run

The best `h=8` anchored result came from the older anchored-bank sweep, not the later core+focus custom-bank variants.

Anchor family prefixes:
- `C1`:
  - `flip_A0`, `flip_B0`, `target_C1`
- `C2`:
  - `flip_A1`, `flip_B1`, `target_C2`, `target_C1`
- `C3`:
  - `flip_A2`, `flip_B2`, `target_C3`, `target_C2`
- `C4`:
  - `flip_A3`, `flip_B3`, `target_C4`, `target_C3`
- `S0`:
  - `flip_A0`, `flip_B0`
- `S1`:
  - `flip_A1`, `flip_B1`, `target_C1`
- `S2`:
  - `flip_A2`, `flip_B2`, `target_C2`
- `S3`:
  - `flip_A3`, `flip_B3`, `target_C3`

## Sweep that produced the best anchored result

Swept axes inside the anchored runner:
- resolutions: `8,4,2,1`
- OT epsilons: `0.003,0.01,0.03,0.1,0.3`
- `top_k`: `1,2,4,8`
- `lambda`: `0.25,0.5,1,2,4,8`
- selection profiles:
  - `combined:0.0`
  - `sensitivity_only:0.0`

The best `h=8` anchored setting selected:
- fit-family profile: `all`
- fit stratify mode: `row_counterfactual`
- cost metric: `sq_l2`

Exact wrapper for this sweep setting:

```bash
python scripts/run_h8_anchored_ot.py
```

## Best result

Compact best-result summary:
- `best_result_compact.json`

Original full summary artifact in the source workspace:
- `C:\Users\zgzg1\Projects\GW DAS\eval\codex_ot_anchorbanks_h8_sweep\fit-all__strat-row_counterfactual__cost-sq_l2\joint_endogenous_anchor_fitbanks_summary.json`

The raw anchored summary is not copied into this package because that JSON exceeds GitHub's 100 MB per-file limit.

Main numbers:
- carry subset:
  - sensitivity: `0.7451`
  - invariance: `0.9835`
  - combined: `0.8643`
- internal-carry mean (`C1,C2,C3`): `0.8190`

Row-level anchored winners:
- `C1`:
  - resolution `2`, epsilon `0.1`
  - sites `h_0[4,5] + h_0[0,1]`
  - combined `0.8889`
- `C2`:
  - resolution `4`, epsilon `0.1`
  - sites `h_1[4,5,6,7] + h_1[0,1,2,3]`
  - combined `0.8164`
- `C3`:
  - resolution `2`, epsilon `0.1`
  - sites `h_2[4,5] + h_2[0,1]`
  - combined `0.7519`
- `C4`:
  - resolution `8`, epsilon `0.003`
  - site `logit[4]`
  - combined `1.0000`

## Why this is the anchored baseline to share

This is the strongest anchored OT result we obtained on `h=8`.
Later custom core+focus bank variants were explored, but they did **not** beat this older anchored-bank sweep result.
