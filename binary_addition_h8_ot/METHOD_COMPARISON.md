# Shared vs Anchored OT (`h=8`)

This note explains the difference between the two exported OT pipelines in this bundle.

## Shared-bank OT

Path:
- `regular_shared_ot/`

Definition:
- one shared fit bank for all abstract rows
- one shared OT problem over all rows and all candidate sites
- no row-conditioned fit-bank construction

Best exported result:
- carry subset combined: `0.8181`
- internal-carry mean (`C1,C2,C3`): `0.7575`

Best row placements:
- `C1 -> h_0`
- `C2 -> h_1`
- `C3 -> h_2`
- `C4 -> logit[4]`

Interpretation:
- this is the strongest OT pipeline before introducing anchor-conditioned fit banks
- it is the cleanest baseline for the original shared-bank transport story

## Anchored-bank OT

Path:
- `anchored_ot/`

Definition:
- for each abstract variable `Z_i`, build an anchor-specific fit bank
- fit a full OT plan over all rows and all sites using that anchor-conditioned bank
- keep only the anchor row after calibration
- repeat for all rows and aggregate the per-anchor winners

Best exported result:
- carry subset combined: `0.8643`
- internal-carry mean (`C1,C2,C3`): `0.8190`

Best row placements:
- `C1 -> h_0[4,5] + h_0[0,1]`
- `C2 -> h_1[4,5,6,7] + h_1[0,1,2,3]`
- `C3 -> h_2[4,5] + h_2[0,1]`
- `C4 -> logit[4]`

Interpretation:
- this is the strongest `h=8` OT result we obtained overall
- it keeps the OT machinery and output-effect signatures intact
- the gain comes from reducing fit-bank blur, especially for the middle carries

## What actually changes between the two

Held fixed:
- same GRU backbone (`hidden_size=8`)
- same abstract variable set `C1,C2,C3,C4,S0,S1,S2,S3`
- same mixed site family (grouped hidden states + output logits)
- same structured source policy `structured_26_top3carry_c2x5_c3x7_no_random`
- same output-effect feature space
- same OT calibration style (`top_k`, `lambda` sweep on held-out calibration bases)

Changed:
- fit-bank construction
- fit stratification
- per-row selection emerges from anchor-conditioned transport rather than one shared-bank fit

## Why both should be pushed

The pair is scientifically useful:
- `regular_shared_ot/` shows the strongest non-anchored OT baseline
- `anchored_ot/` shows the strongest OT variant after benchmark repair

Together they support the paper narrative:
1. structured banks repair a benchmark-design problem
2. anchored fitting gives OT another real gain
3. the remaining gap to DAS is then easier to interpret as primitive-related rather than fit-bank-related

## Hybrid Follow-on

For the transport-guided DAS follow-on built on top of the shared OT discoverer, see:
- `shared_ot_guided_das/README.md`
- `shared_ot_guided_das/RUN_LOG.md`
