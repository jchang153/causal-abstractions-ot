# Anchor-Prefix DAS (`h=8`)

This folder documents the matched DAS comparator for the recurrent 4-bit binary addition benchmark.

## Setting

- backbone: 4-step GRU adder, hidden size `8`
- abstract rows: `C1,C2,C3,C4,S0,S1,S2,S3`
- source policy: `structured_26_top3carry_c2x5_c3x7_no_random`
- fit-bank mode: `anchored_prefix`
- fit / calibration / test bases: `128 / 64 / 64`
- selection rule: `combined`
- DAS sweep:
  - subspace dims `1,2,4`
  - lrs `0.01,0.003`
  - lambda grid `0.5,1,2,4`

## Exact runner

```bash
python scripts/run_h8_anchorprefix_das.py
```

## Best result

Copied source artifact:
- `best_result_full_summary.json`

Compact summary:
- `best_result_compact.json`

Main numbers:
- carry subset combined: `0.9886`
- internal-carry mean (`C1,C2,C3`): `0.9848`
- `C3 = 0.9597`
- measured runtime: `237.9s`

Best row handles:
- `C1 -> h_0[0,1,2,3,4,5,6,7]`
- `C2 -> h_1[0,1,2,3,4,5,6,7]`
- `C3 -> h_2[0,1,2,3,4,5,6,7]`
- `C4 -> h_3[0,1,2,3,4,5,6,7]`

## Why this comparator is included

This is the strongest matched supervised comparator on the repaired `h=8` benchmark.
It provides the cleanest reference point for the remaining gap between OT-based site discovery and a learned subspace intervention primitive.
