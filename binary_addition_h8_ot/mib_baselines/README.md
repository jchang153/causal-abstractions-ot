# MIB-Style Baselines (`h=8`)

This folder documents the MIB-style comparator methods on the same repaired `h=8` recurrent addition benchmark.

## Included methods

- `full_vector`
- `dbm`
- `dbm_pca`
- `dbm_sae`

## Setting

- backbone: 4-step GRU adder, hidden size `8`
- abstract rows: `C1,C2,C3,C4,S0,S1,S2,S3`
- source policy: `structured_26_top3carry_c2x5_c3x7_no_random`
- fit / calibration / test bases: `128 / 64 / 64`
- selection rule: `combined`
- candidate sites:
  - full hidden states `h_0..h_3`
  - scalar output logits `logit[0..4]`

## Exact runner

```bash
python scripts/run_h8_mib_baselines.py
```

## Summary files

- `best_result_full_summary.json`
- `best_result_compact.json`

## Best method summaries

`full_vector`
- carry subset combined: `0.8181`
- internal-carry mean: `0.7575`
- `C3 = 0.7287`
- runtime: `12.5s`

`dbm`
- carry subset combined: `0.8605`
- internal-carry mean: `0.8105`
- `C3 = 0.8003`
- runtime: `41.3s`

`dbm_pca`
- carry subset combined: `0.9099`
- internal-carry mean: `0.8798`
- `C3 = 0.8434`
- runtime: `42.6s`

`dbm_sae`
- carry subset combined: `0.8473`
- internal-carry mean: `0.7964`
- `C3 = 0.7429`
- runtime: `48.8s`

## Notes

These are included because they provide the strongest non-DAS comparator family we evaluated alongside OT on this benchmark.
Among them, `DBM+PCA` is the strongest `h=8` baseline.
