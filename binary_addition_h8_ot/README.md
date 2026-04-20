# Binary Addition `h=8` OT Share Bundle

This bundle packages the **new recurrent 4-bit binary addition benchmark** in the narrow `h=8` setting that produced the strongest OT scores.

It contains two OT variants:
- `regular_shared_ot/`: the best **pre-anchoring shared-bank OT** pipeline
- `anchored_ot/`: the best **anchored-bank OT** pipeline

Why `h=8` only:
- this is the strongest OT regime we found for the repaired arithmetic benchmark
- best shared-bank OT carry mean: `0.8181`
- best anchored OT carry mean: `0.8643`

## Layout

- `experiments/binary_addition_rnn/`
  - copied benchmark code needed by the two exported pipelines
- `scripts/run_h8_regular_shared_ot.py`
  - exact wrapper for the best shared-bank OT sweep
- `scripts/run_h8_anchored_ot.py`
  - exact wrapper for the best anchored OT sweep
- `regular_shared_ot/README.md`
  - full description of the shared-bank setup and best result
- `anchored_ot/README.md`
  - full description of the anchored setup and best result
- `regular_shared_ot/best_result_compact.json`
- `regular_shared_ot/best_result_full_summary.json`
- `anchored_ot/best_result_compact.json`
  - compact summaries extracted from the original workspace artifacts

## Minimal dependencies

- Python `>=3.10`
- `torch`

Install:

```bash
pip install torch
```

## Running from the bundle root

Shared-bank OT:

```bash
python scripts/run_h8_regular_shared_ot.py
```

Anchored OT:

```bash
python scripts/run_h8_anchored_ot.py
```

Both scripts will reuse `results/checkpoints/gru_h8_seed0.pt` if it exists, otherwise they train the factual GRU backbone and save it there.

## Benchmark constants used by both variants

- backbone: 4-step `GRUCell` adder
- hidden size: `8`
- factual outputs: Bernoulli bits `S0,S1,S2,S3,C4`
- abstract rows: all endogenous bits `C1,C2,C3,C4,S0,S1,S2,S3`
- candidate sites:
  - grouped hidden states `h_0..h_3`
  - scalar output logits `logit[0..4]`
- source bank: `structured_26_top3carry_c2x5_c3x7_no_random`
- base split: `128 / 64 / 64` for fit / calibration / test bases
- factual exact accuracy: `1.0`

## Structured-26 source bank

Per base example, the shared source family menu is:
- operand-local flips:
  - `flip_A0..flip_A3`
  - `flip_B0..flip_B3`
- carry-targeted families:
  - `target_C1` x `3`
  - `target_C2` x `5`
  - `target_C3` x `7`
  - `target_C4` x `3`
- no random source family

That gives `26` source families per base.

## Notes

- This bundle is intentionally narrow and paper-facing.
- It does **not** include DAS, oracle-site diagnostics, or broader benchmark surgery.
- The anchored raw summary is not copied because the original JSON exceeds GitHub's single-file size limit; the anchored README records the original source artifact path.
- The goal is to make the `h=8` OT setting easy to drop into `causal-abstractions-ot` in a presentable form.
