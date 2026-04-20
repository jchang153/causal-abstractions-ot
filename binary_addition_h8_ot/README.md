# Binary Addition `h=8` OT Share Bundle

This bundle packages the **new recurrent 4-bit binary addition benchmark** in the narrow `h=8` setting that produced the strongest OT scores.

It contains four method tracks on the same `h=8` recurrent addition benchmark:
- `regular_shared_ot/`: the best **pre-anchoring shared-bank OT** pipeline
- `anchored_ot/`: the best **anchored-bank OT** pipeline
- `das/`: the matched **anchor-prefix DAS** comparator
- `mib_baselines/`: `Full Vector`, `DBM`, `DBM+PCA`, and `DBM+SAE`

Why `h=8` only:
- this is the strongest OT regime we found for the repaired arithmetic benchmark
- best shared-bank OT carry mean: `0.8181`
- best anchored OT carry mean: `0.8643`

## Layout

- `experiments/binary_addition_rnn/`
  - copied benchmark code needed by the exported method pipelines
- `scripts/run_h8_regular_shared_ot.py`
  - exact wrapper for the best shared-bank OT sweep
- `scripts/run_h8_anchored_ot.py`
  - exact wrapper for the best anchored OT sweep
- `scripts/run_h8_anchorprefix_das.py`
  - exact wrapper for the matched anchor-prefix DAS run
- `scripts/run_h8_mib_baselines.py`
  - exact wrapper for the MIB-style baseline suite
- `regular_shared_ot/README.md`
  - full description of the shared-bank setup and best result
- `anchored_ot/README.md`
  - full description of the anchored setup and best result
- `das/README.md`
  - matched DAS setup and best result
- `mib_baselines/README.md`
  - MIB-style baseline setup and best result summary
- `METHOD_COMPARISON.md`
  - shared-bank OT vs anchored OT
- `BASELINE_COMPARISON.md`
  - OT, DAS, and MIB-style comparator summary
- compact/full result summaries under each method folder

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

Anchor-prefix DAS:

```bash
python scripts/run_h8_anchorprefix_das.py
```

MIB-style baselines:

```bash
python scripts/run_h8_mib_baselines.py
```

All wrapper scripts will reuse `results/checkpoints/gru_h8_seed0.pt` if it exists, otherwise they train the factual GRU backbone and save it there.

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
- It includes DAS and the MIB-style baselines, but it does **not** include oracle-site diagnostics or the broader benchmark-development history.
- The anchored raw summary is not copied because the original JSON exceeds GitHub's single-file size limit; the anchored README records the original source artifact path.
- The goal is to make the `h=8` comparison setting easy to drop into `causal-abstractions-ot` in a presentable form.
