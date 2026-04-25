# Binary Addition RNN Benchmark

Clean benchmark package for 4-bit binary addition with a recurrent ripple-carry backbone.

Current first-pass components:

- `scm.py`: exact abstract 4-bit ripple-carry model and carry interventions
- `data.py`: exhaustive factual examples and exhaustive source-pair bank construction
- `model.py`: GRUCell factual backbone with 5-bit output `(S0,S1,S2,S3,C4)`
- `run_train_backbone.py`: factual-training entry point

The intended next steps are OT/UOT/DAS runners built on full hidden-state sites at each timestep.
