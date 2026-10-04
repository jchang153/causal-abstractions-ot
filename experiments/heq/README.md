# Hierarchical Equality

## Current paper source

The latest paper comparison uses `heq_rerun/plot_6eps_10seed.py`: the stable six-epsilon PLOT rerun, with the saved DAS runs, over seeds
1-10. The seed-7 handle figure and broad epsilon ablation use the corresponding
rerun artifacts. See [paper provenance](../../docs/experiment_history.md).

The `equality_run.py` and epsilon wrappers described below are earlier protocols;
their selected-epsilon runtime convention is not the newer full-grid rerun timer.
The paper's saved DAS budget was 1000 epochs; later code defaults were changed
without rerunning those saved results.

## Shared implementation and earlier entry points

Main-paper HEQ code lives here. The benchmark learns a small MLP for the task

$$O = \mathbf{1}\!\left[(W=X)=(Y=Z)\right].$$

and evaluates intervention handles for the abstract variables `WX` and `YZ`.

Entry points:

- `equality_run.py`: main OT/UOT and DAS comparison runner.
- `equality_calibration_strategy_sweep.py`: shared/separate calibration-bank sweep.
- `equality_clean_epsilon_sweep.py`: OT/UOT epsilon sweep with the calibration protocol fixed.

Epsilon selection is global across abstract variables and calibration-only.
Each epsilon is scored by the equal-weight average variable calibration score;
the test banks are evaluated only once, after the shared epsilon is frozen.
Because HEQ currently fixes the resolution, reported transport runtime is the
selected-epsilon coupling, calibration, and final test evaluation.
- `equality_paper_figures.py`: regenerates the HEQ paper figures in `$PLOT_PAPER_DIR/plots/` (default: `~/Documents/Codex Projects/PLOT/paper/plots/`).

Implementation package:

- `equality_experiment/`: HEQ-specific SCM, pair banks, backbone, OT/UOT, DAS, reporting, and plotting helpers.

Additional HEQ plotting utilities are in `../heq_intervention_heatmaps/`.
