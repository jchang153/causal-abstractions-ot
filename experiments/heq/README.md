# Hierarchical Equality

Main-paper HEQ code lives here. The benchmark learns a small MLP for the task

```text
O = int((W == X) == (Y == Z))
```

and evaluates intervention handles for the abstract variables `WX` and `YZ`.

Entry points:

- `equality_run.py`: main OT/UOT and DAS comparison runner.
- `equality_calibration_strategy_sweep.py`: shared/separate calibration-bank sweep.
- `equality_clean_epsilon_sweep.py`: OT/UOT epsilon sweep with the calibration protocol fixed.
- `equality_paper_figures.py`: regenerates the HEQ paper figures in `paper/plots/`.

Implementation package:

- `equality_experiment/`: HEQ-specific SCM, pair banks, backbone, OT/UOT, DAS, reporting, and plotting helpers.

Additional HEQ plotting utilities are in `../heq_intervention_heatmaps/`.
