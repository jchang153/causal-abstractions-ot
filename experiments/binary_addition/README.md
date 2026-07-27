# 4-Bit Binary Addition

This package contains the main-paper recurrent binary-addition benchmark. The model is a GRUCell ripple-carry backbone over 4-bit inputs, and the abstract variables are `C1`, `C2`, and `C3`.

Entry points:

- `run_train_backbone.py`: train or load a factual GRUCell backbone.
- `run_progressive_plot.py`: full progressive PLOT pipeline, including Stage A timestep localization, native/PCA Stage B handles, PLOT-guided DAS, PLOT-PCA-guided DAS, and full DAS.
- `run_progressive_plot_stage_b_resolution_sweep.py`: rerun native Stage B from a cached Stage A result.
- `plot_progressive_heatmaps.py`: render paper heatmaps for PLOT, PLOT-native, PLOT-PCA, PLOT-DAS, and full DAS handles.
- `run_mib_baselines.py`: run Full State, canonical DBM, and DBM+PCA over all recurrent timesteps, selecting timesteps on calibration data before test reporting.

The Delta launcher `slurm/run_delta_binary_addition_mib_baselines.sh` runs the three MIB-style
baselines for seeds 0--2 and carries `C1`--`C3` in one resume-safe `srun`. It uses the main-paper
128/64/64 base split and the structured source policy used by the progressive PLOT experiment.

Example Stage B rerun:

```bash
python experiments/binary_addition/run_progressive_plot_stage_b_resolution_sweep.py \
  --base-run-dir eval/progressive_plot_10seed \
  --out-dir eval/progressive_plot_resolution_topk \
  --hidden-size 16 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --resolutions 1,2
```

Related addition experiment folders outside the main-paper path:

- `../binary_addition_c1/`: fixed-`C1` MLP binary-addition benchmark.
- `../two_digit_addition/`: two-digit decimal addition experiments and shared helpers.
