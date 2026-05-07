# 4-Bit Binary Addition

This package contains the main-paper recurrent binary-addition benchmark. The model is a GRUCell ripple-carry backbone over 4-bit inputs, and the abstract variables are `C1`, `C2`, and `C3`.

Entry points:

- `run_train_backbone.py`: train or load a factual GRUCell backbone.
- `run_progressive_plot.py`: full progressive PLOT pipeline, including Stage A timestep localization, native/PCA Stage B handles, PLOT-guided DAS, PLOT-PCA-guided DAS, and full DAS.
- `run_progressive_plot_stage_b_resolution_sweep.py`: rerun native Stage B from a cached Stage A result.
- `plot_progressive_heatmaps.py`: render paper heatmaps for PLOT, PLOT-native, PLOT-PCA, PLOT-DAS, and full DAS handles.

Example Stage B rerun:

```bash
python experiments/binary_addition/run_progressive_plot_stage_b_resolution_sweep.py \
  --base-run-dir eval/codex_progressive_plot_10seed \
  --out-dir eval/codex_progressive_plot_resolution_topk \
  --hidden-size 16 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --resolutions 1,2
```

Related addition experiment folders outside the main-paper path:

- `../binary_addition_c1/`: fixed-`C1` MLP binary-addition benchmark.
- `../two_digit_addition/`: two-digit decimal addition experiments and shared helpers.
