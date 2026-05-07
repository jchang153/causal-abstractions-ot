# Binary Addition RNN Benchmark

Clean benchmark package for 4-bit binary addition with a recurrent ripple-carry backbone.

Current components:

- `scm.py`: exact abstract 4-bit ripple-carry model and carry interventions
- `data.py`: exhaustive factual examples and exhaustive source-pair bank construction
- `model.py`: GRUCell factual backbone with 5-bit output `(S0,S1,S2,S3,C4)`
- `run_train_backbone.py`: factual-training entry point
- `run_progressive_plot.py`: full progressive PLOT pipeline: Stage A timestep localization, Stage B canonical/PCA PLOT, PLOT-guided DAS, PLOT-PCA-guided DAS, and full DAS
- `run_progressive_plot_stage_b_resolution_sweep.py`: Stage-B-only canonical PLOT rerun that reuses cached Stage A localization and sweeps resolutions such as `r=1,2` with resolution-specific top-`K` grids
- `plot_progressive_heatmaps.py`: paper heatmap generator for Stage A, PLOT, PLOT-PCA, PLOT-DAS, and full DAS handles

Example corrected two-stage PLOT rerun after a cached 10-seed run:

```powershell
python experiments/binary_addition_rnn/run_progressive_plot_stage_b_resolution_sweep.py `
  --base-run-dir eval/codex_progressive_plot_10seed `
  --out-dir eval/codex_progressive_plot_resolution_topk `
  --hidden-size 16 `
  --seeds 0,1,2,3,4,5,6,7,8,9 `
  --resolutions 1,2
```

This runs only Stage B, using `k=1..d` for `r=1` and `k=1..d/2` for `r=2`, then selects each carry handle by calibration.
