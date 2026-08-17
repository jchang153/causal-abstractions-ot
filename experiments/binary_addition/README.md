# 4-Bit Binary Addition

This package contains the main-paper recurrent binary-addition benchmark. The model is a GRUCell ripple-carry backbone over 4-bit inputs, and the abstract variables are `C1`, `C2`, and `C3`.

Entry points:

- `run_train_backbone.py`: train or load a factual GRUCell backbone.
- `run_progressive_plot.py`: full progressive PLOT pipeline, including Stage A timestep localization, native/PCA Stage B handles, PLOT-guided DAS, PLOT-PCA-guided DAS, and full DAS.
- `run_progressive_plot_stage_b_resolution_sweep.py`: rerun native Stage B from a cached Stage A result.
- `plot_progressive_heatmaps.py`: render paper heatmaps for PLOT, PLOT-native, PLOT-PCA, PLOT-DAS, and full DAS handles.
- `run_mib_baselines.py`: run Full State, canonical DBM, and DBM+PCA over all recurrent timesteps, selecting timesteps on calibration data before test reporting.
- `slurm/run_delta_binary_addition_10seed_suite.sh`: run the complete hidden-size-16, ten-seed paper suite across four allocated GPUs and write one combined `suite_summary.json`. It does not run Boundless DAS.
- `slurm/submit_delta_binary_addition_10seed_suite.sh`: request one four-A40 Delta node and submit the complete suite.

Transport sweeps use one shared epsilon across abstract variables.  For each
epsilon, every variable selects its best resolution on calibration data; the
epsilon score is the equal-weight average of those variable-level best scores.
Only the winning shared epsilon and frozen per-variable resolutions are tested.
Reported runtime includes coupling and calibration at every resolution for the
selected epsilon, plus the final selected test evaluations.
- `run_boundless_das.py`: resume-safe Boundless DAS sweep over structured C1--C3 banks, recurrent timesteps, and model/data seeds, with aggregate table output.
- `run_boundless_das_diagnostics.py`: resume-safe one-seed sweep over BDAS boundary, temperature, optimizer, loss, selection, and fit-bank variants, with a ranked Markdown summary.

Boundless DAS is implemented in `bdas.py`. Its `BDASConfig` uses the defaults from the
authors' released implementation: Adam, rotation learning rate `1e-3`, boundary learning
rate `1e-2`, initial boundary `0.5`, boundary penalty `1.0`, temperature annealing from
`50.0` to `0.1`, three epochs, batch size 16, gradient accumulation 4, and 10% linear
warmup. `run_bdas_sweep` and `run_bdas_rows` intentionally require candidate timesteps
and targets to be supplied explicitly. BDAS interchange strength is fixed to `1.0` by
construction.

Binary-addition DAS now follows the MCQA optimization protocol where applicable: one
fixed learning rate (`0.01`, selected from the completed seed-0/ten-seed calibration
history), two reproducible random restarts, batch size 64, full shared fit banks, and
loss-plateau stopping with 5 minimum epochs, 100 maximum epochs, patience 1, and relative
improvement threshold `1e-3`. DAS intervention strength is fixed to `1.0`; transport
lambda grids are not reused for DAS calibration.

The Delta launcher `slurm/run_delta_binary_addition_10seed_suite.sh` assigns seeds across four
GPU workers. It sweeps canonical PLOT resolutions and PCA-prefix sizes 1, 2, 4, 8, and 16 for
both single- and two-stage variants, alongside PLOT-DAS and Full DAS
branches; canonical/PCA DBM and Full Vector; and the native cosine and brute-force variants.
Optional support-guided DAS variants and Boundless DAS are skipped. All methods use the same
128/64/64 base split, corresponding to 3,328 fit pairs, 1,664 calibration pairs, and 1,664 test
pairs per abstract variable. Canonical and PCA DBM sweep the summed-mask regularization
coefficient over 0, 1e-5, 1e-4, and 1e-3. Calibration jointly selects the coefficient and
timestep for each abstract variable, with mask size as a tie-breaker, and reported DBM runtime
includes every coefficient/timestep candidate.

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
