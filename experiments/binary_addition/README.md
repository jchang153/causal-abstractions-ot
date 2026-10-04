# 4-Bit Binary Addition

## Current paper versus earlier studies

The version v4 main table uses width 16 and seeds 0-4: unrestricted
global/staged OT from `run_staging_comparison.py` / `run_staging_suite.py`,
and DAS/PLOT-DAS reused from the paired run of `run_progressive_plot.py`.
The dimension-capped suite is a separate follow-up. The 8-bit,
variable-length, cross-length and residual-MLP pilots below are not main-table runs.

Saved binary DAS used learning rate $10^{-2}$, two restarts and at most 100 epochs.
Later code defaults were edited to $10^{-3}$ and one restart. These code edits do
not retroactively change the saved table. Runtime conventions also differ between
older selected-epsilon reporting and current full-grid serial summaries.
See [paper provenance](../../docs/experiment_history.md) for exact source paths.

## Preserved implementation and historical recipes

This package contains the main-paper recurrent binary-addition benchmark. The model is a GRUCell ripple-carry backbone over 4-bit inputs, and the abstract variables are `C1`, `C2`, and `C3`.

Entry points:

- `run_train_backbone.py`: train or load a factual GRUCell backbone.
- `run_progressive_plot.py`: full progressive PLOT pipeline, including Stage A timestep localization, native/PCA Stage B handles, PLOT-guided DAS, PLOT-PCA-guided DAS, and full DAS.
- `run_progressive_plot_stage_b_resolution_sweep.py`: rerun native Stage B from a cached Stage A result.
- `plot_progressive_heatmaps.py`: render paper heatmaps for PLOT, PLOT-native, PLOT-PCA, PLOT-DAS, and full DAS handles.
- `run_mib_baselines.py`: run Full State, canonical DBM, and DBM+PCA over all recurrent timesteps, selecting timesteps on calibration data before test reporting.
- `run_local_binary_addition_10seed_suite.sh`: run the complete ten-seed suite serially with the `torch-metal` environment, using CPU by setting `DEVICE=cpu` or Apple Metal with `DEVICE=mps`.
- `run_local_paired_d8_d16_pca_bands_10seed.sh`: rerun only two-stage PLOT-PCA on the existing exact-accuracy checkpoints for hidden widths 8 and 16, using MCQA-style disjoint PCA-band sweeps.
- `run_local_binary_addition_plot4_w8_h64.sh`: run the resume-safe one-seed 8-bit/hidden-size-64 CPU pilot for single- and two-stage PLOT and PLOT-PCA only. It uses internal carries `C1`--`C7`, a 512/256/256 base split, and resolutions 1--64 in powers of two.
- `build_cross_length_counterfactual_dataset.py`: generate and validate the fixed variable-length counterfactual manifest. The checked-in seed-0 manifest is `manifests/variable_length_cross_length_seed0.json`.
- `run_cross_length_plot4.py`: compare single- and two-stage canonical/PCA PLOT on the fixed mixed-length manifest. It evaluates cross-length carries `C1`--`C14`, includes `C15` as an auxiliary same-length row, and caches each resolution independently for safe resume.
- `slurm/run_delta_binary_addition_10seed_suite.sh`: run the complete hidden-size-16, ten-seed paper suite across four allocated GPUs and write one combined `suite_summary.json`. It does not run Boundless DAS.
- `slurm/submit_delta_binary_addition_10seed_suite.sh`: request one four-A40 Delta node and submit the complete suite.

Transport sweeps use one shared epsilon across abstract variables.  For each
epsilon, every variable selects its best resolution on calibration data; the
epsilon score is the equal-weight average of those variable-level best scores.
Only the winning shared epsilon and frozen per-variable resolutions are tested.
Reported runtime includes coupling and calibration at every resolution for the
selected epsilon, plus the final selected test evaluations.

Two-stage PLOT-PCA follows the MCQA band convention.  After Stage A assigns each
carry to one timestep, Stage B partitions that timestep's full PCA basis into
`B` disjoint, nearly equal-width sites.  It sweeps `B=1,2,4,8` for hidden width
8 and additionally `B=16` for hidden width 16.  A separate coupling is computed
for each selected timestep and band count; only carries assigned to that
timestep are calibrated in that coupling.  One epsilon is selected jointly
across carries, and runtime charges every band count at the selected epsilon.
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

The 8-bit pilot trains one factual backbone on all 65,536 input pairs and requires exact
accuracy 1.0 before any intervention method runs. Its width-neutral structured source policy
uses 46 sources per base, producing 23,552 fit pairs and 11,776 calibration/test pairs per
internal carry. Model checkpoints and result directories are local artifacts and are never
tracked by Git.

The variable-length manifest is one pooled dataset rather than a sweep over separate fixed
widths. It balances 832 fit, 416 calibration, and 416 test bases across lengths 4--16 (64,
32, and 32 bases per length). For every cross-length carry `C1`--`C14`, it fixes 4,096 fit
pairs and 2,048 calibration/test pairs, balanced across the eligible ordered length pairs.
`C15` has no cross-length comparison because only length 16 contains it, so it is included
as an auxiliary same-length row with all available distinct directed pairs: 4,032 fit and
992 calibration/test pairs. `C15` participates in the joint transport mapping but is
excluded from shared hyperparameter selection and the primary `C1`--`C14` mean. Example
identities are globally disjoint across fit, calibration, and test, and the manifest hash
fixes the exact pairs used by every method.

`run_local_binary_addition_plot4_w8_mlp8_h64.sh` repeats the four-method pilot with an
eight-hidden-layer, width-64 residual MLP. Its eight hidden activations are treated as the
ordered intervention stages, and the launcher applies the same exact-accuracy gate, data
split, source banks, resolution grid, and runtime definition as the GRU pilot.

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

### Staging comparison without a dimension cap

`run_staging_suite.py` compares global, globally matched single-timestep,
exhaustive-local, and staged native/PCA handles on the paired 4-bit GRU
checkpoints at hidden widths 8 and 16. All arms use disjoint catalogs, balanced
OT, shared-epsilon calibration, and selected-only testing. Dimensions are
reported, never constrained. See [STAGING_COMPARISON.md](STAGING_COMPARISON.md)
for the frozen protocol, differences from historical runs, and pilot/full-sweep
commands. Each method runs in a fresh CPU process with its own timing and caches.

For the UOT version, pass `--transport uot --betas .03,.1,.3,1` to
`run_staging_suite.py`. This replaces every coupling, including Stage A, with
one-sided UOT and selects a shared epsilon/beta pair per stage. Full-sweep
runtime includes every beta and epsilon trial. Save this version in a separate
output root to preserve the balanced-OT baseline.
