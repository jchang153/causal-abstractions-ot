# Experiment version provenance

The current manuscript, saved summaries and project conversations identify the
sources below. This is a source inventory, not a new experiment report.

## Experiment versions

| Version | Preserved evidence |
|---|---|
| v1 | Initial MCQA runs, early addition tasks and intervention notebooks |
| v2 | Follow-up comparison studies, metric exploration and broader MCQA searches |
| v3 | Expanded baselines, IOI and addition extension pilots |
| v4 | Current staging comparisons, corrected MCQA baselines, stable HEQ and supporting ablations |

Versions record research generations rather than complete frozen releases.
An earlier run reused by the current tables appears among the v4 paper sources.
Saved configurations determine the exact protocol; later code edits do not
retroactively change saved results.

## Authoritative current table sources

All paths below are relative to the repo's local, Git-ignored `results/` folder.
Run references use the versioned canonical paths.

### Hierarchical equality

- `versions/v4/paper/heq/heq_cpu_plot_6eps_stable_rerun/summary.json` and
  `comparison_with_das.json`: stable six-epsilon PLOT, seeds 1-10.
- `versions/v4/paper/heq/heq_cpu_paper_rerun_left_grid/`: saved DAS, frozen
  banks, original five-epsilon PLOT and broad epsilon robustness summaries.
- The six-epsilon comparison reuses DAS; it is not a newly trained DAS suite.
  Its PLOT search includes $\varepsilon\in\{0.01,0.25,2,4,8,16\}$ and full
  calibration before testing. The seed-7 handle figure uses the selected
  $\varepsilon=2$, tied with 4 on calibration, while 0.01 wins for most seeds.
- Latest rerun source: local `experiments/heq/heq_rerun/`. Original generic
  epsilon wrappers describe earlier recipes and timing conventions.

### Binary addition

The current table uses the unrestricted width-16 comparison over seeds 0-4:

- `versions/v4/paper/binary_addition/binary_addition_staging_six_seeds_serial/ot/`:
  one-stage/global and two-stage/staged native/PCA PLOT. The six-seed directory
  is sliced to five seeds for the table. UOT arms remain companion studies.
- `versions/v4/paper/binary_addition/binary_addition_paired_current_code_seed0/d16_progressive/`:
  reused Full DAS and PLOT-DAS. Despite the directory name, it contains the
  multi-seed sources used by the derived table.
- `versions/v4/paper/binary_addition/binary_addition_uncapped_comparison/`:
  derived report recording per-seed metrics and authoritative source paths.
  This is a join over existing experiments, not a fresh run.
- `versions/v4/supporting/binary_addition/binary_addition_capped_five_seeds/`:
  separate capped follow-up, excluded from the unrestricted table.
- Earlier PCA and DBM sources reused by the broader derived report are in the
  and supporting buckets. They should not replace the
  new global/staged OT table rows.

The main-table source matches the current manuscript's means and full-grid
serial timings: global native/PCA 24.6/24.7 seconds, staged native/PCA 20.1/20.4
seconds, PLOT-DAS 15.2 seconds and Full DAS 54.7 seconds (rounded means).
Saved binary DAS used learning rate $10^{-2}$ and two restarts. Later edits to
$10^{-3}$ and one restart do not regenerate these saved results.

### MCQA

- `versions/v4/paper/mcqa/mcqa_uot_budget1152_full_beta_five_seeds_a40/collected/comparison_all_beta_budget1152.md`:
  primary one-/two-stage native/PCA UOT rows and full-beta replay timings.
- `versions/v4/paper/mcqa/mcqa_global_staged_five_seeds_a40/`:
  original candidate searches, selected handles and partitions used by that
  replay. Keep it alongside the reporting replay for provenance.
- `versions/v4/paper/mcqa/mcqa_capped_dbm_and_das_five_seeds_a40/`:
  DBM baseline rows. Its superseded Full DAS is not the current DAS source.
- `versions/v4/paper/mcqa/mcqa_corrected_das_five_seeds_a40/collected/`:
  saved corrected Full DAS and PLOT-DAS rows. Their means are 0.896 and 0.875;
  the serial runtime means are 509.60 and 25.27 minutes respectively.
- The old `comparison_with_previous_methods.md` retains approximate seed-0
  PLOT timings. Use the full-beta replay report for the current runtime table.

The saved DAS artifacts explicitly record corrected full-vocabulary
cross-entropy and **MIB generated-answer substring scoring**. The newest
standalone DAS code now uses strict normalized full-vocabulary top-1 IIA for
selection and testing. These are different protocols, even when their numbers
are close. Current code cannot be claimed to reproduce the historical checker
without matching its saved source/configuration or running a new experiment.

### Completed MCQA ablations and follow-ups

All are under `versions/v4/supporting/mcqa/`:

- effect signatures, seeds 0 and 1-4: completed family-wise/TV/KL
  comparison; `diagnostics/pre_fix_tv` is excluded.
- KL four PLOT methods: completed eight-row family-wise/KL report.
- two-stage cosine/full-layer and cosine/KL: completed five-seed
  alternative-matching comparisons.
- PLOT-DAS KL Stage A: **completed**, with
  `collected/comparison_with_previous_plot_das.md`. The old index's incomplete
  label is superseded. Final results use the prior MIB scorer; the later strict
  scorer attempt is retained in failed attempts and must not be merged into it.
- staged-native balanced OT Stage A: **completed**, collected
  `stage_a_ot_vs_uot.json` / `.tex`, five seeds. This is Stage A only, not a
  completed global/staged balanced-OT full experiment. The full
  balanced-OT plan remains classified as plan-only/excluded.

### Supporting and excluded material

IOI's GPT-2/MIB runs belong to the earlier revision archive. Decimal
addition, fixed-C1 MLP, 8-bit/variable-length/MLP pilots and DAS-coordinate
proofs of concept are retained as distinct studies. A completed
pilot or cached artifact does not make a study a main-table result.

`development/incomplete_or_excluded/` keeps stopped matching pilots, the canceled
balanced-OT full-run plan, partial binary staging and discarded parallel timing.
No files were deleted in this organization. Earlier tensor/checkpoint compaction
already recorded in the results index is not reversed or concealed.

## Source availability and relocation

[experiment_inventory.json](experiment_inventory.json) records each canonical
folder, version, role and artifact count. Full legacy mappings and move audits
are retained in the linked workspace outside this repo. No result files were
deleted by this organization. Original execution metadata remains in saved
result payloads; it is not used to name or label the published inventory.

Current packages remain in place. Historical implementation folders have import
compatibility links. Results and model assets remain local and Git-ignored.
Some newer paper execution packages and scientific edits remain uncommitted;
the experiment catalog identifies their availability.
