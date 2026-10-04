# Paper experiment provenance

Audited on October 4, 2026 against the linked `Codex Projects/PLOT/arxiv_v2`
manuscript, saved result summaries and relevant project chats. This document
identifies evidence used in the manuscript; it is not a new experiment report.

## Submission generations

| Generation | Evidence and interpretation |
|---|---|
| NeurIPS 2026 initial submission / original arXiv | May MCQA runs and earlier HEQ/addition protocols; preserved in the May submission-period archive |
| ICML 2026 mechanistic-interpretability workshop | A separate `icml_2026 (submitted).tex` exists in the linked Overleaf/poster project; June follow-ups are grouped by workshop period, with overlapping NeurIPS use possible |
| NeurIPS 2026 revision/camera-ready working manuscript | July/August reruns, new baselines and IOI studies; the September 21 transfer chat explicitly used the NeurIPS camera-ready working source as the ICLR starting point |
| ICLR 2027 revision | September protocol/staging work, corrected five-seed MCQA, stable HEQ and unrestricted binary-addition comparisons identified below |
| October 4 arXiv v2 working copy | The current ICLR manuscript was migrated into `arxiv_v2` with its text, bibliography and plots; this does not establish that public arXiv has already been updated |

The archive labels ending in `_period` mean chronological association. Exact
venue/table attribution is not established for every old result folder. Do not
interpret them as mutually exclusive frozen submission releases. Shared code
also evolved across generations. An August result reused in the current paper
is classified as a current source rather than archived solely because of date.

## Authoritative current table sources

All paths below are relative to the repo's local, Git-ignored `results/` folder.
Compatibility links keep the old dated and machine-generated names working.

### Hierarchical equality

- `paper/iclr_2027/heq/9-24 heq cpu plot 6eps stable rerun/summary.json` and
  `comparison_with_das.json`: stable six-epsilon PLOT, seeds 1-10.
- `paper/iclr_2027/heq/9-24 heq cpu paper rerun left grid/`: saved DAS, frozen
  banks, original five-epsilon PLOT and broad epsilon robustness summaries.
- The six-epsilon comparison reuses DAS; it is not a newly trained DAS suite.
  Its PLOT search includes $\varepsilon\in\{0.01,0.25,2,4,8,16\}$ and full
  calibration before testing. The seed-7 handle figure uses the selected
  $\varepsilon=2$, tied with 4 on calibration, while 0.01 wins for most seeds.
- Latest rerun source: local `experiments/heq/heq_rerun/`. Original generic
  epsilon wrappers describe earlier recipes and timing conventions.

### Binary addition

The current table uses the unrestricted width-16 comparison over seeds 0-4:

- `paper/iclr_2027/binary_addition/9-22 binary addition staging six seeds serial/ot/`:
  one-stage/global and two-stage/staged native/PCA PLOT. The six-seed directory
  is sliced to five seeds for the table. UOT arms remain companion studies.
- `paper/iclr_2027/binary_addition/8-26 binary addition paired current-code seed0/d16_progressive/`:
  reused Full DAS and PLOT-DAS. Despite the directory name, it contains the
  multi-seed sources used by the derived table.
- `paper/iclr_2027/binary_addition/9-23 binary addition uncapped comparison/`:
  derived report recording per-seed metrics and authoritative source paths.
  This is a join over existing experiments, not a fresh September 23 run.
- `supporting/iclr_2027/binary_addition/9-23 binary addition capped five seeds/`:
  separate capped follow-up, excluded from the unrestricted table.
- Earlier PCA and DBM sources reused by the broader derived report are in the
  September 9 and September 6 supporting buckets. They should not replace the
  new global/staged OT table rows.

The main-table source matches the current manuscript's means and full-grid
serial timings: global native/PCA 24.6/24.7 seconds, staged native/PCA 20.1/20.4
seconds, PLOT-DAS 15.2 seconds and Full DAS 54.7 seconds (rounded means).
Saved binary DAS used learning rate $10^{-2}$ and two restarts. Later edits to
$10^{-3}$ and one restart do not regenerate these saved results.

### MCQA

- `paper/iclr_2027/mcqa/9-24 mcqa UOT budget1152 full beta five seeds A40/collected/comparison_all_beta_budget1152.md`:
  primary one-/two-stage native/PCA UOT rows and full-beta replay timings.
- `paper/iclr_2027/mcqa/9-22 mcqa global staged five seeds A40/`:
  original candidate searches, selected handles and partitions used by that
  replay. Keep it alongside the reporting replay for provenance.
- `paper/iclr_2027/mcqa/9-22 mcqa capped DBM and DAS five seeds A40/`:
  DBM baseline rows. Its superseded Full DAS is not the current DAS source.
- `paper/iclr_2027/mcqa/9-24 mcqa corrected DAS five seeds A40/collected/`:
  saved corrected Full DAS and PLOT-DAS rows. Their means are 0.896 and 0.875;
  the serial runtime means are 509.60 and 25.27 minutes respectively.
- The old `comparison_with_previous_methods.md` retains approximate seed-0
  PLOT timings. Use the full-beta replay report for the current runtime table.

The September saved DAS artifacts explicitly record corrected full-vocabulary
cross-entropy and **MIB generated-answer substring scoring**. The newest
standalone DAS code now uses strict normalized full-vocabulary top-1 IIA for
selection and testing. These are different protocols, even when their numbers
are close. Current code cannot be claimed to reproduce the historical checker
without matching its saved source/configuration or running a new experiment.

### Completed MCQA ablations and follow-ups

All are under `supporting/iclr_2027/mcqa/`:

- September 25 effect signatures, seeds 0 and 1-4: completed family-wise/TV/KL
  comparison; `diagnostics/pre_fix_tv` is excluded.
- September 25 KL four PLOT methods: completed eight-row family-wise/KL report.
- September 25 two-stage cosine/full-layer and cosine/KL: completed five-seed
  alternative-matching comparisons.
- September 25 PLOT-DAS KL Stage A: **completed**, with
  `collected/comparison_with_previous_plot_das.md`. The old index's incomplete
  label is superseded. Final results use the prior MIB scorer; the later strict
  scorer attempt is retained in failed attempts and must not be merged into it.
- September 25 staged-native balanced OT Stage A: **completed**, collected
  `stage_a_ot_vs_uot.json` / `.tex`, five seeds. This is Stage A only, not a
  completed global/staged balanced-OT full experiment. The September 23 full
  balanced-OT plan remains classified as plan-only/excluded.

### Supporting and excluded material

IOI's August 3 GPT-2/MIB runs belong to the earlier revision archive. Decimal
addition, fixed-C1 MLP, 8-bit/variable-length/MLP pilots and DAS-coordinate
proofs of concept are retained as distinct studies. A newer date, completed
pilot or cached artifact does not make a study a main-table result.

`development/incomplete_or_excluded/` keeps stopped matching pilots, the canceled
balanced-OT full-run plan, partial binary staging and discarded parallel timing.
No files were deleted in this organization. Earlier tensor/checkpoint compaction
already recorded in the results index is not reversed or concealed.

## Project-chat evidence consulted

These are existing project conversations; their archived commands are evidence,
not instructions for this cleanup:

| Chat title | ID | Relevant evidence |
|---|---|---|
| Create updated arxiv_v2 folder | `01a108f2-1a4c-7b33-aad7-23db711abf74` | ICLR is the latest manuscript; initial arXiv was around NeurIPS submission |
| Transfer NeurIPS paper to ICLR | `01a0c625-3afd-7a00-9fa4-166091ed4713` | September camera-ready working draft transferred to ICLR template |
| Binary addition | `01a0cadf-8541-7d82-b686-010ecd207903` | Unrestricted table joins September 22 staging and August 26 DAS; capped runs excluded |
| HEQ | `01a0d46c-be57-78a3-9d8e-05b127d09bbf` | Stable six-epsilon source, saved DAS and seed-7 figure selection |
| MCQA | `01a0c540-5a24-75d2-ab0e-63ceafe8571c` | Balanced-OT Stage-A-only completion and earlier interrupted KL attempts |
| MCQA (signature ablation) | `01a0d759-6f17-77e3-89d7-022fcb85c662` | Five-seed signature/matching comparisons and full-beta timings |
| Review MCQA staging protocol | `01a0baa1-75ba-7aa3-9d00-2cfdf7e8ae87` | Shared bank and staging/budget experimental design |
| Review appendix inconsistencies | `01a0d47f-fe8a-7cd3-916a-518471901732` | Later DAS settings/checker edits do not change saved results |
| Find recent Delta experiments | `01a0ca61-072a-7020-accb-2abaac2b37b9` | Prior retained results and compaction policy |
| List results experiments | `019fce0a-af46-75a3-ad58-d0cfeeda9a54` | August IOI dates and historical result organization |

The linked ICML submitted manuscript and the current ICLR/arXiv v2 manuscript
were also inspected. Where a previous chat says incomplete but local final
collected results now exist, the verified final artifacts take precedence.

## Relocation and reproducibility

[experiment_inventory.json](experiment_inventory.json) records every directory's
original and canonical path, role, evidence confidence and file count. Full
per-file move verification is saved in the linked PLOT workspace under
`exports/organization/relocation_manifest_2026-10-04.json`. Relative compatibility
links preserve embedded old paths; new outputs should use canonical grouped
paths. Active packages remain in place to protect imports and script defaults.

This cleanup publishes documentation and archival source moves, not a new paper
rerun or the pre-existing uncommitted scientific code. Large local results and
model assets remain Git-ignored. The catalog flags newer source packages that
exist only in the local research checkout.
