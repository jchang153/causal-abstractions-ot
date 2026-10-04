# Experiment results index

## Current organization (2026-10-04)

All 77 physical run directories are grouped below. Original paths remain relative
compatibility symlinks, so saved references and historical commands resolve.
No files were deleted during this reorganization. See
[paper provenance](docs/experiment_history.md) for exact table sources and
[the machine-readable map](docs/experiment_inventory.json) for every relocation.

| Group under `results/` | Purpose | Count |
|---|---|---:|
| `paper/` | Current ICLR 2027 / arXiv v2 table sources | 9 |
| `supporting/` | Completed ablations and companion evidence | 13 |
| `archive/` | Earlier NeurIPS submission, ICML workshop and revision periods | 30 |
| `development/` | Preliminary, pilot, plan-only or excluded results | 25 |

## Complete canonical run map

Historical-period labels are chronological associations unless the confidence
field positively identifies manuscript use. They are not exclusive venue labels.

| Original folder | Canonical folder under `results/` | Role |
|---|---|---|
| `5-3 mcqa full ot uot layerwise` | `archive/neurips_2026_submission_period/mcqa/5-3 mcqa full ot uot layerwise` | May 2026 run; original submission-era evidence. Exact table-level venue use is not established for every folder |
| `5-5 mcqa full das 5 seeds` | `archive/neurips_2026_submission_period/mcqa/5-5 mcqa full das 5 seeds` | May 2026 run; original submission-era evidence. Exact table-level venue use is not established for every folder |
| `5-5 mcqa pca partition band sweep` | `archive/neurips_2026_submission_period/mcqa/5-5 mcqa pca partition band sweep` | May 2026 run; original submission-era evidence. Exact table-level venue use is not established for every folder |
| `5-5 mcqa seeds 0,1,2` | `archive/neurips_2026_submission_period/mcqa/5-5 mcqa seeds 0,1,2` | May 2026 run; original submission-era evidence. Exact table-level venue use is not established for every folder |
| `5-6 mcqa plot das overhaul seed0 v2` | `archive/neurips_2026_submission_period/mcqa/5-6 mcqa plot das overhaul seed0 v2` | May 2026 run; original submission-era evidence. Exact table-level venue use is not established for every folder |
| `5-6 mcqa stageA rowtopk ot-uot rerun` | `archive/neurips_2026_submission_period/mcqa/5-6 mcqa stageA rowtopk ot-uot rerun` | May 2026 run; original submission-era evidence. Exact table-level venue use is not established for every folder |
| `6-17 local heq binary method reruns` | `archive/icml_2026_workshop_period/heq/6-17 local heq binary method reruns` | June 2026 follow-up in the ICML workshop period; may overlap NeurIPS revisions, not a confirmed exclusive venue assignment |
| `6-17 mcqa added total variation` | `archive/icml_2026_workshop_period/mcqa/6-17 mcqa added total variation` | June 2026 follow-up in the ICML workshop period; may overlap NeurIPS revisions, not a confirmed exclusive venue assignment |
| `6-18 mcqa cosine brute-force large native` | `archive/icml_2026_workshop_period/mcqa/6-18 mcqa cosine brute-force large native` | June 2026 follow-up in the ICML workshop period; may overlap NeurIPS revisions, not a confirmed exclusive venue assignment |
| `6-18 mcqa plot native family norm seeds1-2` | `archive/icml_2026_workshop_period/mcqa/6-18 mcqa plot native family norm seeds1-2` | June 2026 follow-up in the ICML workshop period; may overlap NeurIPS revisions, not a confirmed exclusive venue assignment |
| `7-21 mcqa cosine brute-force fixed layers seeds1-2` | `archive/neurips_2026_revision_period/mcqa/7-21 mcqa cosine brute-force fixed layers seeds1-2` | July/August revision-period result; not selected as a current source above |
| `7-21 mcqa cosine brute-force rerun seeds0-2` | `archive/neurips_2026_revision_period/mcqa/7-21 mcqa cosine brute-force rerun seeds0-2` | July/August revision-period result; not selected as a current source above |
| `7-25 mcqa extended grid 200-200-200` | `archive/neurips_2026_revision_period/mcqa/7-25 mcqa extended grid 200-200-200` | July/August revision-period result; not selected as a current source above |
| `7-26 binary addition local baselines` | `archive/neurips_2026_revision_period/binary_addition/7-26 binary addition local baselines` | July/August revision-period result; not selected as a current source above |
| `7-26 mcqa MIB baselines` | `archive/neurips_2026_revision_period/mcqa/7-26 mcqa MIB baselines` | July/August revision-period result; not selected as a current source above |
| `7-29 boundless das` | `archive/neurips_2026_revision_period/mixed/7-29 boundless das` | July/August revision-period result; not selected as a current source above |
| `7-29 shared epsilon protocol binary addition` | `archive/neurips_2026_revision_period/binary_addition/7-29 shared epsilon protocol binary addition` | July/August revision-period result; not selected as a current source above |
| `7-30 standardized das` | `archive/neurips_2026_revision_period/mixed/7-30 standardized das` | July/August revision-period result; not selected as a current source above |
| `7-31 mcqa unified iia seed0` | `archive/neurips_2026_revision_period/mcqa/7-31 mcqa unified iia seed0` | July/August revision-period result; not selected as a current source above |
| `8-17 binary addition regularized 10seed rerun` | `archive/neurips_2026_revision_period/binary_addition/8-17 binary addition regularized 10seed rerun` | July/August revision-period result; not selected as a current source above |
| `8-17 binary addition regularized 10seed rerun cpu` | `archive/neurips_2026_revision_period/binary_addition/8-17 binary addition regularized 10seed rerun cpu` | July/August revision-period result; not selected as a current source above |
| `8-17 binary addition w8 h64 plot4 seed0` | `development/pilots/binary_addition/8-17 binary addition w8 h64 plot4 seed0` | 8-bit/variable-length/MLP extension; not the current 4-bit width-16 main table |
| `8-18 binary addition variable length gru h64 plot4 seed0` | `development/pilots/binary_addition/8-18 binary addition variable length gru h64 plot4 seed0` | 8-bit/variable-length/MLP extension; not the current 4-bit width-16 main table |
| `8-18 binary addition w8 mlp8 h64 plot4 seed0` | `development/pilots/binary_addition/8-18 binary addition w8 mlp8 h64 plot4 seed0` | 8-bit/variable-length/MLP extension; not the current 4-bit width-16 main table |
| `8-19 binary addition cross length gru h64 plot4 seed0` | `development/pilots/binary_addition/8-19 binary addition cross length gru h64 plot4 seed0` | 8-bit/variable-length/MLP extension; not the current 4-bit width-16 main table |
| `8-2 standardized binary addition rerun` | `archive/neurips_2026_revision_period/binary_addition/8-2 standardized binary addition rerun` | July/August revision-period result; not selected as a current source above |
| `8-25 binary addition d8 corrected 10seed local cpu` | `archive/neurips_2026_revision_period/binary_addition/8-25 binary addition d8 corrected 10seed local cpu` | July/August revision-period result; not selected as a current source above |
| `8-25 binary addition d8 corrected heldout 10seed local cpu` | `archive/neurips_2026_revision_period/binary_addition/8-25 binary addition d8 corrected heldout 10seed local cpu` | July/August revision-period result; not selected as a current source above |
| `8-26 binary addition paired current-code seed0` | `paper/iclr_2027/binary_addition/8-26 binary addition paired current-code seed0` | Reused unrestricted DAS/PLOT-DAS source, seeds 0-4 at hidden width 16; older date remains a current paper source |
| `8-3 ioi blind plot oracle das` | `archive/neurips_2026_revision_period/ioi/8-3 ioi blind plot oracle das` | July/August revision-period result; not selected as a current source above |
| `8-3 ioi brute force das` | `archive/neurips_2026_revision_period/ioi/8-3 ioi brute force das` | July/August revision-period result; not selected as a current source above |
| `8-3 ioi plot brute force das k1-144` | `archive/neurips_2026_revision_period/ioi/8-3 ioi plot brute force das k1-144` | July/August revision-period result; not selected as a current source above |
| `8-3 ioi plot brute force das k1-6` | `archive/neurips_2026_revision_period/ioi/8-3 ioi plot brute force das k1-6` | July/August revision-period result; not selected as a current source above |
| `8-6 mcqa` | `archive/neurips_2026_revision_period/mcqa/8-6 mcqa` | July/August revision-period result; not selected as a current source above |
| `8-8 mcqa pca dbm full-vector rerun` | `archive/neurips_2026_revision_period/mcqa/8-8 mcqa pca dbm full-vector rerun` | July/August revision-period result; not selected as a current source above |
| `9-12 mcqa single-stage combined seed0` | `development/iclr_2027_preliminary/mcqa/9-12 mcqa single-stage combined seed0` | September pilot/audit or predecessor; retained for provenance, not selected current table evidence |
| `9-12 mcqa single-stage finer widths seed0` | `development/iclr_2027_preliminary/mcqa/9-12 mcqa single-stage finer widths seed0` | September pilot/audit or predecessor; retained for provenance, not selected current table evidence |
| `9-12 mcqa single-stage native seed0` | `development/iclr_2027_preliminary/mcqa/9-12 mcqa single-stage native seed0` | September pilot/audit or predecessor; retained for provenance, not selected current table evidence |
| `9-13 mcqa token positions balanced OT seed0` | `supporting/iclr_2027/mcqa/9-13 mcqa token positions balanced OT seed0` | Token-position ablation |
| `9-14 mcqa plot-das artifact audit` | `development/iclr_2027_preliminary/mcqa/9-14 mcqa plot-das artifact audit` | September pilot/audit or predecessor; retained for provenance, not selected current table evidence |
| `9-14 mcqa single-stage uot seed0` | `development/iclr_2027_preliminary/mcqa/9-14 mcqa single-stage uot seed0` | September pilot/audit or predecessor; retained for provenance, not selected current table evidence |
| `9-14 mcqa token positions UOT seed0` | `supporting/iclr_2027/mcqa/9-14 mcqa token positions UOT seed0` | Token-position follow-up |
| `9-15 mcqa single-stage pca uot seed0` | `development/iclr_2027_preliminary/mcqa/9-15 mcqa single-stage pca uot seed0` | September pilot/audit or predecessor; retained for provenance, not selected current table evidence |
| `9-17 mcqa das coordinate uot seed0` | `development/iclr_2027_preliminary/mcqa/9-17 mcqa das coordinate uot seed0` | September pilot/audit or predecessor; retained for provenance, not selected current table evidence |
| `9-18 mcqa das dense topk seed0` | `development/iclr_2027_preliminary/mcqa/9-18 mcqa das dense topk seed0` | September pilot/audit or predecessor; retained for provenance, not selected current table evidence |
| `9-18 mcqa single-stage uot seeds1-3` | `development/iclr_2027_preliminary/mcqa/9-18 mcqa single-stage uot seeds1-3` | September pilot/audit or predecessor; retained for provenance, not selected current table evidence |
| `9-19 binary addition staging OT comparison seed0` | `development/iclr_2027_preliminary/binary_addition/9-19 binary addition staging OT comparison seed0` | September pilot/audit or predecessor; retained for provenance, not selected current table evidence |
| `9-19 binary addition staging UOT beta sweep seed0` | `development/iclr_2027_preliminary/binary_addition/9-19 binary addition staging UOT beta sweep seed0` | September pilot/audit or predecessor; retained for provenance, not selected current table evidence |
| `9-19 binary addition staging comparison` | `development/incomplete_or_excluded/binary_addition/9-19 binary addition staging comparison` | Stopped, discarded timing, partial pilot or plan; not final paper evidence |
| `9-20 binary addition staging five seeds` | `development/iclr_2027_preliminary/binary_addition/9-20 binary addition staging five seeds` | September pilot/audit or predecessor; retained for provenance, not selected current table evidence |
| `9-21 mcqa global staged five seeds` | `development/iclr_2027_preliminary/mcqa/9-21 mcqa global staged five seeds` | September pilot/audit or predecessor; retained for provenance, not selected current table evidence |
| `9-22 binary addition six-seed selected-settings` | `supporting/iclr_2027/binary_addition/9-22 binary addition six-seed selected-settings` | Selected-setting companion summary; primary report uses full-grid serial runtime |
| `9-22 binary addition staging parallel timing discarded` | `development/incomplete_or_excluded/binary_addition/9-22 binary addition staging parallel timing discarded` | Stopped, discarded timing, partial pilot or plan; not final paper evidence |
| `9-22 binary addition staging seed5 full` | `development/iclr_2027_preliminary/binary_addition/9-22 binary addition staging seed5 full` | September pilot/audit or predecessor; retained for provenance, not selected current table evidence |
| `9-22 binary addition staging six seeds serial` | `paper/iclr_2027/binary_addition/9-22 binary addition staging six seeds serial` | Unrestricted global/staged native/PCA OT main comparison; paper uses seeds 0-4 at width 16 |
| `9-22 mcqa capped DBM and DAS five seeds A40` | `paper/iclr_2027/mcqa/9-22 mcqa capped DBM and DAS five seeds A40` | DBM baseline sources; Full DAS portion superseded by September 24 corrected runs |
| `9-22 mcqa global staged five seeds A40` | `paper/iclr_2027/mcqa/9-22 mcqa global staged five seeds A40` | Original global/staged UOT candidates and partition evidence; runtime table superseded by full-beta replay |
| `9-23 binary addition capped five seeds` | `supporting/iclr_2027/binary_addition/9-23 binary addition capped five seeds` | Separate capped comparison; not the unrestricted main table |
| `9-23 binary addition uncapped comparison` | `paper/iclr_2027/binary_addition/9-23 binary addition uncapped comparison` | Derived table and handle index joining original runs; not an independent fresh run |
| `9-23 mcqa MIB DAS L24 d1152 A40` | `development/iclr_2027_preliminary/mcqa/9-23 mcqa MIB DAS L24 d1152 A40` | September pilot/audit or predecessor; retained for provenance, not selected current table evidence |
| `9-23 mcqa MIB DAS L24 d1152 ten seeds A40` | `development/iclr_2027_preliminary/mcqa/9-23 mcqa MIB DAS L24 d1152 ten seeds A40` | September pilot/audit or predecessor; retained for provenance, not selected current table evidence |
| `9-23 mcqa global staged OT five seeds plan` | `development/incomplete_or_excluded/mcqa/9-23 mcqa global staged OT five seeds plan` | Stopped, discarded timing, partial pilot or plan; not final paper evidence |
| `9-24 heq cpu paper rerun left grid` | `paper/iclr_2027/heq/9-24 heq cpu paper rerun left grid` | Saved DAS runs, fixed banks, original PLOT trials and broad epsilon ablation |
| `9-24 heq cpu plot 6eps stable rerun` | `paper/iclr_2027/heq/9-24 heq cpu plot 6eps stable rerun` | Current stable six-epsilon PLOT comparison with saved DAS, seeds 1-10 |
| `9-24 mcqa UOT budget1152 full beta five seeds A40` | `paper/iclr_2027/mcqa/9-24 mcqa UOT budget1152 full beta five seeds A40` | Current five-seed native/PCA global/staged UOT main-table reporting, full beta replay |
| `9-24 mcqa corrected DAS five seeds A40` | `paper/iclr_2027/mcqa/9-24 mcqa corrected DAS five seeds A40` | Current saved Full DAS/PLOT-DAS main-table sources, corrected MIB loss and substring score |
| `9-24 mcqa single stage matching seed0 stopped` | `development/incomplete_or_excluded/mcqa/9-24 mcqa single stage matching seed0 stopped` | Stopped, discarded timing, partial pilot or plan; not final paper evidence |
| `9-25 mcqa KL four PLOT methods five seeds A40` | `supporting/iclr_2027/mcqa/9-25 mcqa KL four PLOT methods five seeds A40` | Completed KL/family-wise four-method ablation |
| `9-25 mcqa PLOT DAS KL stage A five seeds A40` | `supporting/iclr_2027/mcqa/9-25 mcqa PLOT DAS KL stage A five seeds A40` | Completed KL Stage-A DAS follow-up; final collected results verified, excludes failed strict-scorer attempt |
| `9-25 mcqa effect signatures seed0 A40` | `supporting/iclr_2027/mcqa/9-25 mcqa effect signatures seed0 A40` | Seed-0 effect-signature ablation; pre-fix TV diagnostics excluded |
| `9-25 mcqa effect signatures seeds1-4 A40` | `supporting/iclr_2027/mcqa/9-25 mcqa effect signatures seeds1-4 A40` | Effect-signature ablation completion, paired with seed 0 |
| `9-25 mcqa two stage cosine KL five seeds A40` | `supporting/iclr_2027/mcqa/9-25 mcqa two stage cosine KL five seeds A40` | Five-seed cosine/KL follow-up |
| `9-25 mcqa two stage cosine full-layer five seeds A40` | `supporting/iclr_2027/mcqa/9-25 mcqa two stage cosine full-layer five seeds A40` | Five-seed alternative-matching ablation |
| `9-25 mcqa two stage cosine full-layer seed0 A40` | `development/iclr_2027_preliminary/mcqa/9-25 mcqa two stage cosine full-layer seed0 A40` | September pilot/audit or predecessor; retained for provenance, not selected current table evidence |
| `9-6 binary addition paired dbm full-vector 10seed local cpu` | `supporting/iclr_2027/binary_addition/9-6 binary addition paired dbm full-vector 10seed local cpu` | Unrestricted baseline comparison source referenced by the derived report |
| `9-9 binary addition mcqa-style pca bands 10seed local cpu` | `supporting/iclr_2027/binary_addition/9-9 binary addition mcqa-style pca bands 10seed local cpu` | Earlier PCA-band source referenced by the derived report; not the new one-/two-stage table rows |
| `mcqa_staged_native_family_ot_stage_a_5seed_a40_20260925` | `supporting/iclr_2027/mcqa/9-25 mcqa staged native balanced OT Stage A five seeds A40` | Completed five-seed balanced-OT versus UOT Stage-A-only comparison |

## Earlier artifact and compaction record

The following preserved notes use original dated paths, which now resolve through
compatibility links. Results remain local and Git-ignored.

Local experiment artifacts live in `results/`. The historical folder convention was `M-D <task> <experiment> [seed scope]`; original runner-created folders remain inside each dated folder. `results/` is excluded by `.gitignore`, so this index is the repository's portable map, while the data files are available only in workspaces that have the local results directory. The historical-name table below maps old paths in logs and conversations to the canonical dated folders.

## Post-submission artifact policy (2026-09-28)

Run-result JSON/CSV/Markdown files remain available, including historical runs. Saved effect/signature tensors and their archived copies were removed; rerunning code that assumes those caches still exist will need to regenerate them. Intermediate checkpoint sweeps were pruned for the completed five-seed MCQA baselines:

| Experiment | Retained learned handles | Selection source |
|---|---:|---|
| `9-24 mcqa corrected DAS five seeds A40/full_das` | 10 selected checkpoints (one answer-pointer and one answer-token handle per seed) | `collected/per_seed.json`, `Full DAS` rows |
| `9-22 mcqa capped DBM and DAS five seeds A40/dbm` | 30 selected checkpoints (three DBM methods, two answer positions, five seeds) | `collected/per_seed.json`, `DBM` rows |

The superseded September 22 Full DAS layer-sweep checkpoints were removed; use the corrected September 24 run for learned DAS handles. Older July/August DBM layer checkpoints were also removed in favor of the September 22 selected DBM handles. The September 18 single-stage UOT seed-1–3 transfer archives were unpacked as result-only files under `downloads/unpacked_results/`, without their signature tensors. Older PCA bases and other method-specific checkpoints remain where supersession was not unambiguous; they have not been reclassified as final handles.

For the September 20 five-seed and September 22 six-seed binary-addition staging runs, the per-run `banks.json`, `calibration.json`, and `evidence.json` files were removed on 2026-09-29. All 320 per-run `result.json` summaries and `frozen_handles.json` selections remain, along with their protocols and aggregate reports. Reproducing the full candidate-search trace now requires rerunning those sweeps.

### Further result compaction (2026-09-29)

The following older result folders were trimmed or compacted without removing their selected result summaries, frozen selections, or selected learned checkpoints/bases:

| Folders | Removed/regenerated material | Retained basis for interpretation |
|---|---|---|
| `9-19 binary addition staging OT comparison seed0`; `9-19 binary addition staging UOT beta sweep seed0`; `9-22 binary addition staging parallel timing discarded`; `9-22 binary addition staging seed5 full` | Per-run `banks.json`, `calibration.json`, `evidence.json` | Per-run `result.json`, `frozen_handles.json`, protocols |
| `8-17 binary addition w8 h64 plot4 seed0`; `8-18 binary addition variable length gru h64 plot4 seed0`; `8-18 binary addition w8 mlp8 h64 plot4 seed0`; `8-19 binary addition cross length gru h64 plot4 seed0` | Embedded resolution-level trial arrays in seed summaries; cross-length cache | Selected configurations, final/aggregate metrics, methods, protocol |
| `9-18 mcqa single-stage uot seeds1-3`; `8-6 mcqa` | Expanded per-candidate `coupling_*.json` files; large candidate/method payloads in JSON; redundant verified-output archives in `8-6` | Summary, frozen selection, selected-test results, selected joint configuration and metric rows |
| `9-17 mcqa das coordinate uot seed0`; `9-18 mcqa das dense topk seed0` | Candidate calibrations/couplings/selections and unselected DAS checkpoints | Summary, frozen selection, selected tests, and selected answer-pointer/token dimension-768 checkpoints |
| `5-5 mcqa seeds 0,1,2` | PCA bases embedded redundantly in JSON | Run/site metrics and external `.pt` PCA bases referenced by each compacted JSON |
| `6-17 local heq binary method reruns` | Per-example records and method payloads, full trial histories and fit diagnostics | Aggregate equality metrics, selected methods/best trials, final tests, and method result rows |

Compacted JSON files carry an `_compaction` field recording their prior byte size and removed field names. They are result/handle references, not full candidate-search archives: reconstructing exact candidate traces or per-example predictions requires rerunning the corresponding experiment. Because `results/` is Git-ignored, this table documents local artifacts and does not imply those artifacts are uploaded to GitHub.

## Recent MCQA studies

| Study | Local folder | Main entry point | Status |
|---|---|---|---|
| Balanced OT across last token, correct symbol, and following period | `results/9-13 mcqa token positions balanced OT seed0` | `RESULTS.md` | Verified seed 0 |
| UOT follow-up across the same positions | `results/9-14 mcqa token positions UOT seed0` | `RESULTS.md` | Verified seed 0 |
| Global single-stage native/PCA UOT | `results/9-18 mcqa single-stage uot seeds1-3` | `comparison_seeds0-3.md` | Six Delta A40 jobs complete; seed 0 included from earlier runs |
| Global versus staged native/PCA UOT with dimension budgets and global-handle lesions | `results/9-21 mcqa global staged five seeds` | `collected/RESULTS_SUMMARY.md` | 20 of 20 runs complete |
| A40 rerun of global/staged native/PCA UOT, all budgets and global-handle lesions | `results/9-22 mcqa global staged five seeds A40` | `collected/README.md` | 20 of 20 summaries complete |
| Capped DBM, PLOT-DAS, and Full DAS baselines | `results/9-22 mcqa capped DBM and DAS five seeds A40` | `collected/table_with_full_vector.md` | Five-seed baseline suite complete |
| Global/staged balanced-OT follow-up plan | `results/9-23 mcqa global staged OT five seeds plan` | `run_plan.json` | Plan only; do not treat as results |
| MIB-style L24, dimension-1152 DAS checks | `results/9-23 mcqa MIB DAS L24 d1152 A40`; `results/9-23 mcqa MIB DAS L24 d1152 ten seeds A40` | `ten_seed_summary.json` in the ten-seed folder | Layer-24 validation and ten-seed follow-up |
| Corrected PLOT-DAS and Full DAS | `results/9-24 mcqa corrected DAS five seeds A40` | `collected/comparison_with_previous_methods.md` | Five-seed corrected-loss results complete |
| Budget-1152, full-beta global/staged UOT replay with saved caches | `results/9-24 mcqa UOT budget1152 full beta five seeds A40` | `collected/comparison_all_beta_budget1152.md` | 20 method-seed results complete; fresh-equivalent runtimes |
| Two-stage native UOT effect-signature comparison: alphabet-logit, TV, KL | `results/9-25 mcqa effect signatures seed0 A40`; `results/9-25 mcqa effect signatures seeds1-4 A40` | `collected/comparison.md` in the seeds-1–4 folder | Five seeds complete; exclude `diagnostics/pre_fix_tv` |
| KL versus alphabet-logit UOT for global/staged native/PCA | `results/9-25 mcqa KL four PLOT methods five seeds A40` | `collected/aggregate.json`, `collected/per_seed.csv` | All eight five-seed comparison rows complete; all experiment pods terminated |
| Two-stage native cosine versus full-layer matching | `results/9-25 mcqa two stage cosine full-layer seed0 A40`; `results/9-25 mcqa two stage cosine full-layer five seeds A40` | `five_seed_comparison.md` in the five-seed folder | Seed-0 pilot and five-seed follow-up complete |
| Two-stage native cosine with KL signatures | `results/9-25 mcqa two stage cosine KL five seeds A40` | `five_seed_comparison.md` | Five seeds complete |
| Single-stage matching pilot | `results/9-24 mcqa single stage matching seed0 stopped` | `run_manifest.json` | Stopped; partial outputs are not the final comparison |
| PLOT-DAS KL Stage A follow-up | `results/9-25 mcqa PLOT DAS KL stage A five seeds A40` | `collected/comparison_with_previous_plot_das.md` | Completed five-seed follow-up verified October 4; failed strict-scorer attempt excluded |
| Unified IIA MCQA seed-0 rerun | `results/7-31 mcqa unified iia seed0` | `mcqa_unified_iia_seed0_v1_plot_mcqa_hierarchical_sweep/hierarchical_sweep_summary.txt` | Historical rerun |

## Recent binary-addition studies

| Study | Local folder | Main entry point | Status |
|---|---|---|---|
| Eight-strategy balanced-OT staging pilot | `results/9-19 binary addition staging OT comparison seed0` | `summary.md` | Seed 0 complete |
| Eight-strategy UOT beta-sweep pilot | `results/9-19 binary addition staging UOT beta sweep seed0` | `summary.md` | Seed 0 complete |
| Five-seed OT/UOT staging comparison | `results/9-20 binary addition staging five seeds` | `report.md` | 160 runs complete |
| Capped binary-addition follow-up | `results/9-23 binary addition capped five seeds` | Folder-level results | Five-seed run |
| Uncapped binary-addition comparison assembled from earlier runs | `results/9-23 binary addition uncapped comparison` | `9-23 binary addition uncapped five-seed comparison from prior runs.json` | Derived comparison; seed-1 handle convenience index alongside it, with authoritative source paths inside |
| Earlier staging comparison scratch run | `results/9-19 binary addition staging comparison` | `h8/seed_0/staged_native` | Partial; use the seed-0 OT comparison above for the completed pilot |

## Renamed top-level folders

These old names may still appear in conversation history, archived logs, or remote Delta paths. The mapping below concerns local `results/` folders only; remote execution paths in provenance files retain their original names.

| Former name under `results/` | Current name under `results/` |
|---|---|
| `5-5 mcqa full_das_5seed` | `5-5 mcqa full das 5 seeds` |
| `5-5 mcqa plot_pca_partition_band_sweep_compact` | `5-5 mcqa pca partition band sweep` |
| `5-6 mcqa seed0 plot_das_overhaul_v2` | `5-6 mcqa plot das overhaul seed0 v2` |
| `6-18 mcqa_cosine_bruteforce_large_native` | `6-18 mcqa cosine brute-force large native` |
| `6-18 mcqa_plot_native_family_norm_seed1_seed2` | `6-18 mcqa plot native family norm seeds1-2` |
| `7-21 cosine:brute-force rerun` | `7-21 mcqa cosine brute-force rerun seeds0-2` |
| `7-21 cosine:brute-force rerun seeds1-2` | `7-21 mcqa cosine brute-force fixed layers seeds1-2` |
| `7-31 ` | `7-31 mcqa unified iia seed0` |
| `9-19 binary addition staging comparison v1` | `9-19 binary addition staging OT comparison seed0` |
| `9-19 binary addition staging UOT beta sweep v1` | `9-19 binary addition staging UOT beta sweep seed0` |
| `mcqa_token_positions_20260913` | `9-13 mcqa token positions balanced OT seed0` |
| `mcqa_token_positions_uot_20260914` | `9-14 mcqa token positions UOT seed0` |
| `mcqa_global_staged_five_seed_20260921` | `9-21 mcqa global staged five seeds` |
| `mcqa_capped_suite_a40_20260922` | `9-22 mcqa capped DBM and DAS five seeds A40` |
| `mcqa_das_corrected_5seed_a40_20260924` | `9-24 mcqa corrected DAS five seeds A40` |
| `mcqa_das_mib_l24_d1152_a40_20260923` | `9-23 mcqa MIB DAS L24 d1152 A40` |
| `mcqa_das_mib_l24_d1152_tenseed_a40_20260923` | `9-23 mcqa MIB DAS L24 d1152 ten seeds A40` |
| `mcqa_effect_signature_ablation_seed0_20260925` | `9-25 mcqa effect signatures seed0 A40` |
| `mcqa_effect_signature_ablation_seeds1to4_20260925` | `9-25 mcqa effect signatures seeds1-4 A40` |
| `mcqa_global_staged_five_seed_a40_20260922` | `9-22 mcqa global staged five seeds A40` |
| `mcqa_global_staged_ot_five_seed_20260923` | `9-23 mcqa global staged OT five seeds plan` |
| `mcqa_kl_four_methods_5seed_a40_20260925` | `9-25 mcqa KL four PLOT methods five seeds A40` |
| `mcqa_plot_das_kl_stagea_5seed_a40_20260925` | `9-25 mcqa PLOT DAS KL stage A five seeds A40` |
| `mcqa_single_stage_matching_seed0_20260924` | `9-24 mcqa single stage matching seed0 stopped` |
| `mcqa_two_stage_cosine_kl_5seed_a40_20260925` | `9-25 mcqa two stage cosine KL five seeds A40` |
| `mcqa_two_stage_matching_5seed_a40_20260925` | `9-25 mcqa two stage cosine full-layer five seeds A40` |
| `mcqa_two_stage_matching_seed0_20260925` | `9-25 mcqa two stage cosine full-layer seed0 A40` |
| `mcqa_uot1152_fullbeta_cached_a40_20260924` | `9-24 mcqa UOT budget1152 full beta five seeds A40` |

Use the grouped canonical paths in the complete run map for new references. Earlier cleanup retained two short-name compatibility symlinks: `mcqa_plot_das_kl_stagea_5seed_a40_20260925` and `mcqa_das_corrected_5seed_a40_20260924`. The KL follow-up is now complete; these links remain for historical references, alongside the new dated-path links. The two former top-level `9-23 binary addition uncapped ... .json` files now live together in `results/9-23 binary addition uncapped comparison/`.
