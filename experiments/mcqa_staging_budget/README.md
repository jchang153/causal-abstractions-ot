# MCQA paper execution workflows (September 2026)

These local research scripts launched and collected the current ICLR 2027
five-seed runs. They are distinct from the older generic MCQA staged sweep.
See [paper provenance](../../docs/experiment_history.md) and the
[results index](../../RESULTS_INDEX.md) for canonical artifact paths and statuses.

| Workflow | Source files | Saved evidence |
|---|---|---|
| Global/staged native/PCA UOT | `run_focused.py`, `collect_focused.py` | September 22 A40 comparison |
| Full-beta budget-1152 replay | `launch_cached_a40_tasks.py`, `collect_cached_1152.py` | September 24 main-table UOT runtime rows |
| Corrected Full DAS/PLOT-DAS | `evaluate_full_das_corrected.py`, `collect_das_corrected.py`, `task_runpod_*das_corrected.sh` | September 24 completed corrected DAS suite |
| Effect signatures / four KL methods | `collect_signature_five_seed.py`, `collect_kl_four_methods.py` | September 25 completed ablations |
| Cosine/full-layer matching | `run_two_stage_matching.py`, `collect_two_stage_cosine_kl.py` | September 25 matching comparisons |
| KL-guided DAS | `collect_plot_das_kl_stagea.py`, `task_runpod_plot_das_kl_*.sh` | Completed five-seed follow-up, not the old incomplete-attempt status |
| Balanced OT versus UOT Stage A | `collect_stage_a_ot_comparison.py`, `task_runpod_stage_a_ot.sh` | Completed Stage-A-only five-seed comparison |

Paper scripts and generated task CSVs were local/uncommitted before this cleanup.
The CSVs are run artifacts and have been moved under the corrected DAS paper
results folder, with compatibility file links. The organization commit adds this
catalog; it does not publish or rerun these cloud jobs or change their protocols.

Saved DAS scoring and the latest standalone baseline scoring differ. Use saved
`run_manifest.json`, source snapshots, partition digests and collected protocol
notes for exact replay. Cluster paths and resource IDs in old scripts are
historical execution metadata and must be adapted for a new run.
