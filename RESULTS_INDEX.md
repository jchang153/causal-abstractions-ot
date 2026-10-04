# Experiment results index

Runs are stored under `results/versions/`. No files were deleted during the
organization. [Paper provenance](docs/experiment_history.md) identifies current
table sources; [the portable inventory](docs/experiment_inventory.json) records
canonical paths and roles. Result data remain local and Git-ignored.

## Versions

| Version | Role | Run folders |
|---|---|---:|
| `v1` | Initial experiments | 6 |
| `v2` | Intermediate follow-ups | 4 |
| `v3` | Expanded baselines and pilots | 24 |
| `v4` | Current comparisons, ablations and development | 43 |

## Complete run inventory

Current `paper` folders contain selected table evidence, `supporting` contains
completed companion studies, and `preliminary`, `pilots` and `excluded` remain
separate. Consult saved configurations before comparing scores.

| Version | Folder under `results/versions/` | Role |
|---|---|---|
| `v1` | `v1/historical/mcqa/mcqa_full_ot_uot_layerwise` | historical |
| `v1` | `v1/historical/mcqa/mcqa_full_das_5_seeds` | historical |
| `v1` | `v1/historical/mcqa/mcqa_pca_partition_band_sweep` | historical |
| `v1` | `v1/historical/mcqa/mcqa_seeds_0_1_2` | historical |
| `v1` | `v1/historical/mcqa/mcqa_plot_das_overhaul_seed0_v2` | historical |
| `v1` | `v1/historical/mcqa/mcqa_stagea_rowtopk_ot_uot_rerun` | historical |
| `v2` | `v2/historical/heq/local_heq_binary_method_reruns` | historical |
| `v2` | `v2/historical/mcqa/mcqa_added_total_variation` | historical |
| `v2` | `v2/historical/mcqa/mcqa_cosine_brute_force_large_native` | historical |
| `v2` | `v2/historical/mcqa/mcqa_plot_native_family_norm_seeds1_2` | historical |
| `v3` | `v3/historical/mcqa/mcqa_cosine_brute_force_fixed_layers_seeds1_2` | historical |
| `v3` | `v3/historical/mcqa/mcqa_cosine_brute_force_rerun_seeds0_2` | historical |
| `v3` | `v3/historical/mcqa/mcqa_extended_grid_200_200_200` | historical |
| `v3` | `v3/historical/binary_addition/binary_addition_local_baselines` | historical |
| `v3` | `v3/historical/mcqa/mcqa_mib_baselines` | historical |
| `v3` | `v3/historical/mixed/boundless_das` | historical |
| `v3` | `v3/historical/binary_addition/shared_epsilon_protocol_binary_addition` | historical |
| `v3` | `v3/historical/mixed/standardized_das` | historical |
| `v3` | `v3/historical/mcqa/mcqa_unified_iia_seed0` | historical |
| `v3` | `v3/historical/binary_addition/binary_addition_regularized_10seed_rerun` | historical |
| `v3` | `v3/historical/binary_addition/binary_addition_regularized_10seed_rerun_cpu` | historical |
| `v3` | `v3/pilots/binary_addition/binary_addition_w8_h64_plot4_seed0` | pilots |
| `v3` | `v3/pilots/binary_addition/binary_addition_variable_length_gru_h64_plot4_seed0` | pilots |
| `v3` | `v3/pilots/binary_addition/binary_addition_w8_mlp8_h64_plot4_seed0` | pilots |
| `v3` | `v3/pilots/binary_addition/binary_addition_cross_length_gru_h64_plot4_seed0` | pilots |
| `v3` | `v3/historical/binary_addition/standardized_binary_addition_rerun` | historical |
| `v3` | `v3/historical/binary_addition/binary_addition_d8_corrected_10seed_local_cpu` | historical |
| `v3` | `v3/historical/binary_addition/binary_addition_d8_corrected_heldout_10seed_local_cpu` | historical |
| `v4` | `v4/paper/binary_addition/binary_addition_paired_current_code_seed0` | paper |
| `v3` | `v3/historical/ioi/ioi_blind_plot_oracle_das` | historical |
| `v3` | `v3/historical/ioi/ioi_brute_force_das` | historical |
| `v3` | `v3/historical/ioi/ioi_plot_brute_force_das_k1_144` | historical |
| `v3` | `v3/historical/ioi/ioi_plot_brute_force_das_k1_6` | historical |
| `v3` | `v3/historical/mcqa/mcqa` | historical |
| `v3` | `v3/historical/mcqa/mcqa_pca_dbm_full_vector_rerun` | historical |
| `v4` | `v4/preliminary/mcqa/mcqa_single_stage_combined_seed0` | preliminary |
| `v4` | `v4/preliminary/mcqa/mcqa_single_stage_finer_widths_seed0` | preliminary |
| `v4` | `v4/preliminary/mcqa/mcqa_single_stage_native_seed0` | preliminary |
| `v4` | `v4/supporting/mcqa/mcqa_token_positions_balanced_ot_seed0` | supporting |
| `v4` | `v4/preliminary/mcqa/mcqa_plot_das_artifact_audit` | preliminary |
| `v4` | `v4/preliminary/mcqa/mcqa_single_stage_uot_seed0` | preliminary |
| `v4` | `v4/supporting/mcqa/mcqa_token_positions_uot_seed0` | supporting |
| `v4` | `v4/preliminary/mcqa/mcqa_single_stage_pca_uot_seed0` | preliminary |
| `v4` | `v4/preliminary/mcqa/mcqa_das_coordinate_uot_seed0` | preliminary |
| `v4` | `v4/preliminary/mcqa/mcqa_das_dense_topk_seed0` | preliminary |
| `v4` | `v4/preliminary/mcqa/mcqa_single_stage_uot_seeds1_3` | preliminary |
| `v4` | `v4/preliminary/binary_addition/binary_addition_staging_ot_comparison_seed0` | preliminary |
| `v4` | `v4/preliminary/binary_addition/binary_addition_staging_uot_beta_sweep_seed0` | preliminary |
| `v4` | `v4/excluded/binary_addition/binary_addition_staging_comparison` | excluded |
| `v4` | `v4/preliminary/binary_addition/binary_addition_staging_five_seeds` | preliminary |
| `v4` | `v4/preliminary/mcqa/mcqa_global_staged_five_seeds` | preliminary |
| `v4` | `v4/supporting/binary_addition/binary_addition_six_seed_selected_settings` | supporting |
| `v4` | `v4/excluded/binary_addition/binary_addition_staging_parallel_timing_discarded` | excluded |
| `v4` | `v4/preliminary/binary_addition/binary_addition_staging_seed5_full` | preliminary |
| `v4` | `v4/paper/binary_addition/binary_addition_staging_six_seeds_serial` | paper |
| `v4` | `v4/paper/mcqa/mcqa_capped_dbm_and_das_five_seeds_a40` | paper |
| `v4` | `v4/paper/mcqa/mcqa_global_staged_five_seeds_a40` | paper |
| `v4` | `v4/supporting/binary_addition/binary_addition_capped_five_seeds` | supporting |
| `v4` | `v4/paper/binary_addition/binary_addition_uncapped_comparison` | paper |
| `v4` | `v4/preliminary/mcqa/mcqa_mib_das_l24_d1152_a40` | preliminary |
| `v4` | `v4/preliminary/mcqa/mcqa_mib_das_l24_d1152_ten_seeds_a40` | preliminary |
| `v4` | `v4/excluded/mcqa/mcqa_global_staged_ot_five_seeds_plan` | excluded |
| `v4` | `v4/paper/heq/heq_cpu_paper_rerun_left_grid` | paper |
| `v4` | `v4/paper/heq/heq_cpu_plot_6eps_stable_rerun` | paper |
| `v4` | `v4/paper/mcqa/mcqa_uot_budget1152_full_beta_five_seeds_a40` | paper |
| `v4` | `v4/paper/mcqa/mcqa_corrected_das_five_seeds_a40` | paper |
| `v4` | `v4/excluded/mcqa/mcqa_single_stage_matching_seed0_stopped` | excluded |
| `v4` | `v4/supporting/mcqa/mcqa_kl_four_plot_methods_five_seeds_a40` | supporting |
| `v4` | `v4/supporting/mcqa/mcqa_plot_das_kl_stage_a_five_seeds_a40` | supporting |
| `v4` | `v4/supporting/mcqa/mcqa_effect_signatures_seed0_a40` | supporting |
| `v4` | `v4/supporting/mcqa/mcqa_effect_signatures_seeds1_4_a40` | supporting |
| `v4` | `v4/supporting/mcqa/mcqa_two_stage_cosine_kl_five_seeds_a40` | supporting |
| `v4` | `v4/supporting/mcqa/mcqa_two_stage_cosine_full_layer_five_seeds_a40` | supporting |
| `v4` | `v4/preliminary/mcqa/mcqa_two_stage_cosine_full_layer_seed0_a40` | preliminary |
| `v4` | `v4/supporting/binary_addition/binary_addition_paired_dbm_full_vector_10seed_local_cpu` | supporting |
| `v4` | `v4/supporting/binary_addition/binary_addition_mcqa_style_pca_bands_10seed_local_cpu` | supporting |
| `v4` | `v4/supporting/mcqa/mcqa_staged_native_balanced_ot_stage_a_five_seeds_a40` | supporting |

## Previously compacted artifacts

Earlier compaction retained summary JSON/CSV/Markdown files, frozen selections,
selected learned handles and bases where available. Effect/signature caches,
unselected checkpoint sweeps, some candidate-search traces, redundant PCA
payloads and per-example diagnostics had already been removed. Those removals
precede this organization; no files were deleted during the folder moves.

The corrected MCQA Full DAS suite retains ten selected checkpoints. The DBM
suite retains thirty selected checkpoints. Superseded Full DAS layer sweeps and
older unselected DBM layers require a rerun. Completed binary staging sweeps
retain result summaries and frozen handles, but some bank, calibration and
evidence traces require regeneration. Compacted JSON payloads record their
changes in `_compaction` fields.

The KL-guided PLOT-DAS and balanced-OT Stage-A follow-ups are complete. Failed
strict-scorer attempts, canceled full-run plans, partial matching runs and
parallel timing experiments are excluded from current table evidence. Saved
MCQA DAS results use the MIB substring scorer; current baseline code uses strict
normalized full-vocabulary top-1 IIA.
