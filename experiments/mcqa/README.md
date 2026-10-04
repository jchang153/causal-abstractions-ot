# MCQA

## Current paper versus the latest code

The main-table sources are the full-beta UOT replay and corrected
Full DAS/PLOT-DAS, plus the DBM baseline. Five-seed signature and
matching ablations live in the supporting studies. The completed
KL Stage-A DAS follow-up is distinct from its retained failed attempts.

Saved DAS scores use the MIB generated-answer substring relation;
current `das.py` selects and scores with strict normalized full-vocabulary top-1
IIA. Follow [BASELINE_DAS.md](BASELINE_DAS.md) for current-code experiments, and
[paper provenance](../../docs/experiment_history.md) for frozen historical results.
Do not label old cached metrics as outputs of the newer agreement checker.

The legacy staged/Delta entry points below remain available; the newest paper
GPU workflows are documented in [../mcqa_staging_budget/](../mcqa_staging_budget/README.md).

Main-paper MCQA code lives here. The benchmark evaluates Gemma-2-2B on CopyColors-style multiple-choice prompts and localizes the abstract variables `answer_pointer` and `answer_token`.

## Evaluation metric

Every calibration, selection, and test verdict uses `iia_acc`: Gemma's actual
full-vocabulary top-1 next token is decoded, NFKC-normalized, stripped, folded
to uppercase, and accepted only when it is exactly one ASCII symbol A-Z matching
the causal interchange's final expected answer. Alphabet-restricted and raw
token-ID accuracies are diagnostics only. OT/UOT/cosine effect signatures remain
projected onto the 26 answer symbols; full-layer evaluates candidate
interventions directly with `iia_acc`. Calibration `iia_acc` is pooled over all
calibration examples, so counterfactual families contribute in proportion to
their sample counts for every method. Per-family accuracies remain diagnostic
only and never affect model or hyperparameter selection.

Method families:

- `PLOT`: Stage A UOT layer localization.
- `PLOT-native`: Stage A plus native-coordinate Stage B handles.
- `PLOT-PCA`: Stage A plus PCA-basis Stage B handles.
- `PLOT-DAS`: DAS restricted to the Stage A layer.
- `PLOT-native-DAS`: full-layer DAS with its dimension grid centered on the native Stage B effective dimension.
- `PLOT-PCA-DAS`: DAS over the full retained PCA basis with its dimension grid centered on the PCA Stage B effective dimension.
- `Full DAS`: DAS over all layers and the full subspace grid.
- `bDAS`: Boundless DAS over all layers with a learned rotated-prefix boundary.
- `DBM`: MIB-style desiderata-based masking in the canonical, PCA, or Gemma Scope SAE basis.
- `Full layer`: the pre-DAS control that swaps the entire last-token residual vector at one layer.

Main entry points:

- `mcqa_delta_hierarchical_sweep.py`: serial staged paper sweep.
- `mcqa_delta_hierarchical_parallel.py`: staged task planner/aggregator for cluster runs.
- `mcqa_run_cloud.py`: configurable single-run launcher, including full DAS.
- `mcqa_paper_runtime.py`: paper-runtime summarizer.
- `mcqa_boundless_das.py`: selected-only-test Full Boundless DAS runner.
- `mcqa_plot_das_pca_support.py`: PLOT-PCA-DAS runner that consumes the cached
  Stage B PCA support and basis without repeating localization.

PLOT sweeps select one epsilon globally across abstract variables.  Within each
epsilon, each variable selects its best native resolution or PCA support config
using calibration data, and the epsilon score is the equal-weight average of
those best variable scores.  Only the frozen winning configurations are tested.
The paper-facing PCA sweep uses `1,2,4,8,16,32,64` equal-width bands; the
PLOT-PCA-DAS stage consumes the calibration-selected PLOT-PCA support and its
effective-dimension hint.
Paper runtime charges every resolution/config evaluated at the selected epsilon
plus the final selected test evaluations; it does not charge only the
retrospective winning resolution, nor the sweep over unselected epsilons.
- `mcqa_plot_layer.py`, `mcqa_plot_native_support.py`, `mcqa_ot_pca_focus.py`, `mcqa_plot_das_layer.py`, `mcqa_plot_das_native_support.py`: individual stage runners.
- `mcqa_dbm_baselines.py`: resumable DBM/full-layer sweep. It selects layers using calibration accuracy and evaluates only the frozen selected layer on test. Factual filtering defaults to batch size 64 independently of the evaluation batch size, matching the PLOT/DAS data path. Pass `--partition-reference PATH` (optionally with a `{seed}` placeholder) to require exact fit/calibration/test partition equality with a PLOT or Full-DAS artifact.
- `slurm/submit_delta_mcqa_boundless_das.sh`: submit the three-seed bDAS array on Delta.
- `slurm/submit_delta_mcqa_corrected_reruns.sh`: submit the three-seed unified-IIA
  reruns for PLOT, PLOT-guided DAS, Full DAS, MIB, and bDAS.
- `slurm/run_delta_mcqa_one_seed_unified_iia.sh`: run the requested one-seed
  unified-IIA comparison inside one allocation-inheriting `srun`, including
  PLOT-bDAS on the per-variable UOT-selected layers.
- `slurm/run_delta_mcqa_pca_mib_4seed.sh`: focused four-A40 rerun of PLOT-PCA,
  PLOT-PCA-DAS, full-vector, and DBM canonical/PCA/SAE for seeds 0--3. It assigns
  one seed per GPU and checks each baseline partition against that seed's PLOT
  artifact.

MCQA bDAS uses the recommended binary-addition settings: batch size 64,
no gradient accumulation, 12 maximum epochs, 5 minimum epochs, training-loss
plateau patience 1, one restart, rotation learning rate
`1e-2`, boundary learning rate `1e-4`, boundary penalty `1.0`, and temperature
annealing from `1.0` to `0.1`.  It calibrates every last-token layer, selects
one layer independently for each abstract variable, and evaluates only those
frozen layer/boundary candidates on test.  Its reported runtime includes the
complete layer training/calibration sweep plus selected test evaluation.

Cluster launchers are in `slurm/`.

### DAS-coordinate UOT proof of concept

`mcqa_das_coordinate_uot.py` trains separate DAS bases for AP at layer index 18
and AT at layer index 24, using dimensions 1152, 768, 576, and 384. It freezes
each basis, measures one-coordinate effects on the fit bank, and solves a
single-row UOT problem per target/dimension/epsilon. Epsilons are 0.5, 1, 2, 4;
the neural KL coefficient is fixed at 0.1. This is supervised selection inside
a DAS representation; it does not rerun layer localization.

Calibration selects powers-of-two handle sizes up to each dimension, plus the
full dimension. Handles fully swap the selected coordinates: there is no lambda,
transport-mass scaling, or post-selection retraining. With a uniform neural
prior, single-row UOT rankings are invariant across epsilon; the runner records
that diagnostic and reuses identical hard-handle calibration results. Exact
ties prefer smaller K, then the listed dimension/epsilon order. Selection uses
pooled IIA and the existing shared-epsilon AP/AT macro-average rule.

All eight checkpoints (materialized bases and full state dictionaries), exact
fit/calibration/test pair banks, incremental signature caches, costs, couplings,
and calibration records are saved. Test evaluation occurs only after freezing
selection: two selected handles plus two full-subspace baselines using the same
selected bases. Every trained basis must pass an exact all-coordinate-versus-DAS
logit check. The launcher `slurm/delta_mcqa_das_coordinate_uot.sbatch` uses one A40
and requires exact partition equality with the preceding seed-0 PCA-UOT run.

### Single-stage native PLOT

`slurm/delta_mcqa_single_stage_native.sbatch` runs seed 0 through
`mcqa_single_stage_native.py`. For each of five block widths (768, 1152,
1536, 1920, 2304), the candidate set pools blocks from all 26 layers.
This gives 78, 52, 52, 52, and 26 candidate sites respectively; nondividing
widths leave smaller final blocks. There is no Stage A or per-layer calibration.

All five effect-signature and cost caches are constructed first. Then each
of four epsilon values solves one two-row coupling per width: exactly 20
couplings. AP and AT calibrate their respective rows from each shared coupling,
using the original native top-k and intervention-strength grids with pooled IIA.
Selection freezes a shared epsilon and each variable's best width/handle before
exactly two final test evaluations. Handles may jointly intervene across layers.
The summary reports all five signature costs and all five coupling/calibration
costs at the selected epsilon, plus selected test evaluation. Full sweep wall
time is reported separately. Cached signatures, couplings, and calibration
outputs support resuming without repeating completed computations.

The launcher uses one A40 and validates the partition against the prior seed-0
artifact. Override `VENV_PATH`, `PRECHECK_STAMP`, `PARTITION_REFERENCE`, and
`RUN_ROOT` as needed. Run the launcher with `--dry-run` under bash to print the
experiment command; submit with `sbatch` from the repository root.

On Delta, first create and validate the isolated NVMe environment with
`bash experiments/mcqa/slurm/setup_delta_mcqa_env.sh` from an active GPU
allocation.  It installs under `/work/nvme/bgvo/$USER/venvs`, keeps all package,
model, dataset, Torch, and temporary caches off the small home quota, checks the
resolved dependency graph, runs the MCQA regression tests, loads the gated Gemma
model and MCQA dataset, downloads and validates all 26 Gemma Scope SAEs, and runs
an encode/decode probe with a real SAE.
The one-seed launcher requires the resulting preflight stamp.

For the Delta A40 allocation used by the MCQA reruns, launch the complete baseline sweep with
`bash experiments/mcqa/slurm/run_delta_mcqa_mib_baselines.sh`. The launcher creates exactly one
`srun` for the complete sweep, uses `/work/nvme/bgvo/$USER/hf_cache`, and resumes existing JSON
outputs within the chosen `RUN_NAME` folder.
Set `VENV_PATH` when using a virtual environment other than `/u/$USER/.venv`; the launcher
checks that its interpreter is Python 3.10 or newer before starting the Slurm step.

Related MCQA experiment folders outside the main-paper path:

- `../mcqa_broad_sweep/`: broad Delta sweep and launcher.
- `../mcqa_layerwise/`: layerwise OT analysis.
- `../mcqa_block_focus/`: OT/DAS block-focus run.
- `../mcqa_diagnostics/`: filter diagnostic notebook.
