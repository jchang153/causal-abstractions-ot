# MCQA

Main-paper MCQA code lives here. The benchmark evaluates Gemma-2-2B on CopyColors-style multiple-choice prompts and localizes the abstract variables `answer_pointer` and `answer_token`.

## Evaluation metric

Every calibration, selection, and test verdict uses `iia_acc`: Gemma's actual
full-vocabulary top-1 next token is decoded, NFKC-normalized, stripped, folded
to uppercase, and accepted only when it is exactly one ASCII symbol A-Z matching
the causal interchange's final expected answer. Alphabet-restricted and raw
token-ID accuracies are diagnostics only. OT/UOT/cosine effect signatures remain
projected onto the 26 answer symbols; brute-force evaluates candidate
interventions directly with `iia_acc`.

Method families:

- `PLOT`: Stage A UOT layer localization.
- `PLOT-native`: Stage A plus native-coordinate Stage B handles.
- `PLOT-PCA`: Stage A plus PCA-basis Stage B handles.
- `PLOT-DAS`: DAS restricted to the Stage A layer.
- `PLOT-native-DAS`: DAS guided by the native Stage B support.
- `PLOT-PCA-DAS`: DAS guided by the PCA Stage B support.
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
Paper runtime charges every resolution/config evaluated at the selected epsilon
plus the final selected test evaluations; it does not charge only the
retrospective winning resolution, nor the sweep over unselected epsilons.
- `mcqa_plot_layer.py`, `mcqa_plot_native_support.py`, `mcqa_ot_pca_focus.py`, `mcqa_plot_das_layer.py`, `mcqa_plot_das_native_support.py`: individual stage runners.
- `mcqa_dbm_baselines.py`: resumable DBM/full-layer sweep. It selects layers using calibration accuracy and evaluates only the frozen selected layer on test.
- `slurm/submit_delta_mcqa_boundless_das.sh`: submit the three-seed bDAS array on Delta.
- `slurm/submit_delta_mcqa_corrected_reruns.sh`: submit the three-seed unified-IIA
  reruns for PLOT, PLOT-guided DAS, Full DAS, MIB, and bDAS.
- `slurm/run_delta_mcqa_one_seed_unified_iia.sh`: run the requested one-seed
  unified-IIA comparison inside one allocation-inheriting `srun`, including
  PLOT-bDAS on the per-variable UOT-selected layers.

MCQA bDAS uses the recommended binary-addition settings: batch size 64,
no gradient accumulation, 12 epochs, one restart, rotation learning rate
`1e-2`, boundary learning rate `1e-4`, boundary penalty `1.0`, and temperature
annealing from `1.0` to `0.1`.  It calibrates every last-token layer, selects
one layer independently for each abstract variable, and evaluates only those
frozen layer/boundary candidates on test.  Its reported runtime includes the
complete layer training/calibration sweep plus selected test evaluation.

Cluster launchers are in `slurm/`.

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
