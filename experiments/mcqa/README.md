# MCQA

Main-paper MCQA code lives here. The benchmark evaluates Gemma-2-2B on CopyColors-style multiple-choice prompts and localizes the abstract variables `answer_pointer` and `answer_token`.

Method families:

- `PLOT`: Stage A UOT layer localization.
- `PLOT-native`: Stage A plus native-coordinate Stage B handles.
- `PLOT-PCA`: Stage A plus PCA-basis Stage B handles.
- `PLOT-DAS`: DAS restricted to the Stage A layer.
- `PLOT-native-DAS`: DAS guided by the native Stage B support.
- `PLOT-PCA-DAS`: DAS guided by the PCA Stage B support.
- `Full DAS`: DAS over all layers and the full subspace grid.
- `DBM`: MIB-style desiderata-based masking in the canonical, PCA, or Gemma Scope SAE basis.
- `Full layer`: the pre-DAS control that swaps the entire last-token residual vector at one layer.

Main entry points:

- `mcqa_delta_hierarchical_sweep.py`: serial staged paper sweep.
- `mcqa_delta_hierarchical_parallel.py`: staged task planner/aggregator for cluster runs.
- `mcqa_run_cloud.py`: configurable single-run launcher, including full DAS.
- `mcqa_paper_runtime.py`: paper-runtime summarizer.
- `mcqa_plot_layer.py`, `mcqa_plot_native_support.py`, `mcqa_ot_pca_focus.py`, `mcqa_plot_das_layer.py`, `mcqa_plot_das_native_support.py`: individual stage runners.
- `mcqa_dbm_baselines.py`: resumable DBM/full-layer sweep. It selects layers using calibration accuracy and preserves all per-layer test results.

Cluster launchers are in `slurm/`.

For the Delta A40 allocation used by the MCQA reruns, launch the complete baseline sweep with
`bash experiments/mcqa/slurm/run_delta_mcqa_mib_baselines.sh`. The launcher creates exactly one
`srun` for the complete sweep, uses `/work/nvme/bgvo/$USER/hf_cache`, and resumes existing JSON
outputs within the chosen `RUN_NAME` folder.

Related MCQA experiment folders outside the main-paper path:

- `../mcqa_broad_sweep/`: broad Delta sweep and launcher.
- `../mcqa_layerwise/`: layerwise OT analysis.
- `../mcqa_block_focus/`: OT/DAS block-focus run.
- `../mcqa_diagnostics/`: filter diagnostic notebook.
