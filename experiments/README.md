# Experiment catalog

The current paper experiments are version v4. [Paper provenance](../docs/experiment_history.md)
identifies the exact saved runs behind its tables. Earlier implementations are
preserved in version folders under `archive/`.

## Current implementations and paper workflows

| Folder | Role | Entry points and status |
|---|---|---|
| [common/](common/) | Shared primitives across experiment versions | Runtime, pyvene helpers and variable-width MLP |
| [heq/](heq/README.md) | Main HEQ benchmark | `heq_rerun/` is the paper recipe; older `equality_run.py` and epsilon wrappers remain for previous protocols |
| [binary_addition/](binary_addition/README.md) | Main 4-bit recurrent benchmark | `run_staging_suite.py` and `run_staging_comparison.py`: new one-/two-stage direct PLOT; `run_progressive_plot.py`: original DAS source reused in the table |
| [mcqa/](mcqa/README.md) | Main Gemma-2-2B benchmark and DAS | [BASELINE_DAS.md](mcqa/BASELINE_DAS.md) is the tracked standalone current-code recipe; legacy staged sweeps are retained |
| [mcqa_staging_budget/](mcqa_staging_budget/README.md) | paper runs | `run_focused.py`, corrected DAS evaluation, full-beta replay and ablation launch/collect scripts |
| [ioi/](ioi/README.md) | Earlier supporting benchmark | GPT-2/MIB runs; not a main experiment in the latest manuscript |

The standalone MCQA baseline and shared implementations are committed. The
newer `heq/heq_rerun/` executable package, binary-addition staging runners and
`mcqa_staging_budget/` execution scripts remain local/uncommitted research code
at this cleanup. Their documentation is included to map saved results honestly;
this is not a claim that a GitHub clone contains every paper rerun or checkpoint.
Pre-existing scientific edits were not swept into the organization commit.

## Historical experiment folders

These original files were moved without deleting them. The old paths remain as
relative symlinks; imports and historical commands can keep using them.

| Canonical folder | Purpose | Historical association |
|---|---|---|
| [archive/v1/early_tasks/binary_addition_c1/](archive/v1/early_tasks/binary_addition_c1/) | Fixed-C1 MLP task | Early task development, not the recurrent main benchmark |
| [archive/v1/early_tasks/two_digit_addition/](archive/v1/early_tasks/two_digit_addition/) | Decimal-addition implementation/helpers | Early task development; shared by fixed-C1 code |
| [archive/v2/mcqa_exploration/mcqa_broad_sweep/](archive/v2/mcqa_exploration/mcqa_broad_sweep/) | Earlier broad cluster sweeps | Previous MCQA protocols, including family-weighted selection |
| [archive/v2/mcqa_exploration/mcqa_block_focus/](archive/v2/mcqa_exploration/mcqa_block_focus/) | Block-focused OT/DAS search | Earlier site/grid exploration |
| [archive/v2/mcqa_exploration/mcqa_layerwise/](archive/v2/mcqa_exploration/mcqa_layerwise/) | Layerwise OT analysis | Earlier localization exploration |
| [archive/v2/mcqa_exploration/mcqa_diagnostics/](archive/v2/mcqa_exploration/mcqa_diagnostics/) | Filtering diagnostic notebook | Historical diagnostic, not a paper result source |
| [archive/v1/demos/notebook_demos/](archive/v1/demos/notebook_demos/) | Original DAS/addition notebooks | Teaching and early explorations |
| [archive/v3/figure_utilities/heq_intervention_heatmaps/](archive/v3/figure_utilities/heq_intervention_heatmaps/) | Earlier HEQ heatmap utility | Old result layout, not the rerun figure path |

Archival scripts preserve their original scientific settings. Their old
selection objectives may no longer be accepted by the evolving current shared
library; use an appropriate historical commit/environment for exact replay.

## Studies inside current benchmark packages

- HEQ: the stable six-epsilon rerun supersedes the original five-epsilon PLOT
  comparison; saved DAS is reused. Broad epsilon robustness is a separate ablation.
- Binary addition: the current table uses unrestricted width-16 staging and reused
  DAS, seeds 0-4. Capped runs, old selected-epsilon timings, width-8 studies, the
  8-bit pilot, residual-MLP pilot and variable/cross-length extensions remain
  distinct studies. Their original scripts are preserved in `binary_addition/`.
- MCQA: `mcqa_run_cloud.py` is a general serial launcher, not a frozen snapshot
  of every five-seed paper run. `mcqa_delta_hierarchical_sweep.py` and its cluster
  launchers preserve older staged workflows. The single-stage, token-position and
  DAS-coordinate runners support specific ablations/pilots; consult the results
  index before choosing a command.

Exact result classification and each old-to-new path are in
[experiment_inventory.json](../docs/experiment_inventory.json).
