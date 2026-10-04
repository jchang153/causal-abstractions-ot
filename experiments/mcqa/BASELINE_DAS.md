# Run the MCQA baseline DAS

Run from the repository root using Python 3.10+ and a CUDA GPU. Install the
repository requirements and set HF_TOKEN with access to google/gemma-2-2b.

```bash
pip install -r requirements.txt
bash experiments/mcqa/run_baseline_das.sh
```

The example runs answer_token (AT) at exactly 128 dimensions over all 26
last-token residual layers, selects a layer on calibration, and evaluates only
the selected handle on test. This dimension comparison is not the paper's full
sweep and does not imply a particular accuracy.

```bash
# Fixed zero-based layer:
LAYERS=24 bash experiments/mcqa/run_baseline_das.sh
# Both variables over the baseline grid:
TARGET_VARS=answer_pointer,answer_token DAS_SUBSPACE_DIMS=32,64,96,128,256,512,768,1024,1536,2048,2304 bash experiments/mcqa/run_baseline_das.sh
```

Use a fresh RESULTS_TIMESTAMP for each run. SPLIT_SEED and TRAINING_SEED default
to zero and can be overridden independently. RESULTS_ROOT defaults to results.

## Compare preprocessing and scoring

- Model: google/gemma-2-2b (base), eager attention, float16 on CUDA, left padding
  and explicit position IDs.
- Dataset: jchang153/copycolors_mcqa, no config. This matches the serial launcher;
  the lower-level data module defaults to mib-bench/copycolors_mcqa with config
  4_answer_choices. Match the dataset explicitly when comparing implementations.
- Load the first 2,000 raw rows per available dataset split before filtering.
  Pool answerPosition, randomLetter and answerPosition_randomLetter families.
- Retain pairs where both base and source answers pass factual filtering. This
  preserves the legacy token-variant or bidirectional-substring factual checker.
  Set factual filtering batch size to 64 independently of intervention batching.
- Use base_group_disjoint_v1: variants of one factual row stay in one partition.
  Construct both AP and AT banks, even when running AT alone, with 200 shared
  training pairs and 200 calibration plus 200 test pairs per variable.
  Calibration/test use target-sensitive pairs. Compare saved partition digests;
  the same split seed does not ensure the same cohort if factual filtering differs.
- Train the rotated subspace with full-vocabulary token cross-entropy. The example
  uses batch 64, learning rate 0.001, one restart, 100 maximum epochs, five minimum
  epochs, plateau patience one and relative loss threshold 0.001.
- Report iia_acc: actual full-vocabulary top-1 next token, decoded, NFKC-normalized,
  stripped and case-normalized, matching exactly one ASCII A-Z causal answer.
  Alphabet-restricted argmax and substring checks are different metrics. The MIB
  substring helper is an optional diagnostic and is not used for DAS selection.
- Select on pooled calibration IIA across examples; evaluate the frozen winner
  on test. --calibration-only saves the selected checkpoint without touching test.

These are comparison points, not a diagnosis of another implementation's score.
A 128-dimensional result alone does not establish a preprocessing discrepancy.

Implementation: mcqa_experiment/das.py and intervention.py. Preprocessing:
mcqa_experiment/data.py. Loss/scoring: mcqa_experiment/metrics.py and checking.py.

Existing CPU regression checks (no model download):

```bash
PYTHONPATH=.:experiments/mcqa python -m pytest -q experiments/mcqa/test_unified_iia.py experiments/mcqa/test_data_partition.py
```

Full GPU training and dataset/model access must be validated separately.
