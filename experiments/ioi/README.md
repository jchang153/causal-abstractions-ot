# Blind IOI Head Localization with PLOT-DAS

This folder implements the Indirect Object Identification causal-variable track
from MIB using GPT-2 Small and the released `mib-bench/ioi` Hugging Face dataset.
It compares three localization regimes while keeping the DAS objective and
high-level model fixed:

- **Blind DAS:** joint DAS over all $12 \times 12 = 144$ attention heads.
- **PLOT-DAS:** a $2 \times 144$ one-sided UOT coupling localizes heads, followed
  by joint DAS on each row's top $K \in \{1,2,3\}$ heads.
- **Oracle DAS:** joint DAS over the known IOI heads $7.3, 7.9, 8.6, 8.10$.

Every intervention acts on a head's $64$-dimensional attention-weighted value
output immediately before GPT-2's attention output projection $W_O$, at all
token positions. Each selected DAS head has an independent orthogonal
$64 \times 32$ subspace, shared across token positions. The head subspaces are
optimized jointly for a given method and abstract variable.

## Data protocol

The runner uses these released counterfactual families:

- `s1_io_flip`
- `s2_io_flip`
- `s1_ioi_flip_s2_ioi_flip`

The runner pins dataset revision `5024626`: the later `main` revision reduced
the public test file to $1{,}000$ examples and cannot support this protocol.
The reference run validates that both selected files contain $10{,}000$ rows.
All released training rows form the fit split. Public test rows are shuffled
with seed $0$ *before filtering*: $2{,}000$ raw rows become calibration and
the other $8{,}000$ become held-out test. MIB's greedy one-token base/source
correctness filter is applied independently within every family and split. The
no-change regression bank is independently filtered as well.

The fitted and then frozen high-level model is

\[
\Delta_H=\beta_0+\beta_{\mathrm{Pos}}S_{\mathrm{Pos}}+
\beta_{\mathrm{Tok}}S_{\mathrm{Tok}}.
\]

It is fit using the no-change condition and full-vector patching of all four
oracle heads for the three counterfactual conditions. The runner records
$R^2$, counts, and differences from MIB's reported
$(0.048, 2.005, 0.768)$. A coefficient difference above $0.15$ stops the run
unless `--allow-coefficient-mismatch` is supplied.

## Install and run

Install the repository lock, including the MIB-compatible `pyvene==0.1.8`:

```bash
pip install -r requirements.txt
```

Run the complete experiment on a GPU:

```bash
python experiments/ioi/run.py \
  --device cuda \
  --methods blind,plot,oracle \
  --microbatch-size 8 \
  --output-dir results/ioi/gpt2_seed0
```

The effective batch size remains $1024$; reduce `--microbatch-size` if GPU
memory is limited. Useful CLI controls include `--split-seed`,
`--signature-bank-size`, `--uot-epsilons`, `--uot-beta-neural`, `--plot-k`, and
`--das-dimension`. `--quick-rows` is only for development and will not reproduce
the benchmark coefficients.

At startup the runner validates the locked Torch, Transformers, and pyvene
versions. `--allow-version-mismatch` is available for explicitly non-reference
smoke runs, and the mismatch is recorded in `manifest.json`.

The default full run is computationally large: blind DAS installs 144 trainable
head rotations and PLOT signature collection performs full-vector interventions
for every head. Re-running the same command and output directory resumes from
compatible artifacts. A changed configuration hash is rejected unless a new
output directory or `--force` is used.

## Selection and artifacts

PLOT uses a deterministic $1{,}000$-raw-row training subset for signatures.
For each UOT setting, full-vector top-$K$ interventions are calibrated first.
One UOT setting is selected by the macro-average of the two rows' independently
best calibration MSE. After freezing the coupling, six DAS candidates are
trained and one $K$ is selected per row, with ties going to smaller $K$.
Blind and oracle DAS do not use calibration for model selection.

The output directory contains:

- `banks.json` and `causal_model.json`;
- `plot/signatures.json`, `plot/costs.json`, and `plot/uot_calibration.json`;
- all blind, oracle, and PLOT-DAS candidate checkpoints under `checkpoints/`;
- `manifest.json`, `summary.json`, and a concise `summary.txt`.

`summary.json` reports per-family and macro MSE, selected heads and $K$, full
coupling rows, training curves, trainable parameter counts, and cold/amortized
runtime components. Held-out evaluation occurs only after every selection for a
method has been frozen.

## Tests

Run the offline suite with:

```bash
PYTHONPATH=. pytest -q experiments/ioi/test_ioi.py
```

An opt-in tiny GPT-2/network smoke test is available with:

```bash
IOI_NETWORK_SMOKE=1 PYTHONPATH=. pytest -q \
  experiments/ioi/test_ioi.py -k opt_in_gpt2_smoke
```
