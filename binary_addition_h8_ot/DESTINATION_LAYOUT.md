# Destination Layout For `causal-abstractions-ot`

Recommended destination in the student repo:

```text
causal-abstractions-ot/
  binary_addition_h8_ot/
    README.md
    METHOD_COMPARISON.md
    DESTINATION_LAYOUT.md
    PUSH_CHECKLIST.md
    requirements_min.txt
    regular_shared_ot/
    anchored_ot/
    scripts/
    experiments/
    results/
```

## Why this root-level folder is the cleanest push target

The current student repo already has a stable flat workflow at the root:
- `addition_train.py`
- `addition_compare.py`
- `addition_seed_sweep.py`
- `addition_experiment/`

This exported benchmark is materially different:
- recurrent GRU instead of the current MLP workflow
- bitwise Bernoulli outputs instead of the current classification target
- separate shared-bank and anchored-bank OT runners

So the least disruptive integration is to add it as **one new root-level folder** rather than trying to merge it into the existing `addition_experiment/` package immediately.

## Push recommendation

Rename the bundle folder on push to:
- `binary_addition_h8_ot/`

That preserves the internal relative paths used by the wrapper scripts.

## What to copy

Copy the entire local folder:
- `C:\Users\zgzg1\Projects\GW DAS\share\binary_addition_h8_ot_bundle`

into the target repo as:
- `binary_addition_h8_ot/`

## What not to change during the first push

Do not rewrite imports or fold this into the current top-level addition scripts yet.
The present goal should be:
- preserve the exact runnable package
- preserve the exact result summaries
- preserve the paper-facing documentation

Once it is in the student repo, a later cleanup pass can decide whether to:
- merge common utilities into `addition_experiment/`
- add seed-sweep wrappers
- or wire it into the root README

## Minimal post-push validation

From the repo root:

```bash
cd binary_addition_h8_ot
python -m py_compile experiments/binary_addition_rnn/*.py scripts/*.py
```

Then, if desired, run one smoke benchmark:

```bash
python scripts/run_h8_regular_shared_ot.py
```

and compare the produced summary against:
- `regular_shared_ot/best_result_compact.json`

## Optional follow-up after push

If you want a repo-native integration later, the next clean step is to add a short section to the repo root `README.md`:
- name the new folder
- explain that it is the recurrent `h=8` binary-addition OT package
- point readers to `binary_addition_h8_ot/README.md`
