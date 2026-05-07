# Push Checklist

Before pushing to `jchang153/causal-abstractions-ot`:

1. Push the folder as a new root-level directory named `binary_addition_h8_ot/`.
2. Keep the folder contents unchanged on the first push.
3. Verify these files are present after the copy:
   - `README.md`
   - `METHOD_COMPARISON.md`
   - `BASELINE_COMPARISON.md`
   - `DESTINATION_LAYOUT.md`
   - `PUSH_CHECKLIST.md`
   - `regular_shared_ot/README.md`
   - `anchored_ot/README.md`
   - `das/README.md`
   - `mib_baselines/README.md`
   - `scripts/run_h8_regular_shared_ot.py`
   - `scripts/run_h8_anchored_ot.py`
   - `scripts/run_h8_anchorprefix_das.py`
   - `scripts/run_h8_mib_baselines.py`
4. Run a quick syntax check inside the pushed folder.
5. Optionally run one smoke command per method track and compare against the compact summary JSONs.

Recommended first commit scope:
- only the new `binary_addition_h8_ot/` directory
- no unrelated README rewrites
- no refactors into the older MLP workflow yet
