from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "binary_addition_rnn" / "run_joint_endogenous_resolution_sweep.py"
OUT_DIR = ROOT / "results" / "regular_shared_ot_h8_run"
CHECKPOINT = ROOT / "results" / "checkpoints" / "gru_h8_seed0.pt"

cmd = [
    sys.executable,
    str(SCRIPT),
    "--out-dir", str(OUT_DIR),
    "--abstract-mode", "all_endogenous",
    "--hidden-size", "8",
    "--resolutions", "8,4,2,1",
    "--methods", "ot",
    "--seed", "0",
    "--fit-bases", "128",
    "--calib-bases", "64",
    "--test-bases", "64",
    "--model-checkpoint", str(CHECKPOINT),
    "--ot-epsilons", "0.003,0.01,0.03,0.1,0.3",
    "--top-k-grid", "1,2,4,8",
    "--lambda-grid", "0.25,0.5,1,2,4,8",
    "--selection-profiles", "combined:0.0;sensitivity_only:0.0",
    "--source-policy", "structured_26_top3carry_c2x5_c3x7_no_random",
    "--normalize-signatures",
    "--fit-signature-mode", "all",
    "--fit-family-profile", "all",
    "--cost-metric", "sq_l2",
]
subprocess.run(cmd, check=True, cwd=ROOT)
