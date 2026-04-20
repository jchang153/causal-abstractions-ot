from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "binary_addition_rnn" / "run_joint_endogenous_das.py"
OUT_DIR = ROOT / "results" / "anchorprefix_das_h8_run"
CHECKPOINT = ROOT / "results" / "checkpoints" / "gru_h8_seed0.pt"

cmd = [
    sys.executable,
    str(SCRIPT),
    "--out-dir", str(OUT_DIR),
    "--abstract-mode", "all_endogenous",
    "--hidden-size", "8",
    "--seed", "0",
    "--fit-bases", "128",
    "--calib-bases", "64",
    "--test-bases", "64",
    "--model-checkpoint", str(CHECKPOINT),
    "--source-policy", "structured_26_top3carry_c2x5_c3x7_no_random",
    "--selection-rule", "combined",
    "--fit-bank-mode", "anchored_prefix",
]
subprocess.run(cmd, check=True, cwd=ROOT)
