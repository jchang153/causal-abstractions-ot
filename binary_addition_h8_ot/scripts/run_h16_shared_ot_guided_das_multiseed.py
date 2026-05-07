from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "binary_addition_rnn" / "run_shared_ot_guided_das_multiseed.py"
OUT_DIR = ROOT / "results" / "shared_ot_guided_das_h16_multiseed"
CHECKPOINT_MAP = ";".join(
    [
        f"0={ROOT / 'results' / 'checkpoints' / 'gru_h16_seed0.pt'}",
        f"1={ROOT / 'results' / 'checkpoints' / 'gru_h16_seed1.pt'}",
        f"2={ROOT / 'results' / 'checkpoints' / 'gru_h16_seed2.pt'}",
    ]
)

cmd = [
    sys.executable,
    str(SCRIPT),
    "--out-dir", str(OUT_DIR),
    "--seeds", "0,1,2",
    "--checkpoint-map", CHECKPOINT_MAP,
    "--abstract-mode", "all_endogenous",
    "--hidden-size", "16",
    "--device", "cpu",
    "--fit-bases", "128",
    "--calib-bases", "64",
    "--test-bases", "64",
    "--train-on", "all",
    "--source-policy", "structured_26_top3carry_c2x5_c3x7_no_random",
    "--rows", "C1,C2,C3",
    "--discovery-resolutions", "16,8,4,2,1",
    "--ot-epsilons", "0.01,0.03",
    "--top-k-grid", "1,2",
    "--ot-lambda-grid", "0.5,1,2",
    "--selection-profiles", "combined:0.0",
    "--support-relative-threshold", "0.98",
    "--support-max-trials", "12",
    "--mask-thresholds", "0.8,0.9",
    "--mask-modes", "StepMask,S80,S90",
    "--guided-selection-rule", "combined",
    "--guided-invariance-floor", "0.0",
    "--guided-lambda-grid", "0.25,0.5,1,2,4,8",
    "--das-subspace-dims", "1,2,4",
    "--das-lrs", "0.01,0.003",
    "--das-resolutions", "16,8,4,2,1",
]
subprocess.run(cmd, check=True, cwd=ROOT)
