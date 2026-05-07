from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run shared-OT-guided DAS vs full DAS over multiple seeds.")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--checkpoint-map", type=str, required=True)
    ap.add_argument("--abstract-mode", type=str, default="all_endogenous", choices=["carries_only", "all_endogenous"])
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--hidden-size", type=int, default=8)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--fit-bases", type=int, default=128)
    ap.add_argument("--calib-bases", type=int, default=64)
    ap.add_argument("--test-bases", type=int, default=64)
    ap.add_argument("--train-on", type=str, default="all", choices=["all", "fit_only"])
    ap.add_argument("--source-policy", type=str, default="structured_26_top3carry_c2x5_c3x7_no_random")
    ap.add_argument("--rows", type=str, default="C1,C2,C3")
    ap.add_argument("--discovery-resolutions", type=str, default="8,4,2,1")
    ap.add_argument("--ot-epsilons", type=str, default="0.01,0.03")
    ap.add_argument("--top-k-grid", type=str, default="1,2")
    ap.add_argument("--ot-lambda-grid", type=str, default="0.5,1,2")
    ap.add_argument("--selection-profiles", type=str, default="combined:0.0")
    ap.add_argument("--support-relative-threshold", type=float, default=0.98)
    ap.add_argument("--support-max-trials", type=int, default=12)
    ap.add_argument("--mask-thresholds", type=str, default="0.8,0.9")
    ap.add_argument("--mask-modes", type=str, default="StepMask,S80,S90")
    ap.add_argument("--guided-selection-rule", type=str, default="combined")
    ap.add_argument("--guided-invariance-floor", type=float, default=0.0)
    ap.add_argument("--guided-lambda-grid", type=str, default="0.25,0.5,1,2,4,8")
    ap.add_argument("--das-subspace-dims", type=str, default="1,2,4")
    ap.add_argument("--das-lrs", type=str, default="0.01,0.003")
    ap.add_argument("--das-resolutions", type=str, default="8,4,2,1")
    return ap.parse_args()


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(text).split(",") if x.strip())


def _parse_checkpoint_map(text: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for chunk in str(text).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, value = chunk.split("=", 1)
        mapping[int(key.strip())] = value.strip()
    return mapping


def _run(cmd: list[str]) -> float:
    start = time.perf_counter()
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return time.perf_counter() - start


def _internal_metrics(per_row: dict[str, object], rows: tuple[str, ...]) -> dict[str, object]:
    values = [float(per_row[row_key]["test"]["combined"]) for row_key in rows if row_key in per_row]
    return {
        "mean_combined": 0.0 if not values else float(sum(values) / len(values)),
        "per_row": {
            row_key: per_row[row_key]["test"]
            for row_key in rows
            if row_key in per_row
        },
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_csv_ints(args.seeds)
    requested_rows = tuple(x.strip() for x in args.rows.split(",") if x.strip())
    checkpoint_map = _parse_checkpoint_map(args.checkpoint_map)

    per_seed: dict[str, object] = {}
    hybrid_totals = []
    full_das_totals = []
    guided_internal = []
    guided_c3 = []
    full_internal = []
    full_c3 = []

    for seed in seeds:
        if seed not in checkpoint_map:
            raise ValueError(f"missing checkpoint for seed {seed}")
        seed_dir = out_dir / f"seed_{seed}"
        discovery_dir = seed_dir / "discovery"
        guided_dir = seed_dir / "guided_das"
        full_das_dir = seed_dir / "full_das"
        checkpoint = checkpoint_map[seed]

        discovery_cmd = [
            sys.executable,
            str(ROOT / "experiments" / "binary_addition_rnn" / "run_shared_ot_discovery.py"),
            "--out-dir", str(discovery_dir),
            "--abstract-mode", str(args.abstract_mode),
            "--width", str(args.width),
            "--hidden-size", str(args.hidden_size),
            "--seed", str(seed),
            "--device", str(args.device),
            "--fit-bases", str(args.fit_bases),
            "--calib-bases", str(args.calib_bases),
            "--test-bases", str(args.test_bases),
            "--train-on", str(args.train_on),
            "--model-checkpoint", str(checkpoint),
            "--source-policy", str(args.source_policy),
            "--normalize-signatures",
            "--fit-signature-mode", "all",
            "--fit-family-profile", "all",
            "--fit-stratify-mode", "none",
            "--cost-metric", "sq_l2",
            "--resolutions", str(args.discovery_resolutions),
            "--ot-epsilons", str(args.ot_epsilons),
            "--top-k-grid", str(args.top_k_grid),
            "--lambda-grid", str(args.ot_lambda_grid),
            "--selection-profiles", str(args.selection_profiles),
        ]
        discovery_seconds = _run(discovery_cmd)

        support_path = discovery_dir / "effective_support.json"
        support_cmd = [
            sys.executable,
            str(ROOT / "experiments" / "binary_addition_rnn" / "run_extract_effective_support.py"),
            "--discovery-summary", str(discovery_dir / "shared_ot_discovery_summary.json"),
            "--out-path", str(support_path),
            "--rows", str(args.rows),
            "--relative-threshold", str(args.support_relative_threshold),
            "--max-trials", str(args.support_max_trials),
            "--mask-thresholds", str(args.mask_thresholds),
        ]
        support_seconds = _run(support_cmd)

        guided_cmd = [
            sys.executable,
            str(ROOT / "experiments" / "binary_addition_rnn" / "run_transport_guided_das.py"),
            "--support-summary", str(support_path),
            "--out-dir", str(guided_dir),
            "--abstract-mode", str(args.abstract_mode),
            "--width", str(args.width),
            "--hidden-size", str(args.hidden_size),
            "--seed", str(seed),
            "--device", str(args.device),
            "--fit-bases", str(args.fit_bases),
            "--calib-bases", str(args.calib_bases),
            "--test-bases", str(args.test_bases),
            "--train-on", str(args.train_on),
            "--model-checkpoint", str(checkpoint),
            "--source-policy", str(args.source_policy),
            "--fit-bank-mode", "anchored_prefix",
            "--rows", str(args.rows),
            "--mask-modes", str(args.mask_modes),
            "--selection-rule", str(args.guided_selection_rule),
            "--invariance-floor", str(args.guided_invariance_floor),
            "--lambda-grid", str(args.guided_lambda_grid),
            "--das-subspace-dims", str(args.das_subspace_dims),
            "--das-lrs", str(args.das_lrs),
        ]
        guided_seconds = _run(guided_cmd)

        full_das_cmd = [
            sys.executable,
            str(ROOT / "experiments" / "binary_addition_rnn" / "run_joint_endogenous_das.py"),
            "--out-dir", str(full_das_dir),
            "--abstract-mode", str(args.abstract_mode),
            "--width", str(args.width),
            "--hidden-size", str(args.hidden_size),
            "--resolutions", str(args.das_resolutions),
            "--seed", str(seed),
            "--device", str(args.device),
            "--fit-bases", str(args.fit_bases),
            "--calib-bases", str(args.calib_bases),
            "--test-bases", str(args.test_bases),
            "--train-on", str(args.train_on),
            "--model-checkpoint", str(checkpoint),
            "--source-policy", str(args.source_policy),
            "--fit-bank-mode", "anchored_prefix",
            "--rows", str(args.rows),
            "--selection-rule", str(args.guided_selection_rule),
            "--invariance-floor", str(args.guided_invariance_floor),
            "--lambda-grid", str(args.guided_lambda_grid),
            "--das-subspace-dims", str(args.das_subspace_dims),
            "--das-lrs", str(args.das_lrs),
        ]
        full_das_seconds = _run(full_das_cmd)

        support_summary = json.loads((discovery_dir / "effective_support.json").read_text(encoding="utf-8"))
        guided_summary = json.loads((guided_dir / "transport_guided_das_summary.json").read_text(encoding="utf-8"))
        full_das_summary = json.loads((full_das_dir / "joint_endogenous_das_summary.json").read_text(encoding="utf-8"))

        best_mask_mode = guided_summary["best_mask_mode"]
        guided_best = guided_summary["best_result"]
        full_best = full_das_summary["best_result"]

        guided_internal_metrics = _internal_metrics(guided_best["per_row"], requested_rows)
        full_internal_metrics = _internal_metrics(full_best["test"]["per_row"], requested_rows)

        hybrid_total = float(discovery_seconds + support_seconds + guided_seconds)
        hybrid_totals.append(hybrid_total)
        full_das_totals.append(float(full_das_seconds))
        guided_internal.append(float(guided_internal_metrics["mean_combined"]))
        guided_c3.append(float(guided_best["per_row"]["C3"]["test"]["combined"]))
        full_internal.append(float(full_internal_metrics["mean_combined"]))
        full_c3.append(float(full_best["test"]["per_row"]["C3"]["test"]["combined"]))

        per_seed[str(seed)] = {
            "checkpoint": str(checkpoint),
            "shared_ot_support": {
                row_key: {
                    "dominant_hidden_timestep": support_summary["row_support"][row_key]["dominant_hidden_timestep"],
                    "dominant_hidden_timestep_stability": support_summary["row_support"][row_key]["dominant_hidden_timestep_stability"],
                    "masks": support_summary["row_support"][row_key]["masks"],
                    "compression": support_summary["row_support"][row_key]["compression"],
                }
                for row_key in requested_rows
            },
            "guided_das": {
                "best_mask_mode": str(best_mask_mode),
                "best_result": guided_best,
                "internal_carry_mean": float(guided_internal_metrics["mean_combined"]),
                "C3": float(guided_best["per_row"]["C3"]["test"]["combined"]),
            },
            "full_das": {
                "best_resolution": int(full_best["resolution"]),
                "best_result": full_best,
                "internal_carry_mean": float(full_internal_metrics["mean_combined"]),
                "C3": float(full_best["test"]["per_row"]["C3"]["test"]["combined"]),
            },
            "runtimes_seconds": {
                "discovery": float(discovery_seconds),
                "support": float(support_seconds),
                "guided_das": float(guided_seconds),
                "hybrid_total": float(hybrid_total),
                "full_das_total": float(full_das_seconds),
                "speedup_vs_full_das": float(full_das_seconds / hybrid_total) if hybrid_total > 0 else None,
            },
        }

    def _mean(values: list[float]) -> float:
        return 0.0 if not values else float(sum(values) / len(values))

    summary = {
        "config": vars(args),
        "per_seed": per_seed,
        "aggregate": {
            "guided_das": {
                "internal_carry_mean": _mean(guided_internal),
                "C3": _mean(guided_c3),
            },
            "full_das": {
                "internal_carry_mean": _mean(full_internal),
                "C3": _mean(full_c3),
            },
            "runtimes_seconds": {
                "hybrid_total_mean": _mean(hybrid_totals),
                "full_das_total_mean": _mean(full_das_totals),
                "mean_speedup_vs_full_das": float(_mean(full_das_totals) / _mean(hybrid_totals)) if _mean(hybrid_totals) > 0 else None,
            },
        },
    }

    summary_path = out_dir / "shared_ot_guided_das_multiseed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "seeds": list(seeds)}, indent=2))


if __name__ == "__main__":
    main()
