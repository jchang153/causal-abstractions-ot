from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from experiments.binary_addition_rnn.support import coord_group_mask, full_timestep_mask, parse_site_key


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract effective row-dominant OT support for transport-guided DAS.")
    ap.add_argument("--discovery-summary", type=str, required=True)
    ap.add_argument("--out-path", type=str, required=True)
    ap.add_argument("--rows", type=str, default="C1,C2,C3")
    ap.add_argument("--relative-threshold", type=float, default=0.98)
    ap.add_argument("--max-trials", type=int, default=12)
    ap.add_argument("--mask-thresholds", type=str, default="0.8,0.9")
    return ap.parse_args()


def _parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def _parse_rows(text: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in text.split(",") if x.strip())


def _row_normalize(coupling: torch.Tensor) -> torch.Tensor:
    return coupling / coupling.sum(dim=1, keepdim=True).clamp_min(1e-30)


def _trial_units(summary: dict[str, object], *, row_key: str) -> list[dict[str, object]]:
    units: list[dict[str, object]] = []
    for resolution_block in summary["per_resolution"]:
        sites = tuple(resolution_block["sites"])
        for trial_idx, trial in enumerate(resolution_block["trials"]):
            for profile_key, profile_result in trial["profile_results"].items():
                row_result = profile_result["per_row"][row_key]
                units.append(
                    {
                        "resolution": int(resolution_block["resolution"]),
                        "trial_index": int(trial_idx),
                        "epsilon": float(trial["config"]["epsilon"]),
                        "profile_key": str(profile_key),
                        "sites": sites,
                        "coupling": torch.tensor(trial["coupling"], dtype=torch.float32),
                        "row_score": float(row_result["calibration"]["combined"]),
                        "row_calibration": row_result["calibration"],
                        "row_selected": row_result,
                    }
                )
    return units


def _retain_trial_units(units: list[dict[str, object]], *, relative_threshold: float, max_trials: int) -> list[dict[str, object]]:
    if not units:
        return []
    best_score = max(float(unit["row_score"]) for unit in units)
    retained = [unit for unit in units if float(unit["row_score"]) >= float(relative_threshold) * float(best_score)]
    retained.sort(key=lambda unit: (-float(unit["row_score"]), int(unit["resolution"]), float(unit["epsilon"])))
    return retained[: int(max_trials)]


def _coord_mask_from_threshold(coord_evidence: list[float], *, threshold: float) -> tuple[int, ...]:
    total = float(sum(coord_evidence))
    if total <= 0:
        return tuple(range(len(coord_evidence)))
    ranked = sorted(range(len(coord_evidence)), key=lambda idx: (-float(coord_evidence[idx]), int(idx)))
    running = 0.0
    chosen: list[int] = []
    for idx in ranked:
        chosen.append(int(idx))
        running += float(coord_evidence[idx])
        if running / total >= float(threshold):
            break
    return tuple(sorted(chosen))


def main() -> None:
    args = parse_args()
    summary_path = Path(args.discovery_summary).resolve()
    out_path = Path(args.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    hidden_size = int(summary["config"]["hidden_size"])
    row_keys = tuple(summary["row_keys"])
    requested_rows = _parse_rows(args.rows)
    mask_thresholds = _parse_floats(args.mask_thresholds)
    row_index = {row_key: idx for idx, row_key in enumerate(row_keys)}

    extracted_rows = {}
    for row_key in requested_rows:
        units = _trial_units(summary, row_key=row_key)
        retained = _retain_trial_units(
            units,
            relative_threshold=float(args.relative_threshold),
            max_trials=int(args.max_trials),
        )
        if not retained:
            continue
        weight_sum = sum(float(unit["row_score"]) for unit in retained)
        if weight_sum <= 0:
            weights = [1.0 / len(retained) for _ in retained]
        else:
            weights = [float(unit["row_score"]) / weight_sum for unit in retained]

        hidden_coord_evidence = {
            timestep: [0.0 for _ in range(hidden_size)]
            for timestep in range(int(summary["config"]["width"]))
        }
        logit_evidence: dict[str, float] = {}
        retained_trial_meta = []
        dominant_steps = []

        for unit, weight in zip(retained, weights):
            coupling = _row_normalize(unit["coupling"])
            row_idx = int(row_index[row_key])
            row_mass = coupling[row_idx]
            if coupling.size(0) > 1:
                other_rows = torch.cat([coupling[:row_idx], coupling[row_idx + 1 :]], dim=0)
                other_max = other_rows.max(dim=0).values
            else:
                other_max = torch.zeros_like(row_mass)
            dominant = torch.clamp(row_mass - other_max, min=0.0)

            step_trial_evidence = {timestep: 0.0 for timestep in hidden_coord_evidence}

            for site_key, evidence in zip(unit["sites"], dominant.tolist()):
                parsed = parse_site_key(site_key, hidden_size=hidden_size)
                if parsed.kind == "output_logit":
                    logit_evidence[site_key] = float(logit_evidence.get(site_key, 0.0) + float(weight) * float(evidence))
                    continue
                timestep = int(parsed.timestep)
                coords = parsed.coord_indices if parsed.coord_indices is not None else tuple(range(hidden_size))
                if not coords:
                    continue
                per_coord = float(weight) * float(evidence) / float(len(coords))
                for coord in coords:
                    hidden_coord_evidence[timestep][int(coord)] += per_coord
                step_trial_evidence[timestep] += float(weight) * float(evidence)

            dominant_step = max(step_trial_evidence.items(), key=lambda item: (float(item[1]), -int(item[0])))[0]
            dominant_steps.append(int(dominant_step))
            retained_trial_meta.append(
                {
                    "resolution": int(unit["resolution"]),
                    "epsilon": float(unit["epsilon"]),
                    "profile_key": str(unit["profile_key"]),
                    "row_score": float(unit["row_score"]),
                    "weight": float(weight),
                    "dominant_hidden_timestep": int(dominant_step),
                    "selected_top_k": int(unit["row_selected"]["top_k"]),
                    "selected_lambda": float(unit["row_selected"]["lambda"]),
                }
            )

        timestep_evidence = {
            timestep: float(sum(coord_values))
            for timestep, coord_values in hidden_coord_evidence.items()
        }
        dominant_timestep = max(timestep_evidence.items(), key=lambda item: (float(item[1]), -int(item[0])))[0]
        sorted_steps = sorted(timestep_evidence.items(), key=lambda item: (-float(item[1]), int(item[0])))
        dominant_stability = float(sum(int(step == int(dominant_timestep)) for step in dominant_steps) / max(1, len(dominant_steps)))
        dominant_coord_evidence = hidden_coord_evidence[int(dominant_timestep)]

        masks = {
            "StepMask": full_timestep_mask(timestep=int(dominant_timestep)),
        }
        compression = {
            "StepMask": {
                "size": int(hidden_size),
                "compression_ratio": 1.0,
            }
        }
        for threshold in mask_thresholds:
            coords = _coord_mask_from_threshold(dominant_coord_evidence, threshold=float(threshold))
            label = f"S{int(round(float(threshold) * 100))}"
            masks[label] = coord_group_mask(timestep=int(dominant_timestep), coord_indices=coords)
            compression[label] = {
                "size": int(len(coords)),
                "compression_ratio": float(len(coords) / max(1, hidden_size)),
            }

        extracted_rows[row_key] = {
            "best_row_score": float(max(float(unit["row_score"]) for unit in retained)),
            "retained_trial_count": int(len(retained)),
            "retained_trials": retained_trial_meta,
            "dominant_hidden_timestep": int(dominant_timestep),
            "dominant_hidden_timestep_stability": float(dominant_stability),
            "top_hidden_timesteps": [
                {"timestep": int(timestep), "evidence": float(evidence)}
                for timestep, evidence in sorted_steps
            ],
            "hidden_timestep_evidence": {f"h_{timestep}": float(evidence) for timestep, evidence in timestep_evidence.items()},
            "hidden_coord_evidence": {
                f"h_{timestep}": [float(value) for value in coord_values]
                for timestep, coord_values in hidden_coord_evidence.items()
            },
            "logit_evidence": {site_key: float(evidence) for site_key, evidence in sorted(logit_evidence.items())},
            "masks": masks,
            "compression": compression,
        }

    result = {
        "config": vars(args),
        "source_summary": str(summary_path),
        "hidden_size": hidden_size,
        "row_support": extracted_rows,
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"support_summary": str(out_path), "rows": list(extracted_rows.keys())}, indent=2))


if __name__ == "__main__":
    main()
