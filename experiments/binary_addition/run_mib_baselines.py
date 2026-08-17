#!/usr/bin/env python3
"""Run Full State, DBM, and DBM+PCA on the main binary-addition GRU."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from experiments.binary_addition.data import enumerate_all_examples, stratified_base_split
from experiments.binary_addition.dbm_baselines import (
    IdentityBasis,
    PCABasis,
    collect_pca_observations,
    evaluate_candidate,
    train_dbm,
)
from experiments.binary_addition.interventions import build_run_cache
from experiments.binary_addition.model import exact_accuracy
from experiments.binary_addition.run_joint_endogenous_resolution_sweep import (
    EndogenousRowSpec,
    _bank_summaries,
    _build_banks,
    _load_or_train_model,
)


METHODS = ("full-state", "dbm-canonical", "dbm-pca")


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_floats(value: str) -> tuple[float, ...]:
    values: list[float] = []
    for item in value.split(","):
        if not item.strip():
            continue
        parsed = float(item.strip())
        if not math.isfinite(parsed) or parsed < 0.0:
            raise ValueError(f"regularization coefficients must be finite and nonnegative: {parsed}")
        if parsed not in values:
            values.append(parsed)
    if not values:
        raise ValueError("regularization coefficient grid cannot be empty")
    return tuple(values)


def regularization_grid(args: argparse.Namespace) -> tuple[float, ...]:
    grid = str(getattr(args, "regularization_coefficients", "")).strip()
    if grid:
        return parse_floats(grid)
    return parse_floats(str(args.regularization_coefficient))


def float_slug(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace("+", "p").replace(".", "p")


def parse_checkpoint_map(value: str) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for chunk in str(value).split(";"):
        if not chunk.strip():
            continue
        seed, path = chunk.split("=", 1)
        mapping[int(seed.strip())] = Path(path.strip())
    return mapping


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=True)) if len(values) > 1 else 0.0,
    }


def default_checkpoint(seed: int, hidden_size: int) -> Path:
    if int(seed) == 0:
        return ROOT / "eval" / f"binary_backbone_h{hidden_size}_seed0" / "gru_adder.pt"
    return ROOT / "eval" / "shared_checkpoints" / f"gru_h{hidden_size}_seed{seed}.pt"


def resume_protocol(
    args: argparse.Namespace,
    *,
    seed: int,
    checkpoint: str,
) -> dict[str, object]:
    """Return every setting that can change a cached baseline artifact."""

    return {
        "version": 1,
        "seed": int(seed),
        "checkpoint": str(Path(checkpoint).resolve()),
        "rows": [item.strip() for item in str(args.rows).split(",") if item.strip()],
        "timesteps": list(parse_ints(str(args.timesteps))),
        "width": int(args.width),
        "hidden_size": int(args.hidden_size),
        "fit_bases": int(args.fit_bases),
        "calib_bases": int(args.calib_bases),
        "test_bases": int(args.test_bases),
        "source_policy": str(args.source_policy),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "temperature_start": float(args.temperature_start),
        "temperature_end": float(args.temperature_end),
        "regularization_coefficient": float(args.regularization_coefficient),
        "regularization_coefficients": list(regularization_grid(args)),
        "pca_rank": None if args.pca_rank is None else int(args.pca_rank),
        "train_on": str(args.train_on),
        "train_epochs": int(args.train_epochs),
        "train_batch_size": int(args.train_batch_size),
        "train_lr": float(args.train_lr),
    }


def artifact_protocol(
    base: dict[str, object],
    *,
    artifact: str,
    method: str | None = None,
    row: str | None = None,
    timestep: int,
    regularization_coefficient: float | None = None,
) -> dict[str, object]:
    protocol = {**base, "artifact": str(artifact), "timestep": int(timestep)}
    if method is not None:
        protocol["method"] = str(method)
    if row is not None:
        protocol["row"] = str(row)
    if regularization_coefficient is not None:
        protocol["regularization_coefficient"] = float(regularization_coefficient)
    return protocol


def can_resume(payload: object, expected_protocol: dict[str, object]) -> bool:
    return isinstance(payload, dict) and payload.get("resume_protocol") == expected_protocol


def candidate_sort_key(record: dict[str, object]) -> tuple[float, float, float, int, float, int]:
    calibration = record["calibration"]
    if not isinstance(calibration, dict):
        raise TypeError(f"candidate has invalid calibration payload: {calibration!r}")
    return (
        -float(calibration["combined"]),
        -float(calibration["sensitivity"]),
        -float(calibration["invariance"]),
        int(record.get("site_size", 1 << 30)),
        float(record.get("regularization_coefficient") or 0.0),
        int(record["timestep"]),
    )


def warm_dbm_optimizer(device: torch.device) -> None:
    """Pay one-time PyTorch optimizer initialization outside method timers."""
    parameter = torch.nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.AdamW((parameter,), lr=1e-3, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    parameter.sum().backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "delta")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--checkpoint-map",
        default="",
        help="Optional semicolon-separated seed=checkpoint paths; prevents fallback retraining.",
    )
    parser.add_argument("--rows", default="C1,C2,C3")
    parser.add_argument("--timesteps", default="0,1,2,3")
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--fit-bases", type=int, default=128)
    parser.add_argument("--calib-bases", type=int, default=64)
    parser.add_argument("--test-bases", type=int, default=64)
    parser.add_argument("--source-policy", default="structured_26_top3carry_c2x5_c3x7_no_random")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=0.01)
    parser.add_argument("--regularization-coefficient", type=float, default=0.0)
    parser.add_argument(
        "--regularization-coefficients",
        default="",
        help=(
            "Optional comma-separated DBM sparsity sweep. Each coefficient is trained and "
            "calibrated at every timestep; calibration jointly selects coefficient and timestep. "
            "When omitted, --regularization-coefficient supplies the single value."
        ),
    )
    parser.add_argument("--pca-rank", type=int, default=None)
    parser.add_argument("--train-on", choices=("all", "fit_only"), default="all")
    parser.add_argument("--train-epochs", type=int, default=120)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--train-lr", type=float, default=0.02)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Write only per-seed candidate artifacts; a later resume pass can build the aggregate.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    seeds = parse_ints(args.seeds)
    checkpoint_map = parse_checkpoint_map(args.checkpoint_map)
    if checkpoint_map:
        missing_seeds = sorted(set(seeds) - set(checkpoint_map))
        if missing_seeds:
            raise ValueError(f"Checkpoint map is missing seeds: {missing_seeds}")
        missing_paths = [str(path) for path in checkpoint_map.values() if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(f"Checkpoint paths do not exist: {missing_paths}")
    timesteps = parse_ints(args.timesteps)
    dbm_regularization_grid = regularization_grid(args)
    row_keys = tuple(item.strip() for item in args.rows.split(",") if item.strip())
    specs = tuple(
        EndogenousRowSpec(key=row, kind="carry", index=int(row[1:])) for row in row_keys
    )
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA requested but unavailable; run inside the allocated srun step")
    if any(method.startswith("dbm-") for method in methods):
        warm_dbm_optimizer(device)
    run_name = args.run_name or f"binary_addition_mib_baselines_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config_name = (
        f"config_worker_{'_'.join(str(seed) for seed in seeds)}.json"
        if args.skip_aggregate
        else "config.json"
    )
    atomic_json(run_dir / config_name, {**vars(args), "out_dir": str(args.out_dir)})

    all_examples = enumerate_all_examples(width=args.width)
    for seed in seeds:
        split = stratified_base_split(
            all_examples,
            fit_count=args.fit_bases,
            calib_count=args.calib_bases,
            test_count=args.test_bases,
            seed=seed,
        )
        standard_checkpoint = default_checkpoint(seed, args.hidden_size)
        fallback_checkpoint = run_dir / "checkpoints" / f"gru_h{args.hidden_size}_seed{seed}.pt"
        args.seed = int(seed)
        args.model_checkpoint = str(
            checkpoint_map[seed]
            if checkpoint_map
            else (standard_checkpoint if standard_checkpoint.exists() else fallback_checkpoint)
        )
        model, model_metadata = _load_or_train_model(args, all_examples, split)
        model.eval()
        model.requires_grad_(False)
        seed_protocol = resume_protocol(
            args,
            seed=seed,
            checkpoint=str(args.model_checkpoint),
        )
        run_cache = build_run_cache(model, all_examples, device=device)
        banks = _build_banks(
            split,
            specs,
            width=args.width,
            seed=seed,
            source_policy=args.source_policy,
            all_examples=all_examples,
        )
        # The paper protocol gives every row the same 3,328 fit pairs.  PCA only
        # needs one copy; collect_pca_observations performs a second defensive
        # pair-level deduplication.
        fit_records = tuple(banks["fit_by_row"][row_keys[0]])
        pca_by_timestep: dict[int, PCABasis] = {}
        pca_metadata: dict[str, object] = {}
        if "dbm-pca" in methods:
            for timestep in timesteps:
                pca_path = run_dir / "pca" / f"seed{seed}_t{timestep}.pt"
                expected_pca_protocol = artifact_protocol(
                    seed_protocol,
                    artifact="pca_basis",
                    timestep=timestep,
                )
                if pca_path.exists() and not args.no_resume:
                    cached_pca = torch.load(pca_path, map_location="cpu")
                    cached_metadata = dict(cached_pca.get("metadata", {}))
                    if can_resume(cached_metadata, expected_pca_protocol):
                        pca_by_timestep[timestep] = PCABasis(cached_pca["components"])
                        pca_metadata[str(timestep)] = cached_metadata
                        continue
                    print(f"[rebuild] incompatible PCA artifact: {pca_path}")
                if timestep not in pca_by_timestep:
                    synchronize_device(device)
                    pca_started = perf_counter()
                    observations = collect_pca_observations(
                        fit_records, timestep=timestep, run_cache=run_cache
                    )
                    pca_by_timestep[timestep] = PCABasis.fit(observations, rank=args.pca_rank)
                    synchronize_device(device)
                    pca_metadata[str(timestep)] = {
                        "observation_count": int(observations.shape[0]),
                        "hidden_size": int(observations.shape[1]),
                        "feature_dim": int(pca_by_timestep[timestep].feature_dim),
                        "fit_seconds": float(perf_counter() - pca_started),
                        "resume_protocol": expected_pca_protocol,
                    }
                    pca_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "components": pca_by_timestep[timestep].components.cpu(),
                            "metadata": pca_metadata[str(timestep)],
                        },
                        pca_path,
                    )
        atomic_json(
            run_dir / f"seed{seed}_metadata.json",
            {
                "seed": seed,
                "model": model_metadata,
                "factual_exact": exact_accuracy(model, all_examples, device=device),
                "banks": {
                    "fit": _bank_summaries(
                        banks["fit_by_row"],
                        {key: tuple(record for record in banks["fit_by_row"][key] if record.is_active) for key in row_keys},
                        {key: tuple(record for record in banks["fit_by_row"][key] if not record.is_active) for key in row_keys},
                    )
                },
                "pca": pca_metadata,
            },
        )

        for method in methods:
            for row in row_keys:
                row_payloads: list[tuple[Path, dict[str, object]]] = []
                for timestep in timesteps:
                    coefficient_candidates: tuple[float | None, ...] = (
                        tuple(dbm_regularization_grid)
                        if method.startswith("dbm-")
                        else (None,)
                    )
                    for regularization_coefficient in coefficient_candidates:
                        coefficient_suffix = (
                            f"_lambda{float_slug(regularization_coefficient)}"
                            if regularization_coefficient is not None
                            else ""
                        )
                        stem = f"{method}_seed{seed}_{row}_t{timestep}{coefficient_suffix}"
                        output_path = run_dir / method / f"{stem}.json"
                        expected_candidate_protocol = artifact_protocol(
                            seed_protocol,
                            artifact="candidate",
                            method=method,
                            row=row,
                            timestep=timestep,
                            regularization_coefficient=regularization_coefficient,
                        )
                        if output_path.exists() and not args.no_resume:
                            payload = json.loads(output_path.read_text())
                            if can_resume(payload, expected_candidate_protocol):
                                print(f"[resume] {output_path}")
                                if "candidate_search_seconds" not in payload:
                                    prior_test_seconds = float(
                                        (payload.get("test") or {}).get("evaluation_seconds", 0.0)
                                    )
                                    payload["candidate_search_seconds"] = max(
                                        0.0, float(payload.get("total_seconds", 0.0)) - prior_test_seconds
                                    )
                                row_payloads.append((output_path, payload))
                                continue
                            print(f"[rebuild] incompatible candidate artifact: {output_path}")
                        synchronize_device(device)
                        started = perf_counter()
                        basis = (
                            pca_by_timestep[timestep]
                            if method == "dbm-pca"
                            else IdentityBasis(args.hidden_size)
                        )
                        payload: dict[str, object] = {
                            "method": method,
                            "seed": seed,
                            "row": row,
                            "timestep": timestep,
                            "basis": "pca" if method == "dbm-pca" else "canonical",
                            "regularization_coefficient": regularization_coefficient,
                            "resume_protocol": expected_candidate_protocol,
                        }
                        if method == "full-state":
                            gate = torch.ones(basis.feature_dim, device=device)
                            payload["site_size"] = int(args.hidden_size)
                        else:
                            if regularization_coefficient is None:
                                raise RuntimeError(f"DBM candidate is missing a coefficient: {stem}")
                            mask, training = train_dbm(
                                model=model,
                                records=banks["fit_by_row"][row],
                                timestep=timestep,
                                basis=basis,
                                run_cache=run_cache,
                                device=device,
                                batch_size=args.batch_size,
                                epochs=args.epochs,
                                learning_rate=args.learning_rate,
                                temperature_start=args.temperature_start,
                                temperature_end=args.temperature_end,
                                regularization_coefficient=regularization_coefficient,
                                seed=seed,
                            )
                            gate = mask.gate(temperature=1.0, hard=True)
                            payload["site_size"] = int(gate.sum().item())
                            payload["training"] = training
                            checkpoint_path = run_dir / method / f"{stem}.pt"
                            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                            checkpoint_payload: dict[str, object] = {
                                "mask_logits": mask.logits.detach().cpu(),
                                "basis": payload["basis"],
                                "regularization_coefficient": regularization_coefficient,
                            }
                            if isinstance(basis, PCABasis):
                                checkpoint_payload["pca_components"] = basis.components.cpu()
                            payload["checkpoint"] = str(checkpoint_path.relative_to(run_dir))
                        payload["calibration"] = evaluate_candidate(
                            model=model,
                            positive_records=banks["calib_positive_by_row"][row],
                            invariant_records=banks["calib_invariant_by_row"][row],
                            timestep=timestep,
                            basis=basis,
                            gate=gate,
                            run_cache=run_cache,
                            device=device,
                            batch_size=args.eval_batch_size,
                        )
                        # Candidate-search runtime includes mask training and
                        # calibration for every coefficient/timestep pair, but
                        # excludes checkpoint/result serialization and test.
                        synchronize_device(device)
                        payload["candidate_search_seconds"] = float(perf_counter() - started)
                        payload["test"] = None
                        payload["selected_for_test"] = False
                        payload["total_seconds"] = float(payload["candidate_search_seconds"])
                        if method != "full-state":
                            torch.save(checkpoint_payload, checkpoint_path)
                        atomic_json(output_path, payload)
                        row_payloads.append((output_path, payload))
                        print(
                            f"[candidate] {stem} cal={payload['calibration']['combined']:.4f} "
                            f"size={payload['site_size']}"
                        )

                row_payloads.sort(key=lambda item: candidate_sort_key(item[1]))
                selected_path, selected_payload = row_payloads[0]
                if not bool(selected_payload.get("selected_for_test")) or not isinstance(
                    selected_payload.get("test"), dict
                ):
                    selected_timestep = int(selected_payload["timestep"])
                    selected_basis = (
                        pca_by_timestep[selected_timestep]
                        if method == "dbm-pca"
                        else IdentityBasis(args.hidden_size)
                    )
                    if method == "full-state":
                        selected_gate = torch.ones(selected_basis.feature_dim, device=device)
                    else:
                        checkpoint_path = run_dir / str(selected_payload["checkpoint"])
                        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
                        selected_gate = (checkpoint_payload["mask_logits"] > 0).to(
                            device=device, dtype=torch.float32
                        )
                    selected_payload["test"] = evaluate_candidate(
                        model=model,
                        positive_records=banks["test_positive_by_row"][row],
                        invariant_records=banks["test_invariant_by_row"][row],
                        timestep=selected_timestep,
                        basis=selected_basis,
                        gate=selected_gate,
                        run_cache=run_cache,
                        device=device,
                        batch_size=args.eval_batch_size,
                    )
                for output_path, payload in row_payloads:
                    is_selected = output_path == selected_path
                    payload["selected_for_test"] = bool(is_selected)
                    if not is_selected:
                        payload["test"] = None
                    selected_test_seconds = (
                        float(payload["test"].get("evaluation_seconds", 0.0))
                        if is_selected and isinstance(payload.get("test"), dict)
                        else 0.0
                    )
                    payload["total_seconds"] = float(payload["candidate_search_seconds"]) + selected_test_seconds
                    atomic_json(output_path, payload)
                print(
                    f"[selected] {method} seed={seed} row={row} "
                    f"t={selected_payload['timestep']} "
                    f"lambda={selected_payload.get('regularization_coefficient')} "
                    f"size={selected_payload.get('site_size')} "
                    f"test={selected_payload['test']['combined']:.4f}"
                )

    if args.skip_aggregate:
        print(f"Per-seed candidate outputs saved under: {run_dir.resolve()}")
        return

    active_timesteps = set(timesteps)
    records = [
        record
        for path in run_dir.glob("*/*.json")
        for record in [json.loads(path.read_text())]
        if record.get("method") in methods
        and record.get("seed") in seeds
        and record.get("row") in row_keys
        and record.get("timestep") in active_timesteps
    ]
    selections: dict[str, object] = {}
    for method in methods:
        for seed in seeds:
            for row in row_keys:
                candidates = [
                    record for record in records
                    if record.get("method") == method and record.get("seed") == seed and record.get("row") == row
                ]
                candidates.sort(key=candidate_sort_key)
                selections[f"{method}/seed{seed}/{row}"] = {
                    "calibration_selected": candidates[0], "all_timesteps": candidates
                }
    aggregate: dict[str, object] = {}
    for method in methods:
        seed_values = []
        for seed in seeds:
            selected_rows = [selections[f"{method}/seed{seed}/{row}"]["calibration_selected"] for row in row_keys]
            all_method_candidates = [
                record for record in records
                if record.get("method") == method and record.get("seed") == seed
            ]
            pca_setup_seconds = 0.0
            if method == "dbm-pca":
                seed_metadata = json.loads((run_dir / f"seed{seed}_metadata.json").read_text())
                pca_setup_seconds = sum(
                    float(item.get("fit_seconds", 0.0))
                    for item in seed_metadata.get("pca", {}).values()
                )
            candidate_search_seconds = sum(
                float(item.get("candidate_search_seconds", item.get("total_seconds", 0.0)))
                for item in all_method_candidates
            )
            selected_test_seconds = sum(
                float((item.get("test") or {}).get("evaluation_seconds", 0.0))
                for item in selected_rows
            )
            seed_values.append(
                {
                    "seed": seed,
                    "combined": sum(float(item["test"]["combined"]) for item in selected_rows) / len(selected_rows),
                    "sensitivity": sum(float(item["test"]["sensitivity"]) for item in selected_rows) / len(selected_rows),
                    "invariance": sum(float(item["test"]["invariance"]) for item in selected_rows) / len(selected_rows),
                    "runtime_seconds": pca_setup_seconds + candidate_search_seconds + selected_test_seconds,
                    "pca_setup_seconds": pca_setup_seconds,
                    "candidate_search_seconds": candidate_search_seconds,
                    "selected_test_seconds": selected_test_seconds,
                    "selected_regularization_coefficients_by_row": {
                        row: selections[f"{method}/seed{seed}/{row}"]["calibration_selected"].get(
                            "regularization_coefficient"
                        )
                        for row in row_keys
                    },
                    "selected_candidate_runtime_seconds": sum(
                        float(item["total_seconds"]) for item in selected_rows
                    ),
                }
            )
        aggregate[method] = {
            metric: summary_stats([float(item[metric]) for item in seed_values])
            for metric in ("combined", "sensitivity", "invariance", "runtime_seconds")
        }
        aggregate[method]["runtime_definition"] = (
            "PCA fitting when applicable, candidate training and calibration at every timestep "
            "and every DBM regularization coefficient, and test evaluation only at each row's "
            "jointly calibration-selected coefficient/timestep; shared model, data-bank, and "
            "activation-cache setup and result serialization are excluded"
        )
        aggregate[method]["regularization_coefficients"] = (
            list(dbm_regularization_grid) if method.startswith("dbm-") else []
        )
        aggregate[method]["runtime_breakdown_seconds"] = {
            key: summary_stats([float(item[key]) for item in seed_values])
            for key in ("pca_setup_seconds", "candidate_search_seconds", "selected_test_seconds")
        }
        aggregate[method]["per_seed"] = seed_values
    atomic_json(run_dir / "rankings.json", selections)
    atomic_json(run_dir / "aggregate.json", aggregate)
    print(f"All results saved under: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
