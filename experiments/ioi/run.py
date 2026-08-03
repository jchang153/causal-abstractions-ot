"""Run blind DAS, PLOT-DAS, and oracle DAS on the released MIB IOI task."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.ioi.causal import (  # noqa: E402
    PUBLISHED_COEFFICIENTS,
    VARIABLES,
    LinearCausalModel,
    fit_linear_causal_model,
)
from experiments.ioi.data import IOIBanks, load_and_filter_banks  # noqa: E402
from experiments.ioi.interventions import (  # noqa: E402
    DASTrainConfig,
    HeadRotations,
    ORACLE_HEADS,
    all_gpt2_heads,
    evaluate_mse,
    head_label,
    intervention_logit_differences,
    load_gpt2,
    train_joint_das,
)
from experiments.ioi.plot import (  # noqa: E402
    SignatureBank,
    build_cost_matrix,
    calibrate_uot_grid,
    choose_k,
    collect_signature_bank,
    top_k_heads,
)


DEFAULT_EPSILONS = (0.5, 1.0, 2.0)
DEFAULT_BETA_NEURALS = (0.1, 0.3, 1.0, 3.0)
DEFAULT_K_VALUES = (1, 2, 3)


def _csv(value: str, cast):
    result = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated value")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default="blind,plot,oracle", help="Comma-separated: blind,plot,oracle")
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--dataset-name", default="mib-bench/ioi")
    parser.add_argument(
        "--dataset-revision",
        default="5024626",
        help="Pinned pre-test-shrink MIB dataset revision",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--effective-batch-size", type=int, default=1024)
    parser.add_argument("--filter-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--calibration-rows", type=int, default=2000)
    parser.add_argument("--signature-bank-size", type=int, default=1000)
    parser.add_argument("--uot-epsilons", default="0.5,1,2")
    parser.add_argument("--uot-beta-neural", default="0.1,0.3,1,3")
    parser.add_argument("--plot-k", default="1,2,3")
    parser.add_argument("--das-dimension", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--coefficient-tolerance", type=float, default=0.15)
    parser.add_argument("--allow-coefficient-mismatch", action="store_true")
    parser.add_argument("--allow-version-mismatch", action="store_true")
    parser.add_argument("--quick-rows", type=int, default=None, help="Tiny development run row cap")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/ioi"))
    parser.add_argument("--force", action="store_true", help="Ignore compatible cached artifacts")
    args = parser.parse_args(argv)
    args.methods = _csv(args.methods, str)
    unknown = set(args.methods) - {"blind", "plot", "oracle"}
    if unknown:
        parser.error(f"Unknown methods: {sorted(unknown)}")
    args.uot_epsilons = _csv(args.uot_epsilons, float)
    args.uot_beta_neural = _csv(args.uot_beta_neural, float)
    args.plot_k = _csv(args.plot_k, int)
    for positive in (
        "microbatch_size", "effective_batch_size", "filter_batch_size", "eval_batch_size",
        "calibration_rows", "signature_bank_size", "das_dimension", "epochs",
    ):
        if int(getattr(args, positive)) <= 0:
            parser.error(f"--{positive.replace('_', '-')} must be positive")
    return args


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=_jsonable) + "\n")
    os.replace(temporary, path)


def _read_json(path: Path):
    return json.loads(path.read_text())


def _hash(value: object) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), default=_jsonable)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _configuration(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: value
        for key, value in vars(args).items()
        if key not in {
            "output_dir", "hf_token", "force", "allow_coefficient_mismatch", "allow_version_mismatch"
        }
    }


def _artifact(path: Path, *, force: bool, loader, builder, dumper=_write_json):
    if path.exists() and not force:
        return loader(path), True
    value = builder()
    dumper(path, value)
    return value, False


def _fit_regression(model, tokenizer, banks: IOIBanks, device, batch_size):
    same_values = intervention_logit_differences(
        model,
        tokenizer,
        banks.same_fit,
        factual=True,
        device=device,
        batch_size=batch_size,
    )
    patched = {
        family: intervention_logit_differences(
            model,
            tokenizer,
            rows,
            heads=ORACLE_HEADS,
            rotations=None,
            device=device,
            batch_size=batch_size,
        )
        for family, rows in banks.fit.items()
    }
    causal_model, diagnostics = fit_linear_causal_model(same_values, patched)
    return {"model": causal_model.as_dict(), "diagnostics": diagnostics}


def validate_intervention_stack(*, allow_mismatch: bool = False) -> dict[str, object]:
    """Validate the versions locked for reproducing MIB intervention semantics."""

    expected = {"torch": "2.8.0", "transformers": "4.56.2", "pyvene": "0.1.8"}
    installed = {}
    mismatches = {}
    for package, wanted in expected.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        installed[package] = actual
        if actual != wanted and not actual.startswith(wanted + "+"):
            mismatches[package] = {"expected": wanted, "installed": actual}
    if mismatches and not allow_mismatch:
        raise RuntimeError(
            f"MIB-compatible dependency validation failed: {mismatches}. "
            "Install requirements.txt or pass --allow-version-mismatch for a non-reference smoke run."
        )
    return {"expected": expected, "installed": installed, "mismatches": mismatches}


def _save_checkpoint(path: Path, rotations: HeadRotations, training: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {"state_dict": rotations.state_dict(), "metadata": rotations.metadata(), "training": dict(training)},
        temporary,
    )
    os.replace(temporary, path)


def _load_checkpoint(path: Path, heads, subspace_dim, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    rotations = HeadRotations(heads, subspace_dim=subspace_dim).to(device)
    rotations.load_state_dict(payload["state_dict"])
    rotations.eval()
    return rotations, payload["training"]


def _train_cached(
    path: Path,
    *,
    force: bool,
    model,
    tokenizer,
    banks,
    variable,
    causal_model,
    heads,
    device,
    config,
):
    if path.exists() and not force:
        rotations, training = _load_checkpoint(path, heads, config.subspace_dim, device)
        return rotations, training, True
    rotations, training = train_joint_das(
        model,
        tokenizer,
        banks.fit,
        variable=variable,
        causal_model=causal_model,
        heads=heads,
        device=device,
        config=config,
    )
    _save_checkpoint(path, rotations, training)
    return rotations, training, False


def _evaluate_selected(
    model, tokenizer, banks, variable, causal_model, heads, rotations, device, batch_size
):
    started = perf_counter()
    metrics = evaluate_mse(
        model,
        tokenizer,
        banks.test,
        variable=variable,
        causal_model=causal_model,
        heads=heads,
        rotations=rotations,
        device=device,
        batch_size=batch_size,
    )
    return metrics, float(perf_counter() - started)


def _summary_text(summary: Mapping[str, object]) -> str:
    lines = ["Blind IOI Head Localization with PLOT-DAS", ""]
    causal = summary["causal_model"]["model"]
    lines.append(
        "Causal coefficients: "
        f"bias={causal['bias']:.6f}, position={causal['position_coeff']:.6f}, "
        f"token={causal['token_coeff']:.6f}, R2={causal['r2']:.6f}"
    )
    for method, payload in summary["methods"].items():
        lines.extend(["", method.upper()])
        for variable, result in payload["variables"].items():
            lines.append(
                f"  {variable}: test macro MSE={result['test']['macro_mse']:.6f}; "
                f"heads={','.join(result['heads'])}"
            )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stack_validation = validate_intervention_stack(allow_mismatch=args.allow_version_mismatch)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _configuration(args)
    config_hash = _hash(config)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        old_hash = _read_json(manifest_path).get("config_hash")
        if old_hash != config_hash:
            raise RuntimeError(
                f"Output directory has config hash {old_hash}, not {config_hash}; use a new directory or --force"
            )
    _write_json(
        manifest_path,
        {
            "config": config,
            "config_hash": config_hash,
            "status": "running",
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "intervention_stack": stack_validation,
            },
        },
    )

    device = torch.device(args.device)
    model_load_started = perf_counter()
    model, tokenizer = load_gpt2(args.model_name, device)
    shared_runtime = {"model_loading_seconds": float(perf_counter() - model_load_started)}

    data_started = perf_counter()
    banks_path = output_dir / "banks.json"
    banks, banks_cached = _artifact(
        banks_path,
        force=args.force,
        loader=lambda path: IOIBanks.from_dict(_read_json(path)),
        builder=lambda: load_and_filter_banks(
            model,
            tokenizer,
            device=device,
            split_seed=args.split_seed,
            calibration_rows=args.calibration_rows,
            signature_rows=args.signature_bank_size,
            filter_batch_size=args.filter_batch_size,
            dataset_name=args.dataset_name,
            dataset_revision=args.dataset_revision,
            hf_token=args.hf_token,
            quick_rows=args.quick_rows,
        ),
        dumper=lambda path, value: _write_json(path, value.as_dict()),
    )
    shared_runtime["data_preparation_seconds"] = float(perf_counter() - data_started)

    regression_started = perf_counter()
    regression_path = output_dir / "causal_model.json"
    regression_payload, regression_cached = _artifact(
        regression_path,
        force=args.force,
        loader=_read_json,
        builder=lambda: _fit_regression(
            model, tokenizer, banks, device, args.eval_batch_size
        ),
    )
    shared_runtime["data_filtering_cached"] = banks_cached
    shared_runtime["regression_seconds"] = float(perf_counter() - regression_started)
    shared_runtime["regression_cached"] = regression_cached
    causal_model = LinearCausalModel.from_dict(regression_payload["model"])
    discrepancies = {
        key: abs(float(getattr(causal_model, key)) - float(expected))
        for key, expected in PUBLISHED_COEFFICIENTS.items()
    }
    bad = {key: value for key, value in discrepancies.items() if value > args.coefficient_tolerance}
    if bad and not args.allow_coefficient_mismatch:
        raise RuntimeError(
            f"Refitted causal coefficients exceed tolerance {args.coefficient_tolerance}: {bad}. "
            "Inspect the cached filtering/regression artifacts or pass --allow-coefficient-mismatch."
        )

    das_config = DASTrainConfig(
        subspace_dim=args.das_dimension,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        effective_batch_size=args.effective_batch_size,
        micro_batch_size=args.microbatch_size,
        seed=args.seed,
    )
    method_results: dict[str, object] = {}
    deferred_fixed_evaluations: dict[str, dict[str, tuple[object, object]]] = {}

    for method, heads in (("blind", all_gpt2_heads()), ("oracle", ORACLE_HEADS)):
        if method not in args.methods:
            continue
        variables = {}
        runtime_objects = {}
        method_train_cold = 0.0
        method_train_current = 0.0
        for variable in VARIABLES:
            checkpoint = output_dir / "checkpoints" / f"{method}_{variable}.pt"
            rotations, training, cached = _train_cached(
                checkpoint,
                force=args.force,
                model=model,
                tokenizer=tokenizer,
                banks=banks,
                variable=variable,
                causal_model=causal_model,
                heads=heads,
                device=device,
                config=das_config,
            )
            method_train_cold += float(training["runtime_seconds"])
            method_train_current += 0.0 if cached else float(training["runtime_seconds"])
            variables[variable] = {
                "heads": [head_label(head) for head in heads],
                "k": len(heads),
                "training": training,
                "checkpoint_cached": cached,
            }
            runtime_objects[variable] = (heads, rotations)
        deferred_fixed_evaluations[method] = runtime_objects
        method_results[method] = {
            "variables": variables,
            "runtime": {
                "das_training_seconds": method_train_cold,
                "current_das_training_seconds": method_train_current,
                "heldout_evaluation_seconds": 0.0,
                "cold_seconds": method_train_cold,
                "current_invocation_seconds": method_train_current,
                "amortized_seconds_per_variable": method_train_cold / len(VARIABLES),
            },
        }

    if "plot" in args.methods:
        signature_path = output_dir / "plot" / "signatures.json"
        signatures, signature_cached = _artifact(
            signature_path,
            force=args.force,
            loader=lambda path: SignatureBank.from_dict(_read_json(path)),
            builder=lambda: collect_signature_bank(
                model,
                tokenizer,
                banks.signature_fit,
                device=device,
                batch_size=args.eval_batch_size,
            ),
            dumper=lambda path, value: _write_json(path, value.as_dict()),
        )
        cost, cost_diagnostics = build_cost_matrix(signatures, causal_model)
        _write_json(output_dir / "plot" / "costs.json", cost_diagnostics)
        uot_started = perf_counter()
        uot_path = output_dir / "plot" / "uot_calibration.json"
        if uot_path.exists() and not args.force:
            uot_payload = _read_json(uot_path)
            uot_cached = True
        else:
            selected_trial, trials = calibrate_uot_grid(
                model,
                tokenizer,
                banks.calibration,
                causal_model=causal_model,
                cost=cost,
                heads=signatures.heads,
                epsilons=args.uot_epsilons,
                beta_neurals=args.uot_beta_neural,
                k_values=args.plot_k,
                device=device,
                batch_size=args.eval_batch_size,
            )
            uot_payload = {
                "selected": selected_trial,
                "trials": trials,
                "runtime_seconds": float(perf_counter() - uot_started),
            }
            _write_json(uot_path, uot_payload)
            uot_cached = False
        uot_seconds = float(perf_counter() - uot_started)
        frozen_coupling = np.asarray(uot_payload["selected"]["coupling"], dtype=np.float64)

        plot_variables = {}
        selected_runtime_objects = {}
        plot_training_cold_seconds = 0.0
        plot_training_current_seconds = 0.0
        plot_calibration_seconds = 0.0
        plot_test_seconds = 0.0
        for variable in VARIABLES:
            candidates = []
            rotations_by_k = {}
            heads_by_k = {}
            for k in sorted(set(args.plot_k)):
                heads = top_k_heads(frozen_coupling, signatures.heads, variable, k)
                checkpoint = output_dir / "checkpoints" / f"plot_{variable}_k{k}.pt"
                rotations, training, cached = _train_cached(
                    checkpoint,
                    force=args.force,
                    model=model,
                    tokenizer=tokenizer,
                    banks=banks,
                    variable=variable,
                    causal_model=causal_model,
                    heads=heads,
                    device=device,
                    config=das_config,
                )
                calibration_started = perf_counter()
                metrics = evaluate_mse(
                    model,
                    tokenizer,
                    banks.calibration,
                    variable=variable,
                    causal_model=causal_model,
                    heads=heads,
                    rotations=rotations,
                    device=device,
                    batch_size=args.eval_batch_size,
                )
                calibration_seconds = float(perf_counter() - calibration_started)
                plot_training_cold_seconds += float(training["runtime_seconds"])
                plot_training_current_seconds += (
                    0.0 if cached else float(training["runtime_seconds"])
                )
                plot_calibration_seconds += calibration_seconds
                candidates.append(
                    {
                        "k": int(k),
                        "heads": [head_label(head) for head in heads],
                        "macro_mse": float(metrics["macro_mse"]),
                        "metrics": metrics,
                        "training": training,
                        "checkpoint_cached": cached,
                        "calibration_runtime_seconds": calibration_seconds,
                    }
                )
                rotations_by_k[int(k)] = rotations
                heads_by_k[int(k)] = heads
            selected = choose_k(candidates)
            selected_k = int(selected["k"])
            plot_variables[variable] = {
                "heads": selected["heads"],
                "k": selected_k,
                "coupling_row": frozen_coupling[VARIABLES.index(variable)].tolist(),
                "calibration_candidates": candidates,
            }
            selected_runtime_objects[variable] = (
                heads_by_k[selected_k],
                rotations_by_k[selected_k],
            )

        # Both coupling rows and both K values are frozen before any PLOT held-out access.
        _write_json(
            output_dir / "plot" / "frozen_selection.json",
            {
                "uot": {
                    "epsilon": uot_payload["selected"]["epsilon"],
                    "beta_neural": uot_payload["selected"]["beta_neural"],
                },
                "variables": {
                    variable: {
                        "k": plot_variables[variable]["k"],
                        "heads": plot_variables[variable]["heads"],
                    }
                    for variable in VARIABLES
                },
            },
        )
        for variable in VARIABLES:
            selected_heads, selected_rotations = selected_runtime_objects[variable]
            test_metrics, test_seconds = _evaluate_selected(
                model,
                tokenizer,
                banks,
                variable,
                causal_model,
                selected_heads,
                selected_rotations,
                device,
                args.eval_batch_size,
            )
            plot_test_seconds += test_seconds
            plot_variables[variable]["test"] = test_metrics
            plot_variables[variable]["test_runtime_seconds"] = test_seconds

        signature_seconds = signatures.runtime_seconds
        tuning_seconds = float(uot_payload.get("runtime_seconds", uot_seconds if not uot_cached else 0.0))
        cold = (
            signature_seconds
            + tuning_seconds
            + plot_training_cold_seconds
            + plot_calibration_seconds
            + plot_test_seconds
        )
        current = (
            (0.0 if signature_cached else signature_seconds)
            + (0.0 if uot_cached else tuning_seconds)
            + plot_training_current_seconds
            + plot_calibration_seconds
            + plot_test_seconds
        )
        method_results["plot"] = {
            "variables": plot_variables,
            "selected_uot": {
                "epsilon": uot_payload["selected"]["epsilon"],
                "beta_neural": uot_payload["selected"]["beta_neural"],
            },
            "signature_cached": signature_cached,
            "uot_cached": uot_cached,
            "runtime": {
                "signature_collection_seconds": signature_seconds,
                "uot_tuning_seconds": tuning_seconds,
                "das_training_seconds": plot_training_cold_seconds,
                "current_das_training_seconds": plot_training_current_seconds,
                "calibration_seconds": plot_calibration_seconds,
                "heldout_evaluation_seconds": plot_test_seconds,
                "cold_seconds": cold,
                "current_invocation_seconds": current,
                "amortized_seconds_per_retained_variable": cold / len(VARIABLES),
            },
        }

    # All data-dependent choices are now frozen. Fixed blind/oracle methods are
    # evaluated here as well so no held-out metrics exist during PLOT selection.
    for method, runtime_objects in deferred_fixed_evaluations.items():
        method_test_seconds = 0.0
        for variable in VARIABLES:
            heads, rotations = runtime_objects[variable]
            test_metrics, test_seconds = _evaluate_selected(
                model,
                tokenizer,
                banks,
                variable,
                causal_model,
                heads,
                rotations,
                device,
                args.eval_batch_size,
            )
            method_test_seconds += test_seconds
            method_results[method]["variables"][variable]["test"] = test_metrics
            method_results[method]["variables"][variable]["test_runtime_seconds"] = test_seconds
        runtime = method_results[method]["runtime"]
        runtime["heldout_evaluation_seconds"] = method_test_seconds
        runtime["cold_seconds"] = float(runtime["das_training_seconds"]) + method_test_seconds
        runtime["current_invocation_seconds"] = (
            float(runtime["current_das_training_seconds"]) + method_test_seconds
        )
        runtime["amortized_seconds_per_variable"] = runtime["cold_seconds"] / len(VARIABLES)

    summary = {
        "config": config,
        "config_hash": config_hash,
        "dataset_hash": _hash(banks.metadata),
        "banks": banks.metadata,
        "causal_model": regression_payload,
        "coefficient_absolute_differences": discrepancies,
        "shared_runtime": shared_runtime,
        "methods": method_results,
        "selection_protocol": {
            "fit_uses": ["causal regression", "PLOT signatures", "DAS rotations"],
            "calibration_uses": ["shared UOT setting", "PLOT-DAS K per variable"],
            "test_uses": ["final frozen evaluation only"],
        },
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.txt").write_text(_summary_text(summary))
    manifest = _read_json(manifest_path)
    manifest.update({"status": "complete", "summary": str(output_dir / "summary.json")})
    _write_json(manifest_path, manifest)
    print(_summary_text(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
