#!/usr/bin/env python3
"""Run MIB-style DBM and full-layer interchange baselines on MCQA."""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import random
from time import perf_counter

import torch

from mcqa_experiment.data import build_pair_banks, load_filtered_mcqa_pipeline
from mcqa_experiment.dbm import (
    DBMMask,
    IdentityBasis,
    PCABasis,
    SAEBasis,
    collect_layer_observations,
    evaluate_dbm,
    evaluate_full_layer,
    train_dbm,
)
from mcqa_experiment.runtime import resolve_device


METHODS = ("full-layer", "dbm-canonical", "dbm-pca", "dbm-sae")
TARGETS = ("answer_pointer", "answer_token")


def parse_int_grid(value: str, *, upper_exclusive: int | None = None) -> list[int]:
    if value.strip().lower() == "all":
        if upper_exclusive is None:
            raise ValueError("'all' requires an upper bound")
        return list(range(int(upper_exclusive)))
    values: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            values.update(range(start, end + 1))
        else:
            values.add(int(part))
    resolved = sorted(values)
    if upper_exclusive is not None and any(item < 0 or item >= upper_exclusive for item in resolved):
        raise ValueError(f"Layer grid {resolved} exceeds [0, {upper_exclusive})")
    return resolved


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def load_sae(*, layer: int, release: str, sae_id_template: str, device: torch.device):
    try:
        from sae_lens import SAE
    except ImportError as error:
        raise RuntimeError("dbm-sae requires sae-lens; install requirements.txt") from error
    sae_id = sae_id_template.format(layer=int(layer))
    loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=str(device))
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    sae.eval()
    sae.requires_grad_(False)
    cfg_layer = getattr(sae.cfg, "hook_layer", None)
    if cfg_layer is not None and int(cfg_layer) != int(layer):
        raise ValueError(f"SAE hook_layer={cfg_layer} does not match requested layer={layer}")
    if int(getattr(sae.cfg, "d_in")) != 2304:
        raise ValueError(f"Expected Gemma-2-2B d_in=2304, got {getattr(sae.cfg, 'd_in', None)}")
    print(f"[SAE] loaded release={release} sae_id={sae_id} d_sae={getattr(sae.cfg, 'd_sae', None)}")
    return sae, sae_id


def validate_sae_grid(*, layers: list[int], release: str, sae_id_template: str) -> None:
    """Fail before the long sweep if any requested Gemma Scope alias is unavailable."""
    try:
        from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
    except ImportError as error:
        raise RuntimeError("dbm-sae requires sae-lens; install requirements.txt") from error
    directory = get_pretrained_saes_directory()
    if release not in directory:
        raise ValueError(f"Unknown SAE Lens release {release!r}")
    available = directory[release].saes_map
    requested = [sae_id_template.format(layer=int(layer)) for layer in layers]
    missing = [sae_id for sae_id in requested if sae_id not in available]
    if missing:
        raise ValueError(f"Missing SAE aliases in release {release}: {missing}")
    print(f"[SAE] preflight resolved {len(requested)} requested aliases from {release}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="google/gemma-2-2b")
    parser.add_argument("--dataset-path", default="jchang153/copycolors_mcqa")
    parser.add_argument("--dataset-config", default="none")
    parser.add_argument("--dataset-size", type=int, default=3000)
    parser.add_argument("--train-size", type=int, default=200)
    parser.add_argument("--calibration-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=0.01)
    parser.add_argument("--regularization-coefficient", type=float, default=0.0)
    parser.add_argument("--pca-rank", type=int, default=None)
    parser.add_argument("--sae-release", default="gemma-scope-2b-pt-res-canonical")
    parser.add_argument("--sae-id-template", default="layer_{layer}/width_16k/canonical")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--results-root", type=Path, default=Path("results/delta/mcqa_mib_baselines"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verify-sae-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    if args.device.startswith("cuda") and device.type != "cuda":
        raise RuntimeError("CUDA was requested but is not available; run inside the allocated srun step")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    run_name = args.run_name or f"mcqa_mib_baselines_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.results_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.verify_sae_only:
        sae, sae_id = load_sae(
            layer=18, release=args.sae_release, sae_id_template=args.sae_id_template, device=device
        )
        print(json.dumps({"status": "ok", "sae_id": sae_id, "d_in": int(sae.cfg.d_in), "d_sae": int(sae.cfg.d_sae)}))
        return

    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    targets = tuple(item.strip() for item in args.targets.split(",") if item.strip())
    unknown_methods = sorted(set(methods) - set(METHODS))
    unknown_targets = sorted(set(targets) - set(TARGETS))
    if unknown_methods or unknown_targets:
        raise ValueError(f"Unknown methods={unknown_methods} targets={unknown_targets}")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    model, tokenizer, causal_model, token_positions, filtered = load_filtered_mcqa_pipeline(
        model_name=args.model_name,
        device=str(device),
        batch_size=args.eval_batch_size,
        dataset_size=args.dataset_size,
        hf_token=hf_token,
        dataset_path=args.dataset_path,
        dataset_name=None if args.dataset_config.lower() == "none" else args.dataset_config,
    )
    model.eval()
    model.requires_grad_(False)
    layers = parse_int_grid(args.layers, upper_exclusive=int(model.config.num_hidden_layers))
    seeds = parse_int_grid(args.seeds)
    if "dbm-sae" in methods:
        validate_sae_grid(layers=layers, release=args.sae_release, sae_id_template=args.sae_id_template)
    jobs = [(method, seed, target, layer) for method in methods for seed in seeds for target in targets for layer in layers]
    jobs = [job for index, job in enumerate(jobs) if index % args.num_shards == args.shard_index]
    print(f"[plan] run_dir={run_dir.resolve()} jobs_on_shard={len(jobs)} shard={args.shard_index}/{args.num_shards}")
    atomic_json(
        run_dir / f"config_shard{args.shard_index}.json",
        {**vars(args), "results_root": str(args.results_root), "resolved_layers": layers},
    )

    banks_by_seed: dict[int, dict[str, dict[str, object]]] = {}
    for method, seed, target, layer in jobs:
        stem = f"{method}_seed{seed}_{target}_layer{layer}"
        output_path = run_dir / method / f"{stem}.json"
        if output_path.exists() and not args.no_resume:
            try:
                existing_payload = json.loads(output_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing_payload = None
            if isinstance(existing_payload, dict) and "calibration_candidate_seconds" in existing_payload:
                print(f"[resume] {output_path}")
                continue
            print(f"[rebuild] {output_path} uses the legacy every-layer test protocol")
        if seed not in banks_by_seed:
            banks, metadata = build_pair_banks(
                tokenizer=tokenizer,
                causal_model=causal_model,
                token_positions=token_positions,
                datasets_by_name=filtered,
                counterfactual_names=("answerPosition", "randomLetter", "answerPosition_randomLetter"),
                target_vars=targets,
                split_seed=int(seed),
                train_pool_size=args.train_size,
                calibration_pool_size=args.calibration_size,
                test_pool_size=args.test_size,
            )
            banks_by_seed[seed] = banks
            atomic_json(run_dir / f"data_seed{seed}_shard{args.shard_index}.json", metadata)
        banks = banks_by_seed[seed]
        started = perf_counter()
        payload: dict[str, object] = {
            "method": method, "seed": seed, "target_var": target, "layer": layer,
            "train_size": args.train_size, "calibration_size": args.calibration_size, "test_size": args.test_size,
        }
        if method == "full-layer":
            payload["calibration"] = evaluate_full_layer(
                model=model, bank=banks["calibration"][target], layer=layer, device=device,
                batch_size=args.eval_batch_size, tokenizer=tokenizer,
            )
        else:
            sae = None
            if method == "dbm-canonical":
                basis = IdentityBasis(int(model.config.hidden_size))
                basis_metadata = {"basis": "canonical", "feature_dim": int(model.config.hidden_size)}
            elif method == "dbm-pca":
                observations = collect_layer_observations(
                    model=model, bank=banks["train"][target], layer=layer,
                    batch_size=args.eval_batch_size, device=device,
                )
                basis = PCABasis.fit(observations, rank=args.pca_rank)
                basis_metadata = {"basis": "pca", "feature_dim": basis.feature_dim, "fit_observations": int(observations.shape[0])}
            else:
                sae, sae_id = load_sae(
                    layer=layer, release=args.sae_release, sae_id_template=args.sae_id_template, device=device
                )
                basis = SAEBasis(sae)
                basis_metadata = {"basis": "sae", "feature_dim": basis.feature_dim, "release": args.sae_release, "sae_id": sae_id}
            mask, training = train_dbm(
                model=model, train_bank=banks["train"][target], layer=layer, basis=basis, device=device,
                batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.learning_rate,
                temperature_start=args.temperature_start, temperature_end=args.temperature_end,
                regularization_coefficient=args.regularization_coefficient, seed=seed,
            )
            payload["basis"] = basis_metadata
            payload["training"] = training
            payload["calibration"] = evaluate_dbm(
                model=model, bank=banks["calibration"][target], layer=layer, basis=basis, mask=mask,
                device=device, batch_size=args.eval_batch_size, tokenizer=tokenizer,
            )
            checkpoint = run_dir / method / f"{stem}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_payload = {"mask_logits": mask.logits.detach().cpu(), "basis": basis_metadata}
            if isinstance(basis, PCABasis):
                checkpoint_payload["pca_components"] = basis.components.cpu()
            torch.save(checkpoint_payload, checkpoint)
            payload["checkpoint"] = str(checkpoint.relative_to(run_dir))
            del mask, basis, sae
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        payload["test_evaluated"] = False
        payload["result_split"] = "calibration"
        payload["calibration_candidate_seconds"] = float(perf_counter() - started)
        payload["total_seconds"] = float(payload["calibration_candidate_seconds"])
        atomic_json(output_path, payload)
        print(
            f"[done] {stem} cal={payload['calibration']['exact_acc']:.4f} "
            f"seconds={payload['total_seconds']:.1f}"
        )

    records = []
    for path in run_dir.glob("*/*.json"):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    def banks_for_seed(seed: int) -> dict[str, dict[str, object]]:
        if seed not in banks_by_seed:
            banks, metadata = build_pair_banks(
                tokenizer=tokenizer,
                causal_model=causal_model,
                token_positions=token_positions,
                datasets_by_name=filtered,
                counterfactual_names=("answerPosition", "randomLetter", "answerPosition_randomLetter"),
                target_vars=targets,
                split_seed=int(seed),
                train_pool_size=args.train_size,
                calibration_pool_size=args.calibration_size,
                test_pool_size=args.test_size,
            )
            banks_by_seed[seed] = banks
            atomic_json(run_dir / f"data_seed{seed}_shard{args.shard_index}.json", metadata)
        return banks_by_seed[seed]

    def evaluate_selected(record: dict[str, object]) -> dict[str, object]:
        selected = dict(record)
        if selected.get("test_evaluated") is True:
            return selected
        method = str(selected["method"])
        seed = int(selected["seed"])
        target = str(selected["target_var"])
        layer = int(selected["layer"])
        test_started = perf_counter()
        if method == "full-layer":
            test_metrics = evaluate_full_layer(
                model=model,
                bank=banks_for_seed(seed)["test"][target],
                layer=layer,
                device=device,
                batch_size=args.eval_batch_size,
                tokenizer=tokenizer,
            )
        else:
            checkpoint = torch.load(run_dir / str(selected["checkpoint"]), map_location="cpu", weights_only=True)
            if method == "dbm-canonical":
                basis = IdentityBasis(int(model.config.hidden_size))
                sae = None
            elif method == "dbm-pca":
                basis = PCABasis(components=checkpoint["pca_components"])
                sae = None
            else:
                sae, _ = load_sae(
                    layer=layer,
                    release=args.sae_release,
                    sae_id_template=args.sae_id_template,
                    device=device,
                )
                basis = SAEBasis(sae)
            mask = DBMMask(int(basis.feature_dim)).to(device)
            with torch.no_grad():
                mask.logits.copy_(checkpoint["mask_logits"].to(device=device, dtype=mask.logits.dtype))
            test_metrics = evaluate_dbm(
                model=model,
                bank=banks_for_seed(seed)["test"][target],
                layer=layer,
                basis=basis,
                mask=mask,
                device=device,
                batch_size=args.eval_batch_size,
                tokenizer=tokenizer,
            )
            del mask, basis, sae
        selected_test_seconds = perf_counter() - test_started
        selected["test"] = test_metrics
        selected["test_evaluated"] = True
        selected["result_split"] = "test"
        selected["selected_test_seconds"] = float(selected_test_seconds)
        return selected

    rankings: dict[str, object] = {}
    for method in methods:
        for seed in seeds:
            for target in targets:
                subset = [record for record in records if record.get("method") == method and record.get("seed") == seed and record.get("target_var") == target]
                if not subset:
                    continue
                subset.sort(key=lambda item: (-float(item["calibration"]["exact_acc"]), int(item["layer"])))
                selected = evaluate_selected(subset[0])
                selected_path = run_dir / method / f"{method}_seed{seed}_{target}_layer{int(selected['layer'])}.json"
                atomic_json(selected_path, selected)
                calibration_sweep_seconds = sum(float(item.get("calibration_candidate_seconds", item["total_seconds"])) for item in subset)
                reported_runtime_seconds = calibration_sweep_seconds + float(selected.get("selected_test_seconds", 0.0))
                rankings[f"{method}/seed{seed}/{target}"] = {
                    "calibration_selected": selected,
                    "all_layers": subset,
                    "selection_split": "calibration",
                    "test_used_for_selection": False,
                    "test_evaluation_policy": "selected_layer_only",
                    "calibration_layer_sweep_seconds": float(calibration_sweep_seconds),
                    "selected_test_seconds": float(selected.get("selected_test_seconds", 0.0)),
                    "runtime_seconds": float(reported_runtime_seconds),
                    "runtime_definition": "all-layer training/calibration sweep plus selected-layer test evaluation",
                }
    atomic_json(run_dir / f"rankings_shard{args.shard_index}.json", rankings)
    print(f"All results saved under: {run_dir.resolve()}")


if __name__ == "__main__":
    random.seed(0)
    main()
