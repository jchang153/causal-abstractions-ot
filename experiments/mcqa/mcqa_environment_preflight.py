#!/usr/bin/env python3
"""Fail-fast validation for the Delta MCQA environment and gated artifacts."""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from time import time

import torch

from mcqa_dbm_baselines import load_sae, validate_sae_grid
from mcqa_experiment.data import load_filtered_mcqa_pipeline
from mcqa_experiment.dbm import SAEBasis


MODEL_NAME = "google/gemma-2-2b"
DATASET_PATH = "jchang153/copycolors_mcqa"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID_TEMPLATE = "layer_{layer}/width_16k/canonical"
REQUIRED_IMPORTS = (
    "accelerate",
    "datasets",
    "huggingface_hub",
    "numpy",
    "pyvene",
    "safetensors",
    "sae_lens",
    "scipy",
    "sentencepiece",
    "torch",
    "transformers",
)
REQUIRED_REPO_IMPORTS = (
    "mcqa_boundless_das",
    "mcqa_dbm_baselines",
    "mcqa_delta_hierarchical_sweep",
    "mcqa_ot_pca_focus",
    "mcqa_plot_das_pca_support",
    "mcqa_plot_layer",
    "mcqa_plot_native_support",
    "mcqa_run_cloud",
)
VERSION_DISTRIBUTIONS = (
    "accelerate",
    "datasets",
    "huggingface-hub",
    "numpy",
    "pyvene",
    "safetensors",
    "sae-lens",
    "scipy",
    "sentencepiece",
    "torch",
    "transformers",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--stamp", type=Path, required=True)
    parser.add_argument("--dataset-size", type=int, default=8)
    parser.add_argument("--sae-probe-layer", type=int, default=18)
    parser.add_argument("--prefetch-all-saes", action="store_true")
    return parser.parse_args()


def require_nvme(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    expected = Path(f"/work/nvme/bgvo/{os.environ.get('USER', '')}")
    try:
        resolved.relative_to(expected)
    except ValueError as error:
        raise RuntimeError(f"{label} must live below {expected}, got {resolved}") from error
    return resolved


def main() -> None:
    args = parse_args()
    if sys.version_info < (3, 10):
        raise RuntimeError(f"Python 3.10+ is required, got {sys.version}")
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        raise RuntimeError("Set HF_TOKEN before running the preflight")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run the bootstrap inside the active GPU allocation")

    cache_root = require_nvme(args.cache_root, label="cache root")
    stamp = require_nvme(args.stamp, label="preflight stamp")
    free_bytes = shutil.disk_usage(cache_root).free
    if free_bytes < 50 * 1024**3:
        raise RuntimeError(f"Less than 50 GiB is free below {cache_root}: {free_bytes / 1024**3:.1f} GiB")

    for module_name in REQUIRED_IMPORTS:
        importlib.import_module(module_name)
    for module_name in REQUIRED_REPO_IMPORTS:
        importlib.import_module(module_name)
    versions = {name: importlib.metadata.version(name) for name in VERSION_DISTRIBUTIONS}

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    identity = HfApi().whoami(token=token)
    print(f"[preflight] Hugging Face identity={identity.get('name', '<unknown>')}")

    # Exercise the exact model/data loading and factual-filtering path used by
    # every method, including one real CUDA forward pass.
    model, tokenizer, _, token_positions, filtered = load_filtered_mcqa_pipeline(
        model_name=MODEL_NAME,
        device="cuda",
        batch_size=4,
        dataset_size=int(args.dataset_size),
        hf_token=token,
        dataset_path=DATASET_PATH,
        dataset_name=None,
    )
    layer_count = int(model.config.num_hidden_layers)
    if int(model.config.hidden_size) != 2304 or layer_count != 26:
        raise RuntimeError(
            f"Unexpected {MODEL_NAME} shape: hidden={model.config.hidden_size}, layers={layer_count}"
        )
    if not filtered or not any(filtered.values()):
        raise RuntimeError("MCQA dataset loaded but factual filtering returned no rows")
    print(
        f"[preflight] model/data ok layers={layer_count} hidden={model.config.hidden_size} "
        f"token_positions={[position.id for position in token_positions]}"
    )
    del model, tokenizer, token_positions, filtered
    gc.collect()
    torch.cuda.empty_cache()

    # Resolve every layer alias before the long DBM sweep, then download and
    # execute one representative SAE to catch authentication/API/dtype issues.
    validate_sae_grid(
        layers=list(range(layer_count)),
        release=SAE_RELEASE,
        sae_id_template=SAE_ID_TEMPLATE,
    )
    if args.prefetch_all_saes:
        for layer in range(layer_count):
            prefetched_sae, _ = load_sae(
                layer=layer,
                release=SAE_RELEASE,
                sae_id_template=SAE_ID_TEMPLATE,
                device=torch.device("cpu"),
            )
            del prefetched_sae
            gc.collect()
        print(f"[preflight] downloaded and validated all {layer_count} SAE checkpoints")
    sae, sae_id = load_sae(
        layer=int(args.sae_probe_layer),
        release=SAE_RELEASE,
        sae_id_template=SAE_ID_TEMPLATE,
        device=torch.device("cuda"),
    )
    basis = SAEBasis(sae)
    dtype = next(sae.parameters()).dtype
    probe = torch.zeros((2, 2304), device="cuda", dtype=dtype)
    with torch.no_grad():
        reconstruction = basis.decode(basis.encode(probe))
    if reconstruction.shape != probe.shape or not torch.isfinite(reconstruction).all():
        raise RuntimeError(f"SAE encode/decode probe failed: shape={tuple(reconstruction.shape)}")
    print(f"[preflight] SAE ok id={sae_id} features={basis.feature_dim} dtype={dtype}")

    payload = {
        "status": "ok",
        "timestamp_unix": time(),
        "hostname": platform.node(),
        "python": sys.version,
        "executable": sys.executable,
        "cuda": {
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "versions": versions,
        "model": MODEL_NAME,
        "dataset": DATASET_PATH,
        "sae_release": SAE_RELEASE,
        "sae_probe_id": sae_id,
        "all_sae_checkpoints_prefetched": bool(args.prefetch_all_saes),
        "cache_root": str(cache_root),
        "free_gib": free_bytes / 1024**3,
    }
    stamp.parent.mkdir(parents=True, exist_ok=True)
    temporary = stamp.with_suffix(stamp.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(stamp)
    print(f"[preflight] PASS stamp={stamp}")


if __name__ == "__main__":
    main()
