from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from experiments.binary_addition_rnn.das import RotatedSubspace
from experiments.binary_addition_rnn.data import enumerate_all_examples, stratified_base_split
from experiments.binary_addition_rnn.interventions import RunCache, build_run_cache, intervene_with_site_handle_batch
from experiments.binary_addition_rnn.model import GRUAdder, exact_accuracy
from experiments.binary_addition_rnn.pca_basis import RotatedBasis, fit_pca_rotations
from experiments.binary_addition_rnn.run_joint_endogenous_resolution_sweep import (
    EndogenousPairRecord,
    EndogenousRowSpec,
    _bank_summaries,
    _build_banks,
    _family_order,
    _fit_cost_matrix,
    _load_or_train_model,
    _parse_selection_profiles,
    _partition_records,
    _row_specs,
    _subset_summary,
)
from experiments.binary_addition_rnn.sites import (
    CoordinateGroupSite,
    FullStateSite,
    RotatedGroupSite,
    Site,
    enumerate_group_sites_for_timesteps,
    enumerate_rotated_group_sites_for_timesteps,
    enumerate_rotated_prefix_sites_for_timesteps,
)
from experiments.binary_addition_rnn.transport import (
    TransportConfig,
    enumerate_transport_row_candidates,
    evaluate_single_calibrated_transport,
    select_transport_calibration_candidate,
    sinkhorn_uniform_ot,
)


@dataclass(frozen=True)
class DASSupport:
    name: str
    basis: str
    timestep: int
    indices: tuple[int, ...]

    def dim(self, hidden_size: int) -> int:
        if not self.indices:
            return int(hidden_size)
        return int(len(self.indices))

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Progressive PLOT pipeline for the GRU binary-addition benchmark.")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--hidden-size", type=int, required=True, choices=[8, 16])
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--checkpoint-map", type=str, default="")
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--rows", type=str, default="C1,C2,C3")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--fit-bases", type=int, default=128)
    ap.add_argument("--calib-bases", type=int, default=64)
    ap.add_argument("--test-bases", type=int, default=64)
    ap.add_argument("--train-on", type=str, default="all", choices=["all", "fit_only"])
    ap.add_argument("--train-epochs", type=int, default=120)
    ap.add_argument("--train-batch-size", type=int, default=64)
    ap.add_argument("--train-lr", type=float, default=0.02)
    ap.add_argument("--source-policy", type=str, default="structured_26_top3carry_c2x5_c3x7_no_random")
    ap.add_argument("--normalize-signatures", action="store_true", default=True)
    ap.add_argument("--fit-signature-mode", type=str, default="all")
    ap.add_argument("--fit-family-profile", type=str, default="all")
    ap.add_argument("--fit-stratify-mode", type=str, default="none")
    ap.add_argument("--cost-metric", type=str, default="sq_l2", choices=["sq_l2", "l1", "cosine"])
    ap.add_argument("--stage-a-epsilons", type=str, default="0.01,0.03")
    ap.add_argument("--stage-b-epsilons", type=str, default="0.003,0.01,0.03,0.1")
    ap.add_argument("--stage-a-top-k-grid", type=str, default="1")
    ap.add_argument("--stage-b-top-k-grid", type=str, default="1,2,4")
    ap.add_argument("--ot-lambda-grid", type=str, default="0.25,0.5,1,2,4,8")
    ap.add_argument("--sinkhorn-iters", type=int, default=80)
    ap.add_argument("--selection-rule", type=str, default="combined")
    ap.add_argument("--invariance-floor", type=float, default=0.0)
    ap.add_argument("--canonical-resolutions", type=str, default="")
    ap.add_argument("--pca-resolutions", type=str, default="")
    ap.add_argument("--pca-variant", type=str, default="centered", choices=["uncentered", "centered", "whitened"])
    ap.add_argument("--pca-site-menu", type=str, default="top_prefix", choices=["partition", "top_prefix", "both"])
    ap.add_argument("--canonical-mask-thresholds", type=str, default="0.8,0.9")
    ap.add_argument("--pca-prefix-dims", type=str, default="1,2,4")
    ap.add_argument("--no-full-support-fallback", action="store_true")
    ap.add_argument("--das-fit-bank-mode", type=str, default="anchored_prefix", choices=["shared", "anchored_prefix"])
    ap.add_argument("--das-subspace-dims", type=str, default="")
    ap.add_argument("--das-lrs", type=str, default="0.01,0.003")
    ap.add_argument("--das-epochs", type=int, default=12)
    ap.add_argument("--das-train-records-per-epoch", type=int, default=256)
    ap.add_argument("--das-batch-size", type=int, default=64)
    ap.add_argument("--skip-existing", action="store_true")
    return ap.parse_args()


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(text).split(",") if x.strip())


def _parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in str(text).split(",") if x.strip())


def _parse_rows(text: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in str(text).split(",") if x.strip())


def _checkpoint_map(hidden_size: int) -> dict[int, str]:
    if int(hidden_size) == 8:
        mapping = {
            0: str(ROOT / "eval" / "codex_ot_grid_h8_structured17_top2carry" / "checkpoints" / "gru_h8_seed0.pt"),
            1: str(ROOT / "eval" / "shared_checkpoints" / "gru_h8_seed1.pt"),
            2: str(ROOT / "eval" / "shared_checkpoints" / "gru_h8_seed2.pt"),
        }
        for seed in range(3, 10):
            mapping[seed] = str(ROOT / "eval" / "shared_checkpoints" / f"gru_h8_seed{seed}.pt")
        return mapping
    if int(hidden_size) == 16:
        mapping = {
            0: str(ROOT / "eval" / "codex_binary_backbone_h16_seed0" / "gru_adder.pt"),
            1: str(ROOT / "eval" / "shared_checkpoints" / "gru_h16_seed1.pt"),
            2: str(ROOT / "eval" / "shared_checkpoints" / "gru_h16_seed2.pt"),
        }
        for seed in range(3, 10):
            mapping[seed] = str(ROOT / "eval" / "shared_checkpoints" / f"gru_h16_seed{seed}.pt")
        return mapping
    raise ValueError(f"unsupported hidden_size: {hidden_size}")


def _parse_checkpoint_map(text: str, hidden_size: int) -> dict[int, str]:
    if not str(text).strip():
        return _checkpoint_map(hidden_size)
    mapping: dict[int, str] = {}
    for chunk in str(text).split(";"):
        if not chunk.strip():
            continue
        key, value = chunk.split("=", 1)
        mapping[int(key.strip())] = value.strip()
    return mapping


def _default_resolutions(hidden_size: int) -> tuple[int, ...]:
    if int(hidden_size) == 8:
        return (8, 4, 2, 1)
    if int(hidden_size) == 16:
        return (16, 8, 4, 2, 1)
    return (int(hidden_size), 1)


def _default_das_dims(hidden_size: int) -> tuple[int, ...]:
    if int(hidden_size) == 8:
        return (1, 2, 4, 8)
    if int(hidden_size) == 16:
        return (1, 2, 4, 8, 16)
    return tuple(dim for dim in (1, 2, 4, 8, 16) if dim <= int(hidden_size))


def _dedupe_sites(sites: Sequence[Site]) -> tuple[Site, ...]:
    out: dict[str, Site] = {}
    for site in sites:
        out.setdefault(site.key(), site)
    return tuple(out.values())


def _site_timestep(site_key: str) -> int | None:
    if not site_key.startswith("h_"):
        return None
    head = site_key.split("[", 1)[0]
    return int(head.split("_", 1)[1])


def _site_indices_from_key(site_key: str, *, hidden_size: int, basis: str) -> tuple[int, ...]:
    if "[" not in site_key:
        return tuple(range(int(hidden_size)))
    inner = site_key.split("[", 1)[1].rstrip("]")
    if basis == "pca":
        if ":" in inner:
            inner = inner.split(":", 1)[1]
    if not inner:
        return tuple()
    return tuple(int(part) for part in inner.split(",") if part)


def _row_normalize(row: torch.Tensor) -> torch.Tensor:
    return row / row.sum().clamp_min(1e-30)


def _restricted_row(row: torch.Tensor, sites: Sequence[Site], allowed_timesteps: set[int] | None) -> torch.Tensor:
    if allowed_timesteps is None:
        return _row_normalize(row)
    masked = row.clone()
    for idx, site in enumerate(sites):
        timestep = _site_timestep(site.key())
        if timestep is None or int(timestep) not in allowed_timesteps:
            masked[idx] = 0.0
    if float(masked.sum().item()) <= 0.0:
        return _row_normalize(row)
    return _row_normalize(masked)


def _selected_site_dicts(calibrated: dict[str, object], sites: Sequence[Site]) -> list[dict[str, object]]:
    return [
        {
            "site_index": int(site_rec["site_index"]),
            "site_key": sites[int(site_rec["site_index"])].key(),
            "weight": float(site_rec["weight"]),
        }
        for site_rec in calibrated.get("selected_sites", [])
    ]


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else float(sum(values) / len(values))


def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    return float(math.sqrt(sum((float(v) - mu) ** 2 for v in values) / (len(values) - 1)))


def _summary_stats(values: Sequence[float]) -> dict[str, float]:
    return {"mean": _mean(values), "std": _std(values)}


def _fit_records_for_row(
    records: Sequence[EndogenousPairRecord],
    *,
    row_key: str,
    fit_bank_mode: str,
) -> tuple[EndogenousPairRecord, ...]:
    if str(fit_bank_mode) == "shared":
        return tuple(records)
    if row_key.startswith("C"):
        carry_index = int(row_key[1:])
        prefixes = [f"flip_A{carry_index - 1}", f"flip_B{carry_index - 1}", f"target_C{carry_index}"]
        if carry_index > 1:
            prefixes.append(f"target_C{carry_index - 1}")
    elif row_key.startswith("S"):
        sum_index = int(row_key[1:])
        prefixes = [f"flip_A{sum_index}", f"flip_B{sum_index}"]
        if sum_index > 0:
            prefixes.append(f"target_C{sum_index}")
    else:
        prefixes = []
    return tuple(rec for rec in records if any(rec.family == prefix or rec.family.startswith(prefix + "_") for prefix in prefixes))


def _make_transport_config(
    *,
    epsilons: tuple[float, ...],
    top_k_grid: tuple[int, ...],
    lambda_grid: tuple[float, ...],
    sinkhorn_iters: int,
) -> TransportConfig:
    return TransportConfig(
        epsilon_grid=tuple(float(x) for x in epsilons),
        beta_grid=(0.1,),
        topk_grid=tuple(int(x) for x in top_k_grid),
        lambda_grid=tuple(float(x) for x in lambda_grid),
        sinkhorn_iters=int(sinkhorn_iters),
        temperature=1.0,
        invariance_floor=0.0,
    )


def _run_ot_stage(
    *,
    stage_name: str,
    model: GRUAdder,
    specs: Sequence[EndogenousRowSpec],
    row_keys: Sequence[str],
    banks: dict[str, object],
    sites: Sequence[Site],
    family_order: Sequence[str],
    transport_cfg: TransportConfig,
    selection_rule: str,
    invariance_floor: float,
    device: torch.device,
    run_cache: RunCache,
    batch_size: int,
    normalize_signatures: bool,
    fit_signature_mode: str,
    fit_stratify_mode: str,
    fit_family_profile: str,
    cost_metric: str,
    rotation_map: dict[int, RotatedBasis] | None,
    row_allowed_timesteps: dict[str, set[int]] | None = None,
) -> dict[str, object]:
    start_time = time.perf_counter()
    cost, diagnostics = _fit_cost_matrix(
        model,
        specs=specs,
        fit_by_row=banks["fit_by_row"],
        sites=sites,
        family_order=family_order,
        device=device,
        run_cache=run_cache,
        batch_size=int(batch_size),
        normalize_signatures=bool(normalize_signatures),
        fit_signature_mode=str(fit_signature_mode),
        fit_stratify_mode=str(fit_stratify_mode),
        fit_family_profile=str(fit_family_profile),
        cost_metric=str(cost_metric),
        rotation_map=rotation_map,
    )

    trials = []
    best_trial = None
    best_key = None
    for eps in transport_cfg.epsilon_grid:
        coupling = sinkhorn_uniform_ot(
            cost,
            epsilon=float(eps),
            n_iter=int(transport_cfg.sinkhorn_iters),
            temperature=float(transport_cfg.temperature),
        )
        per_row = {}
        calib_sens = []
        calib_inv = []
        test_sens = []
        test_inv = []
        for row_idx, row_key in enumerate(row_keys):
            row = coupling[row_idx]
            allowed = None if row_allowed_timesteps is None else row_allowed_timesteps.get(row_key)
            row_for_calib = _restricted_row(row, sites, allowed)
            candidates = enumerate_transport_row_candidates(
                model,
                row_for_calib,
                sites,
                banks["calib_positive_by_row"][row_key],
                banks["calib_invariant_by_row"][row_key],
                transport_cfg,
                device=device,
                run_cache=run_cache,
                rotation_map=rotation_map,
            )
            calibrated = select_transport_calibration_candidate(
                candidates,
                selection_rule=str(selection_rule),
                invariance_floor=float(invariance_floor),
            )
            tested = evaluate_single_calibrated_transport(
                model,
                calibrated,
                sites,
                banks["test_positive_by_row"][row_key],
                banks["test_invariant_by_row"][row_key],
                device=device,
                run_cache=run_cache,
                rotation_map=rotation_map,
            )
            calibrated = dict(calibrated)
            calibrated["selected_sites"] = _selected_site_dicts(calibrated, sites)
            tested = dict(tested)
            tested["selected_sites"] = _selected_site_dicts(calibrated, sites)
            per_row[row_key] = {
                "calibration": calibrated["calibration"],
                "test": tested,
                "top_k": int(calibrated["top_k"]),
                "lambda": float(calibrated["lambda"]),
                "selected_sites": calibrated["selected_sites"],
                "allowed_timesteps": None if allowed is None else sorted(int(t) for t in allowed),
                "row_mass": row_for_calib.tolist(),
            }
            calib_sens.append(float(calibrated["calibration"]["sensitivity"]))
            calib_inv.append(float(calibrated["calibration"]["invariance"]))
            test_sens.append(float(tested["sensitivity"]))
            test_inv.append(float(tested["invariance"]))
        calibration = {
            "mean_sensitivity": _mean(calib_sens),
            "mean_invariance": _mean(calib_inv),
            "mean_combined": 0.5 * (_mean(calib_sens) + _mean(calib_inv)),
        }
        test = {
            "per_row": per_row,
            "mean_sensitivity": _mean(test_sens),
            "mean_invariance": _mean(test_inv),
            "mean_combined": 0.5 * (_mean(test_sens) + _mean(test_inv)),
            "subset": _subset_summary({k: v["test"] for k, v in per_row.items()}, row_keys),
        }
        trial = {
            "epsilon": float(eps),
            "coupling": coupling.tolist(),
            "calibration": calibration,
            "test": test,
        }
        trials.append(trial)
        key = (float(calibration["mean_combined"]), float(calibration["mean_sensitivity"]), float(calibration["mean_invariance"]))
        if best_key is None or key > best_key:
            best_key = key
            best_trial = trial

    elapsed = time.perf_counter() - start_time
    return {
        "stage": str(stage_name),
        "runtime_seconds": float(elapsed),
        "sites": [site.key() for site in sites],
        "fit_diagnostics": diagnostics,
        "trials": trials,
        "best_trial": best_trial,
    }


def _stage_a_timesteps(stage_a: dict[str, object], row_keys: Sequence[str]) -> dict[str, int]:
    best = stage_a["best_trial"]
    out: dict[str, int] = {}
    for row_key in row_keys:
        row = best["test"]["per_row"][row_key]
        selected = list(row.get("selected_sites", []))
        selected = [site for site in selected if _site_timestep(str(site["site_key"])) is not None]
        if selected:
            best_site = max(selected, key=lambda item: (float(item["weight"]), -int(_site_timestep(str(item["site_key"])) or 0)))
            out[row_key] = int(_site_timestep(str(best_site["site_key"])))
            continue
        mass = torch.tensor(row["row_mass"], dtype=torch.float32)
        idx = int(torch.argmax(mass).item())
        out[row_key] = int(_site_timestep(stage_a["sites"][idx]) or 0)
    return out


def _pca_component_union(
    row_result: dict[str, object],
    *,
    hidden_size: int,
    timestep: int,
) -> tuple[int, ...]:
    components: set[int] = set()
    for site_rec in row_result.get("selected_sites", []):
        site_key = str(site_rec["site_key"])
        if _site_timestep(site_key) != int(timestep):
            continue
        components.update(_site_indices_from_key(site_key, hidden_size=hidden_size, basis="pca"))
    if not components:
        return (0,)
    return tuple(sorted(components))


def _canonical_site_union(
    row_result: dict[str, object],
    *,
    hidden_size: int,
    timestep: int,
) -> tuple[int, ...]:
    coords: set[int] = set()
    for site_rec in row_result.get("selected_sites", []):
        site_key = str(site_rec["site_key"])
        if _site_timestep(site_key) != int(timestep):
            continue
        coords.update(_site_indices_from_key(site_key, hidden_size=hidden_size, basis="canonical"))
    if not coords:
        return tuple(range(int(hidden_size)))
    return tuple(sorted(coords))


def _coordinate_evidence_from_row_mass(
    row_result: dict[str, object],
    sites: Sequence[str],
    *,
    hidden_size: int,
    timestep: int,
    basis: str,
) -> list[float]:
    evidence = [0.0 for _ in range(int(hidden_size))]
    row_mass = row_result.get("row_mass", [])
    for site_key, mass in zip(sites, row_mass):
        if _site_timestep(str(site_key)) != int(timestep):
            continue
        indices = _site_indices_from_key(str(site_key), hidden_size=int(hidden_size), basis=str(basis))
        if not indices:
            continue
        per_index = float(mass) / float(len(indices))
        for index in indices:
            evidence[int(index)] += per_index
    return evidence


def _threshold_indices(evidence: Sequence[float], *, threshold: float) -> tuple[int, ...]:
    total = float(sum(float(x) for x in evidence))
    if total <= 0.0:
        return tuple(range(len(evidence)))
    ranked = sorted(range(len(evidence)), key=lambda idx: (-float(evidence[idx]), int(idx)))
    chosen: list[int] = []
    running = 0.0
    for idx in ranked:
        chosen.append(int(idx))
        running += float(evidence[idx])
        if running / total >= float(threshold):
            break
    return tuple(sorted(chosen))


def _unique_supports(supports: Sequence[DASSupport]) -> tuple[DASSupport, ...]:
    seen: set[tuple[str, int, tuple[int, ...]]] = set()
    out: list[DASSupport] = []
    for support in supports:
        key = (str(support.basis), int(support.timestep), tuple(int(i) for i in support.indices))
        if key in seen:
            continue
        seen.add(key)
        out.append(support)
    return tuple(out)


def _support_chunk(
    h: torch.Tensor,
    support: DASSupport,
    *,
    hidden_size: int,
    rotation_map: dict[int, RotatedBasis] | None,
) -> torch.Tensor:
    indices = list(support.indices or tuple(range(int(hidden_size))))
    if support.basis == "canonical":
        return h[:, indices]
    if support.basis == "pca":
        if rotation_map is None or int(support.timestep) not in rotation_map:
            raise ValueError(f"missing PCA basis for timestep {support.timestep}")
        basis = rotation_map[int(support.timestep)]
        rotation = basis.rotation.to(device=h.device, dtype=h.dtype)
        mean = basis.mean.to(device=h.device, dtype=h.dtype)
        scale = basis.scale.to(device=h.device, dtype=h.dtype)
        z = ((h - mean.unsqueeze(0)) @ rotation) / scale.unsqueeze(0)
        return z[:, indices]
    raise ValueError(f"unknown support basis: {support.basis!r}")


def _replace_support_chunk(
    h: torch.Tensor,
    updated_chunk: torch.Tensor,
    support: DASSupport,
    *,
    hidden_size: int,
    rotation_map: dict[int, RotatedBasis] | None,
) -> torch.Tensor:
    indices = list(support.indices or tuple(range(int(hidden_size))))
    if support.basis == "canonical":
        out = h.clone()
        out[:, indices] = updated_chunk
        return out
    if support.basis == "pca":
        if rotation_map is None or int(support.timestep) not in rotation_map:
            raise ValueError(f"missing PCA basis for timestep {support.timestep}")
        basis = rotation_map[int(support.timestep)]
        rotation = basis.rotation.to(device=h.device, dtype=h.dtype)
        mean = basis.mean.to(device=h.device, dtype=h.dtype)
        scale = basis.scale.to(device=h.device, dtype=h.dtype)
        z_base = ((h - mean.unsqueeze(0)) @ rotation) / scale.unsqueeze(0)
        z_int = z_base.clone()
        z_int[:, indices] = updated_chunk
        return h + ((z_int - z_base) * scale.unsqueeze(0)) @ rotation.transpose(0, 1)
    raise ValueError(f"unknown support basis: {support.basis!r}")


def _run_das_support_batch(
    model: GRUAdder,
    records: Sequence[EndogenousPairRecord],
    support: DASSupport,
    *,
    lambda_scale: float,
    device: torch.device,
    run_cache: RunCache,
    rotator: RotatedSubspace,
    rotation_map: dict[int, RotatedBasis] | None,
) -> torch.Tensor:
    if not records:
        return torch.empty((0, model.width + 1), dtype=torch.float32)
    base_x = torch.cat([run_cache.get_input(rec.base) for rec in records], dim=0).to(device=device)
    source_states = torch.stack([run_cache.get_run(rec.source).hidden_states for rec in records], dim=0).to(device=device)
    h = torch.zeros(base_x.size(0), model.hidden_size, device=device, dtype=base_x.dtype)
    sum_logits = []
    for step in range(model.width):
        h = model.cell(base_x[:, step, :], h)
        if step == int(support.timestep):
            source_h = source_states[:, step, :].to(device=device, dtype=h.dtype)
            base_chunk = _support_chunk(h, support, hidden_size=model.hidden_size, rotation_map=rotation_map)
            source_chunk = _support_chunk(source_h, support, hidden_size=model.hidden_size, rotation_map=rotation_map)
            updated_chunk = rotator.intervene(base_chunk, source_chunk, lambda_scale=float(lambda_scale))
            h = _replace_support_chunk(
                h,
                updated_chunk,
                support,
                hidden_size=model.hidden_size,
                rotation_map=rotation_map,
            )
        sum_logits.append(model.sum_head(h))
    carry_logit = model.final_carry_head(h)
    return torch.cat(sum_logits + [carry_logit], dim=1)


def _train_das_rotator(
    model: GRUAdder,
    records: Sequence[EndogenousPairRecord],
    support: DASSupport,
    *,
    subspace_dim: int,
    lr: float,
    epochs: int,
    train_records_per_epoch: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    run_cache: RunCache,
    rotation_map: dict[int, RotatedBasis] | None,
) -> RotatedSubspace:
    rotator = RotatedSubspace(support.dim(model.hidden_size), subspace_dim=int(subspace_dim)).to(device)
    optimizer = torch.optim.Adam(rotator.parameters(), lr=float(lr))
    criterion = nn.BCEWithLogitsLoss()
    rng = random.Random(int(seed) + int(subspace_dim) * 1009 + int(round(float(lr) * 1e6)))
    records = list(records)
    for _ in range(int(epochs)):
        sampled = records if len(records) <= int(train_records_per_epoch) else rng.sample(records, int(train_records_per_epoch))
        for start in range(0, len(sampled), int(batch_size)):
            batch = sampled[start : start + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            logits = _run_das_support_batch(
                model,
                batch,
                support,
                lambda_scale=1.0,
                device=device,
                run_cache=run_cache,
                rotator=rotator,
                rotation_map=rotation_map,
            )
            target = torch.tensor([rec.counterfactual.output_bits_lsb for rec in batch], dtype=torch.float32, device=device)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
    return rotator.cpu()


def _das_exact_match_rate(
    model: GRUAdder,
    records: Sequence[EndogenousPairRecord],
    support: DASSupport,
    *,
    lambda_scale: float,
    device: torch.device,
    run_cache: RunCache,
    rotator: RotatedSubspace,
    rotation_map: dict[int, RotatedBasis] | None,
    batch_size: int,
) -> float:
    if not records:
        return 0.0
    hits = 0
    rotator = rotator.to(device)
    for start in range(0, len(records), int(batch_size)):
        batch = records[start : start + int(batch_size)]
        logits = _run_das_support_batch(
            model,
            batch,
            support,
            lambda_scale=float(lambda_scale),
            device=device,
            run_cache=run_cache,
            rotator=rotator,
            rotation_map=rotation_map,
        )
        pred = (torch.sigmoid(logits) >= 0.5).to(torch.int64).cpu()
        tgt = torch.tensor([rec.counterfactual.output_bits_lsb for rec in batch], dtype=torch.int64)
        hits += int((pred == tgt).all(dim=1).sum().item())
    return float(hits / len(records))


def _calibration_key(*, combined: float, sensitivity: float, invariance: float, selection_rule: str, invariance_floor: float) -> tuple[float, float, float]:
    admissible = 1.0 if float(invariance) >= float(invariance_floor) else 0.0
    if selection_rule == "combined":
        return (admissible, float(combined), float(sensitivity))
    if selection_rule == "sensitivity_only":
        return (1.0, float(sensitivity), float(invariance))
    if selection_rule == "sensitivity_then_invariance":
        return (admissible, float(sensitivity), float(invariance))
    raise ValueError(f"unknown selection_rule: {selection_rule!r}")


def _run_das_stage(
    *,
    stage_name: str,
    model: GRUAdder,
    row_keys: Sequence[str],
    banks: dict[str, object],
    support_menu: dict[str, Sequence[DASSupport]],
    fit_bank_mode: str,
    subspace_dims: tuple[int, ...],
    learning_rates: tuple[float, ...],
    lambda_grid: tuple[float, ...],
    epochs: int,
    train_records_per_epoch: int,
    batch_size: int,
    selection_rule: str,
    invariance_floor: float,
    seed: int,
    device: torch.device,
    run_cache: RunCache,
    rotation_map: dict[int, RotatedBasis] | None,
) -> dict[str, object]:
    start_time = time.perf_counter()
    per_row = {}
    all_trials = []
    calib_sens = []
    calib_inv = []
    test_sens = []
    test_inv = []
    for row_key in row_keys:
        fit_records = _fit_records_for_row(
            tuple(banks["fit_by_row"][row_key]),
            row_key=row_key,
            fit_bank_mode=str(fit_bank_mode),
        )
        row_best = None
        row_best_key = None
        for support in support_menu[row_key]:
            valid_dims = tuple(dim for dim in subspace_dims if 1 <= int(dim) <= int(support.dim(model.hidden_size)))
            for subspace_dim in valid_dims:
                for lr in learning_rates:
                    rotator = _train_das_rotator(
                        model,
                        fit_records,
                        support,
                        subspace_dim=int(subspace_dim),
                        lr=float(lr),
                        epochs=int(epochs),
                        train_records_per_epoch=int(train_records_per_epoch),
                        batch_size=int(batch_size),
                        seed=int(seed),
                        device=device,
                        run_cache=run_cache,
                        rotation_map=rotation_map,
                    )
                    for lambda_scale in lambda_grid:
                        sens = _das_exact_match_rate(
                            model,
                            banks["calib_positive_by_row"][row_key],
                            support,
                            lambda_scale=float(lambda_scale),
                            device=device,
                            run_cache=run_cache,
                            rotator=rotator,
                            rotation_map=rotation_map,
                            batch_size=int(batch_size),
                        )
                        inv = _das_exact_match_rate(
                            model,
                            banks["calib_invariant_by_row"][row_key],
                            support,
                            lambda_scale=float(lambda_scale),
                            device=device,
                            run_cache=run_cache,
                            rotator=rotator,
                            rotation_map=rotation_map,
                            batch_size=int(batch_size),
                        )
                        combined = 0.5 * (sens + inv)
                        key = _calibration_key(
                            combined=combined,
                            sensitivity=sens,
                            invariance=inv,
                            selection_rule=str(selection_rule),
                            invariance_floor=float(invariance_floor),
                        )
                        trial = {
                            "row_key": row_key,
                            "support": support.as_dict(),
                            "subspace_dim": int(subspace_dim),
                            "lr": float(lr),
                            "lambda": float(lambda_scale),
                            "calibration": {
                                "sensitivity": float(sens),
                                "invariance": float(inv),
                                "combined": float(combined),
                                "count_positive": int(len(banks["calib_positive_by_row"][row_key])),
                                "count_invariant": int(len(banks["calib_invariant_by_row"][row_key])),
                            },
                            "rotator_state": {k: v.tolist() for k, v in rotator.state_dict().items()},
                        }
                        all_trials.append(trial)
                        if row_best_key is None or key > row_best_key:
                            row_best_key = key
                            row_best = trial
        if row_best is None:
            raise RuntimeError(f"no DAS trial generated for {row_key}")
        support = DASSupport(**row_best["support"])
        rotator = RotatedSubspace(support.dim(model.hidden_size), int(row_best["subspace_dim"]))
        rotator.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in row_best["rotator_state"].items()})
        sens = _das_exact_match_rate(
            model,
            banks["test_positive_by_row"][row_key],
            support,
            lambda_scale=float(row_best["lambda"]),
            device=device,
            run_cache=run_cache,
            rotator=rotator,
            rotation_map=rotation_map,
            batch_size=int(batch_size),
        )
        inv = _das_exact_match_rate(
            model,
            banks["test_invariant_by_row"][row_key],
            support,
            lambda_scale=float(row_best["lambda"]),
            device=device,
            run_cache=run_cache,
            rotator=rotator,
            rotation_map=rotation_map,
            batch_size=int(batch_size),
        )
        test_eval = {
            "sensitivity": float(sens),
            "invariance": float(inv),
            "combined": float(0.5 * (sens + inv)),
            "count_positive": int(len(banks["test_positive_by_row"][row_key])),
            "count_invariant": int(len(banks["test_invariant_by_row"][row_key])),
        }
        per_row[row_key] = {
            "calibration": row_best["calibration"],
            "test": test_eval,
            "support": support.as_dict(),
            "subspace_dim": int(row_best["subspace_dim"]),
            "lambda": float(row_best["lambda"]),
            "lr": float(row_best["lr"]),
        }
        calib_sens.append(float(row_best["calibration"]["sensitivity"]))
        calib_inv.append(float(row_best["calibration"]["invariance"]))
        test_sens.append(float(test_eval["sensitivity"]))
        test_inv.append(float(test_eval["invariance"]))
    elapsed = time.perf_counter() - start_time
    return {
        "stage": str(stage_name),
        "runtime_seconds": float(elapsed),
        "trials": all_trials,
        "calibration": {
            "mean_sensitivity": _mean(calib_sens),
            "mean_invariance": _mean(calib_inv),
            "mean_combined": 0.5 * (_mean(calib_sens) + _mean(calib_inv)),
        },
        "test": {
            "per_row": per_row,
            "mean_sensitivity": _mean(test_sens),
            "mean_invariance": _mean(test_inv),
            "mean_combined": 0.5 * (_mean(test_sens) + _mean(test_inv)),
            "subset": _subset_summary({k: v["test"] for k, v in per_row.items()}, row_keys),
        },
    }


def _build_rotated_sites(
    *,
    timesteps: tuple[int, ...],
    hidden_size: int,
    resolutions: tuple[int, ...],
    site_menu: str,
) -> tuple[Site, ...]:
    sites: list[Site] = []
    for resolution in resolutions:
        if str(site_menu) in {"partition", "both"}:
            sites.extend(
                enumerate_rotated_group_sites_for_timesteps(
                    timesteps=timesteps,
                    hidden_size=int(hidden_size),
                    resolution=int(resolution),
                    basis_name="pca",
                )
            )
        if str(site_menu) in {"top_prefix", "both"}:
            sites.extend(
                enumerate_rotated_prefix_sites_for_timesteps(
                    timesteps=timesteps,
                    hidden_size=int(hidden_size),
                    resolution=int(resolution),
                    basis_name="pca",
                )
            )
    return _dedupe_sites(tuple(sites))


def _method_record(
    *,
    name: str,
    accuracy_source: dict[str, object],
    runtime_seconds: float,
    row_keys: Sequence[str],
) -> dict[str, object]:
    test = accuracy_source["test"]
    return {
        "method": str(name),
        "runtime_seconds": float(runtime_seconds),
        "mean_sensitivity": float(test["subset"]["mean_sensitivity"]),
        "mean_invariance": float(test["subset"]["mean_invariance"]),
        "mean_combined": float(test["subset"]["mean_combined"]),
        "per_row": {
            row_key: test["per_row"][row_key]["test"] if "test" in test["per_row"][row_key] else test["per_row"][row_key]
            for row_key in row_keys
        },
    }


def _run_one_seed(args: argparse.Namespace, *, seed: int, checkpoint: str, out_dir: Path) -> dict[str, object]:
    seed_dir = out_dir / f"h{args.hidden_size}" / f"seed_{seed}"
    summary_path = seed_dir / "progressive_seed_summary.json"
    if bool(args.skip_existing) and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    seed_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    row_keys = _parse_rows(args.rows)
    examples = enumerate_all_examples(width=int(args.width))
    split = stratified_base_split(
        examples,
        fit_count=int(args.fit_bases),
        calib_count=int(args.calib_bases),
        test_count=int(args.test_bases),
        seed=int(seed),
    )

    model_args = argparse.Namespace(**vars(args))
    model_args.seed = int(seed)
    model_args.model_checkpoint = str(checkpoint)
    model, train_summary = _load_or_train_model(model_args, examples, split)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    run_cache = build_run_cache(model, examples, device=device)

    specs_all = _row_specs("all_endogenous", int(args.width))
    specs = tuple(spec for spec in specs_all if spec.key in set(row_keys))
    banks = _build_banks(
        split,
        specs,
        width=int(args.width),
        seed=int(seed),
        source_policy=str(args.source_policy),
        all_examples=examples,
    )
    family_order = _family_order(int(args.width), str(args.source_policy))

    transport_stage_a = _make_transport_config(
        epsilons=_parse_floats(args.stage_a_epsilons),
        top_k_grid=_parse_ints(args.stage_a_top_k_grid),
        lambda_grid=_parse_floats(args.ot_lambda_grid),
        sinkhorn_iters=int(args.sinkhorn_iters),
    )
    transport_stage_b = _make_transport_config(
        epsilons=_parse_floats(args.stage_b_epsilons),
        top_k_grid=_parse_ints(args.stage_b_top_k_grid),
        lambda_grid=_parse_floats(args.ot_lambda_grid),
        sinkhorn_iters=int(args.sinkhorn_iters),
    )

    full_state_sites = tuple(FullStateSite(timestep=t) for t in range(int(args.width)))
    stage_a = _run_ot_stage(
        stage_name="stage_a_full_timestep_ot",
        model=model,
        specs=specs,
        row_keys=row_keys,
        banks=banks,
        sites=full_state_sites,
        family_order=family_order,
        transport_cfg=transport_stage_a,
        selection_rule=str(args.selection_rule),
        invariance_floor=float(args.invariance_floor),
        device=device,
        run_cache=run_cache,
        batch_size=int(args.das_batch_size),
        normalize_signatures=bool(args.normalize_signatures),
        fit_signature_mode=str(args.fit_signature_mode),
        fit_stratify_mode=str(args.fit_stratify_mode),
        fit_family_profile=str(args.fit_family_profile),
        cost_metric=str(args.cost_metric),
        rotation_map=None,
    )
    stage_a_timesteps = _stage_a_timesteps(stage_a, row_keys)
    row_allowed_timesteps = {row_key: {int(stage_a_timesteps[row_key])} for row_key in row_keys}
    selected_timesteps = tuple(sorted(set(stage_a_timesteps.values())))

    canonical_resolutions = _parse_ints(args.canonical_resolutions) if str(args.canonical_resolutions).strip() else _default_resolutions(int(args.hidden_size))
    canonical_sites: list[Site] = []
    for resolution in canonical_resolutions:
        canonical_sites.extend(
            enumerate_group_sites_for_timesteps(
                timesteps=selected_timesteps,
                hidden_size=int(args.hidden_size),
                resolution=int(resolution),
            )
        )
    stage_b_canonical = _run_ot_stage(
        stage_name="stage_b_canonical_ot_inside_stage_a_timesteps",
        model=model,
        specs=specs,
        row_keys=row_keys,
        banks=banks,
        sites=_dedupe_sites(tuple(canonical_sites)),
        family_order=family_order,
        transport_cfg=transport_stage_b,
        selection_rule=str(args.selection_rule),
        invariance_floor=float(args.invariance_floor),
        device=device,
        run_cache=run_cache,
        batch_size=int(args.das_batch_size),
        normalize_signatures=bool(args.normalize_signatures),
        fit_signature_mode=str(args.fit_signature_mode),
        fit_stratify_mode=str(args.fit_stratify_mode),
        fit_family_profile=str(args.fit_family_profile),
        cost_metric=str(args.cost_metric),
        rotation_map=None,
        row_allowed_timesteps=row_allowed_timesteps,
    )

    pca_start = time.perf_counter()
    rotation_map, pca_diagnostics = fit_pca_rotations(
        fit_examples=split.fit,
        run_cache=run_cache,
        width=int(args.width),
        hidden_size=int(args.hidden_size),
        variant=str(args.pca_variant),
    )
    pca_fit_seconds = time.perf_counter() - pca_start

    pca_resolutions = _parse_ints(args.pca_resolutions) if str(args.pca_resolutions).strip() else _default_resolutions(int(args.hidden_size))
    pca_sites = _build_rotated_sites(
        timesteps=selected_timesteps,
        hidden_size=int(args.hidden_size),
        resolutions=pca_resolutions,
        site_menu=str(args.pca_site_menu),
    )
    stage_b_pca = _run_ot_stage(
        stage_name="stage_b_pca_ot_inside_stage_a_timesteps",
        model=model,
        specs=specs,
        row_keys=row_keys,
        banks=banks,
        sites=pca_sites,
        family_order=family_order,
        transport_cfg=transport_stage_b,
        selection_rule=str(args.selection_rule),
        invariance_floor=float(args.invariance_floor),
        device=device,
        run_cache=run_cache,
        batch_size=int(args.das_batch_size),
        normalize_signatures=bool(args.normalize_signatures),
        fit_signature_mode=str(args.fit_signature_mode),
        fit_stratify_mode=str(args.fit_stratify_mode),
        fit_family_profile=str(args.fit_family_profile),
        cost_metric=str(args.cost_metric),
        rotation_map=rotation_map,
        row_allowed_timesteps=row_allowed_timesteps,
    )

    full_timestep_supports = {
        row_key: (
            DASSupport(
                name=f"{row_key}_stage_a_full_h{stage_a_timesteps[row_key]}",
                basis="canonical",
                timestep=int(stage_a_timesteps[row_key]),
                indices=tuple(range(int(args.hidden_size))),
            ),
        )
        for row_key in row_keys
    }
    pca_supports = {}
    canonical_ot_supports = {}
    canonical_thresholds = _parse_floats(args.canonical_mask_thresholds)
    pca_prefix_dims = tuple(dim for dim in _parse_ints(args.pca_prefix_dims) if 1 <= int(dim) <= int(args.hidden_size))
    include_full_fallback = not bool(args.no_full_support_fallback)
    for row_key in row_keys:
        timestep = int(stage_a_timesteps[row_key])
        pca_components = _pca_component_union(
            stage_b_pca["best_trial"]["test"]["per_row"][row_key],
            hidden_size=int(args.hidden_size),
            timestep=timestep,
        )
        pca_menu = [
            DASSupport(
                name=f"{row_key}_pca_ot_selected_h{timestep}",
                basis="pca",
                timestep=timestep,
                indices=pca_components,
            )
        ]
        for dim in pca_prefix_dims:
            pca_menu.append(
                DASSupport(
                    name=f"{row_key}_pca_top{int(dim)}_h{timestep}",
                    basis="pca",
                    timestep=timestep,
                    indices=tuple(range(int(dim))),
                )
            )
        if include_full_fallback:
            pca_menu.append(
                DASSupport(
                    name=f"{row_key}_pca_full_h{timestep}",
                    basis="pca",
                    timestep=timestep,
                    indices=tuple(range(int(args.hidden_size))),
                )
            )
        pca_supports[row_key] = _unique_supports(tuple(pca_menu))

        canonical_coords = _canonical_site_union(
            stage_b_canonical["best_trial"]["test"]["per_row"][row_key],
            hidden_size=int(args.hidden_size),
            timestep=timestep,
        )
        canonical_menu = [
            DASSupport(
                name=f"{row_key}_canonical_ot_selected_h{timestep}",
                basis="canonical",
                timestep=timestep,
                indices=canonical_coords,
            )
        ]
        canonical_evidence = _coordinate_evidence_from_row_mass(
            stage_b_canonical["best_trial"]["test"]["per_row"][row_key],
            stage_b_canonical["sites"],
            hidden_size=int(args.hidden_size),
            timestep=timestep,
            basis="canonical",
        )
        for threshold in canonical_thresholds:
            canonical_menu.append(
                DASSupport(
                    name=f"{row_key}_canonical_s{int(round(float(threshold) * 100))}_h{timestep}",
                    basis="canonical",
                    timestep=timestep,
                    indices=_threshold_indices(canonical_evidence, threshold=float(threshold)),
                )
            )
        if include_full_fallback:
            canonical_menu.append(
                DASSupport(
                    name=f"{row_key}_canonical_full_h{timestep}",
                    basis="canonical",
                    timestep=timestep,
                    indices=tuple(range(int(args.hidden_size))),
                )
            )
        canonical_ot_supports[row_key] = _unique_supports(tuple(canonical_menu))

    das_dims = _parse_ints(args.das_subspace_dims) if str(args.das_subspace_dims).strip() else _default_das_dims(int(args.hidden_size))
    das_lrs = _parse_floats(args.das_lrs)
    das_lambda_grid = _parse_floats(args.ot_lambda_grid)
    stage_b_das_full = _run_das_stage(
        stage_name="stage_b_das_full_stage_a_timestep",
        model=model,
        row_keys=row_keys,
        banks=banks,
        support_menu=full_timestep_supports,
        fit_bank_mode=str(args.das_fit_bank_mode),
        subspace_dims=das_dims,
        learning_rates=das_lrs,
        lambda_grid=das_lambda_grid,
        epochs=int(args.das_epochs),
        train_records_per_epoch=int(args.das_train_records_per_epoch),
        batch_size=int(args.das_batch_size),
        selection_rule=str(args.selection_rule),
        invariance_floor=float(args.invariance_floor),
        seed=int(seed),
        device=device,
        run_cache=run_cache,
        rotation_map=None,
    )
    stage_b_das_canonical = _run_das_stage(
        stage_name="stage_b_das_inside_canonical_ot_support",
        model=model,
        row_keys=row_keys,
        banks=banks,
        support_menu=canonical_ot_supports,
        fit_bank_mode=str(args.das_fit_bank_mode),
        subspace_dims=das_dims,
        learning_rates=das_lrs,
        lambda_grid=das_lambda_grid,
        epochs=int(args.das_epochs),
        train_records_per_epoch=int(args.das_train_records_per_epoch),
        batch_size=int(args.das_batch_size),
        selection_rule=str(args.selection_rule),
        invariance_floor=float(args.invariance_floor),
        seed=int(seed),
        device=device,
        run_cache=run_cache,
        rotation_map=None,
    )
    stage_b_das_pca = _run_das_stage(
        stage_name="stage_b_das_inside_pca_ot_support",
        model=model,
        row_keys=row_keys,
        banks=banks,
        support_menu=pca_supports,
        fit_bank_mode=str(args.das_fit_bank_mode),
        subspace_dims=das_dims,
        learning_rates=das_lrs,
        lambda_grid=das_lambda_grid,
        epochs=int(args.das_epochs),
        train_records_per_epoch=int(args.das_train_records_per_epoch),
        batch_size=int(args.das_batch_size),
        selection_rule=str(args.selection_rule),
        invariance_floor=float(args.invariance_floor),
        seed=int(seed),
        device=device,
        run_cache=run_cache,
        rotation_map=rotation_map,
    )
    full_das_supports = {
        row_key: tuple(
            DASSupport(
                name=f"{row_key}_full_das_h{t}",
                basis="canonical",
                timestep=int(t),
                indices=tuple(range(int(args.hidden_size))),
            )
            for t in range(int(args.width))
        )
        for row_key in row_keys
    }
    full_das = _run_das_stage(
        stage_name="full_das_all_timesteps",
        model=model,
        row_keys=row_keys,
        banks=banks,
        support_menu=full_das_supports,
        fit_bank_mode=str(args.das_fit_bank_mode),
        subspace_dims=das_dims,
        learning_rates=das_lrs,
        lambda_grid=das_lambda_grid,
        epochs=int(args.das_epochs),
        train_records_per_epoch=int(args.das_train_records_per_epoch),
        batch_size=int(args.das_batch_size),
        selection_rule=str(args.selection_rule),
        invariance_floor=float(args.invariance_floor),
        seed=int(seed),
        device=device,
        run_cache=run_cache,
        rotation_map=None,
    )

    methods = {
        "stage_a_ot": _method_record(
            name="PLOT Stage-A OT",
            accuracy_source=stage_a["best_trial"],
            runtime_seconds=float(stage_a["runtime_seconds"]),
            row_keys=row_keys,
        ),
        "plot_in_timestep": _method_record(
            name="PLOT in timestep",
            accuracy_source=stage_b_canonical["best_trial"],
            runtime_seconds=float(stage_a["runtime_seconds"]) + float(stage_b_canonical["runtime_seconds"]),
            row_keys=row_keys,
        ),
        "plot_pca_in_timestep": _method_record(
            name="PLOT-PCA in timestep",
            accuracy_source=stage_b_pca["best_trial"],
            runtime_seconds=float(stage_a["runtime_seconds"]) + float(pca_fit_seconds) + float(stage_b_pca["runtime_seconds"]),
            row_keys=row_keys,
        ),
        "plot_guided_das_full_timestep": _method_record(
            name="PLOT-guided DAS full timestep",
            accuracy_source=stage_b_das_full,
            runtime_seconds=float(stage_a["runtime_seconds"]) + float(stage_b_das_full["runtime_seconds"]),
            row_keys=row_keys,
        ),
        "plot_guided_das_canonical_support": _method_record(
            name="PLOT-guided DAS canonical support",
            accuracy_source=stage_b_das_canonical,
            runtime_seconds=float(stage_a["runtime_seconds"]) + float(stage_b_canonical["runtime_seconds"]) + float(stage_b_das_canonical["runtime_seconds"]),
            row_keys=row_keys,
        ),
        "plot_pca_guided_das": _method_record(
            name="PLOT-PCA-guided DAS",
            accuracy_source=stage_b_das_pca,
            runtime_seconds=float(stage_a["runtime_seconds"]) + float(pca_fit_seconds) + float(stage_b_pca["runtime_seconds"]) + float(stage_b_das_pca["runtime_seconds"]),
            row_keys=row_keys,
        ),
        "full_das": _method_record(
            name="Full DAS",
            accuracy_source=full_das,
            runtime_seconds=float(full_das["runtime_seconds"]),
            row_keys=row_keys,
        ),
    }

    stage_a_expected = {f"C{i}": i - 1 for i in range(1, int(args.width))}
    stage_a_hits = [
        int(stage_a_timesteps[row_key] == stage_a_expected[row_key])
        for row_key in row_keys
        if row_key in stage_a_expected
    ]

    result = {
        "config": {**vars(args), "seed": int(seed), "checkpoint": str(checkpoint)},
        "factual_exact": {
            "all": exact_accuracy(model, examples, device=device),
            "fit": exact_accuracy(model, split.fit, device=device),
            "calib": exact_accuracy(model, split.calib, device=device),
            "test": exact_accuracy(model, split.test, device=device),
        },
        "training": train_summary,
        "bank_summaries": {
            "fit_by_row": _bank_summaries(
                banks["fit_by_row"],
                _partition_records(banks["fit_by_row"], active=True),
                _partition_records(banks["fit_by_row"], active=False),
            ),
            "calib_by_row": _bank_summaries(
                {key: tuple(banks["calib_positive_by_row"][key] + banks["calib_invariant_by_row"][key]) for key in row_keys},
                banks["calib_positive_by_row"],
                banks["calib_invariant_by_row"],
            ),
            "test_by_row": _bank_summaries(
                {key: tuple(banks["test_positive_by_row"][key] + banks["test_invariant_by_row"][key]) for key in row_keys},
                banks["test_positive_by_row"],
                banks["test_invariant_by_row"],
            ),
        },
        "stage_a_timesteps": stage_a_timesteps,
        "stage_a_expected_accuracy": _mean(stage_a_hits),
        "pca_fit_seconds": float(pca_fit_seconds),
        "pca_diagnostics": pca_diagnostics,
        "pca_supports": {row_key: [support.as_dict() for support in pca_supports[row_key]] for row_key in row_keys},
        "canonical_ot_supports": {row_key: [support.as_dict() for support in canonical_ot_supports[row_key]] for row_key in row_keys},
        "stages": {
            "stage_a": stage_a,
            "stage_b_canonical_ot": stage_b_canonical,
            "stage_b_pca_ot": stage_b_pca,
            "stage_b_das_full_timestep": stage_b_das_full,
            "stage_b_das_canonical_support": stage_b_das_canonical,
            "stage_b_das_pca_support": stage_b_das_pca,
            "full_das": full_das,
        },
        "methods": methods,
    }
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_ints(args.seeds)
    checkpoint_map = _parse_checkpoint_map(args.checkpoint_map, int(args.hidden_size))
    for seed in seeds:
        if int(seed) not in checkpoint_map:
            raise ValueError(f"missing checkpoint for seed {seed}")

    per_seed = {}
    for seed in seeds:
        result = _run_one_seed(args, seed=int(seed), checkpoint=checkpoint_map[int(seed)], out_dir=out_dir)
        per_seed[str(seed)] = {
            "summary_path": str(out_dir / f"h{args.hidden_size}" / f"seed_{seed}" / "progressive_seed_summary.json"),
            "factual_exact": result["factual_exact"],
            "stage_a_timesteps": result["stage_a_timesteps"],
            "stage_a_expected_accuracy": result["stage_a_expected_accuracy"],
            "pca_supports": result["pca_supports"],
            "canonical_ot_supports": result["canonical_ot_supports"],
            "methods": result["methods"],
        }

    method_keys = list(next(iter(per_seed.values()))["methods"].keys()) if per_seed else []
    aggregate_methods = {}
    for method_key in method_keys:
        acc = [float(per_seed[str(seed)]["methods"][method_key]["mean_combined"]) for seed in seeds]
        sens = [float(per_seed[str(seed)]["methods"][method_key]["mean_sensitivity"]) for seed in seeds]
        inv = [float(per_seed[str(seed)]["methods"][method_key]["mean_invariance"]) for seed in seeds]
        runtime = [float(per_seed[str(seed)]["methods"][method_key]["runtime_seconds"]) for seed in seeds]
        aggregate_methods[method_key] = {
            "name": per_seed[str(seeds[0])]["methods"][method_key]["method"],
            "accuracy": _summary_stats(acc),
            "sensitivity": _summary_stats(sens),
            "invariance": _summary_stats(inv),
            "runtime_seconds": _summary_stats(runtime),
        }

    aggregate = {
        "config": vars(args),
        "seeds": list(seeds),
        "per_seed": per_seed,
        "aggregate_methods": aggregate_methods,
        "stage_a_expected_accuracy": _summary_stats(
            [float(per_seed[str(seed)]["stage_a_expected_accuracy"]) for seed in seeds]
        ),
    }
    aggregate_path = out_dir / f"h{args.hidden_size}" / "progressive_summary.json"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(aggregate_path), "methods": aggregate_methods}, indent=2))


if __name__ == "__main__":
    main()
