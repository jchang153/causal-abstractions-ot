from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import nn

from .data import CarryPairRecord, ExhaustiveBanks
from .interventions import RunCache, factual_run
from .model import GRUAdder
from .sites import FullStateSite


@dataclass(frozen=True)
class DASConfig:
    subspace_dims: tuple[int, ...] = (2, 4, 8)
    learning_rates: tuple[float, ...] = (1e-2, 3e-3)
    epochs: int = 20
    lambda_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    seed: int = 0
    train_records_per_epoch: int = 256

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class RotatedSubspace(nn.Module):
    def __init__(self, hidden_size: int, subspace_dim: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.subspace_dim = int(subspace_dim)
        self.proj = nn.Parameter(torch.randn(hidden_size, subspace_dim) * 0.05)

    def orth_proj(self) -> torch.Tensor:
        q, _ = torch.linalg.qr(self.proj, mode="reduced")
        return q[:, : self.subspace_dim]

    def intervene(self, base_h: torch.Tensor, source_h: torch.Tensor, lambda_scale: float) -> torch.Tensor:
        q = self.orth_proj()
        delta = source_h - base_h
        return base_h + float(lambda_scale) * (delta @ q @ q.transpose(0, 1))


def _base_bits_tensor(rec: CarryPairRecord, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[[float(rec.base.a_bits_lsb[t]), float(rec.base.b_bits_lsb[t])] for t in range(rec.base.width)]],
        dtype=torch.float32,
        device=device,
    )


def _single_example_rollout(
    model: GRUAdder,
    rec: CarryPairRecord,
    site: FullStateSite,
    rotator: RotatedSubspace,
    lambda_scale: float,
    *,
    device: torch.device,
    run_cache: RunCache | None = None,
) -> torch.Tensor:
    base_bits = _base_bits_tensor(rec, device=device)
    source_run = factual_run(model, rec.source, device=device) if run_cache is None else run_cache.get_run(rec.source)
    source_states = source_run.hidden_states.to(device)
    h = torch.zeros(1, model.hidden_size, device=device, dtype=base_bits.dtype)
    sum_logits = []
    for step in range(model.width):
        h = model.cell(base_bits[:, step, :], h)
        if step == int(site.timestep):
            src_h = source_states[step].unsqueeze(0).to(device=device, dtype=h.dtype)
            h = rotator.intervene(h, src_h, lambda_scale=float(lambda_scale))
        sum_logits.append(model.sum_head(h))
    carry_logit = model.final_carry_head(h)
    return torch.cat(sum_logits + [carry_logit], dim=1)[0]


def _train_single_rotator(
    model: GRUAdder,
    records: Sequence[CarryPairRecord],
    site: FullStateSite,
    subspace_dim: int,
    lr: float,
    epochs: int,
    train_records_per_epoch: int,
    *,
    device: torch.device,
    run_cache: RunCache | None = None,
) -> RotatedSubspace:
    rotator = RotatedSubspace(model.hidden_size, subspace_dim=subspace_dim).to(device)
    optimizer = torch.optim.Adam(rotator.parameters(), lr=float(lr))
    criterion = nn.BCEWithLogitsLoss()
    rng = torch.Generator().manual_seed(0)
    for _ in range(int(epochs)):
        rotator.train()
        if len(records) <= 0:
            break
        if len(records) <= int(train_records_per_epoch):
            sampled_records = records
        else:
            idx = torch.randperm(len(records), generator=rng)[: int(train_records_per_epoch)].tolist()
            sampled_records = [records[i] for i in idx]
        for rec in sampled_records:
            optimizer.zero_grad(set_to_none=True)
            logits = _single_example_rollout(
                model,
                rec,
                site,
                rotator,
                lambda_scale=1.0,
                device=device,
                run_cache=run_cache,
            )
            target = torch.tensor(rec.counterfactual.output_bits_lsb, dtype=torch.float32, device=device)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
    return rotator.cpu()


def _exact_match_rate(
    model: GRUAdder,
    records: Sequence[CarryPairRecord],
    site: FullStateSite,
    rotator: RotatedSubspace,
    lambda_scale: float,
    *,
    device: torch.device,
    run_cache: RunCache | None = None,
) -> float:
    if not records:
        return 0.0
    hits = 0
    rotator = rotator.to(device)
    for rec in records:
        logits = _single_example_rollout(
            model,
            rec,
            site,
            rotator,
            lambda_scale=float(lambda_scale),
            device=device,
            run_cache=run_cache,
        )
        pred = (torch.sigmoid(logits) >= 0.5).to(torch.int64).cpu()
        tgt = torch.tensor(rec.counterfactual.output_bits_lsb, dtype=torch.int64)
        hits += int(torch.equal(pred, tgt))
    return float(hits / len(records))


def run_das_sweep(
    model: GRUAdder,
    banks: ExhaustiveBanks,
    sites: Sequence[FullStateSite],
    config: DASConfig,
    *,
    device: torch.device,
    run_cache: RunCache | None = None,
) -> dict[str, object]:
    torch.manual_seed(int(config.seed))
    trials = []
    best_by_carry: dict[int, dict[str, object]] = {}
    for carry_index in range(1, banks.width + 1):
        fit_records = banks.fit_by_carry[carry_index]
        carry_best_key = None
        carry_best = None
        for site in sites:
            for subspace_dim in config.subspace_dims:
                for lr in config.learning_rates:
                    rotator = _train_single_rotator(
                        model,
                        fit_records,
                        site,
                        subspace_dim=int(subspace_dim),
                        lr=float(lr),
                        epochs=int(config.epochs),
                        train_records_per_epoch=int(config.train_records_per_epoch),
                        device=device,
                        run_cache=run_cache,
                    )
                    for lambda_scale in config.lambda_grid:
                        sens = _exact_match_rate(
                            model,
                            banks.calib_positive_by_carry[carry_index],
                            site,
                            rotator,
                            lambda_scale=float(lambda_scale),
                            device=device,
                            run_cache=run_cache,
                        )
                        inv = _exact_match_rate(
                            model,
                            banks.calib_invariant_by_carry[carry_index],
                            site,
                            rotator,
                            lambda_scale=float(lambda_scale),
                            device=device,
                            run_cache=run_cache,
                        )
                        combined = 0.5 * (sens + inv)
                        key = (combined, sens, inv)
                        trial = {
                            "carry_index": int(carry_index),
                            "site_key": site.key(),
                            "site_timestep": int(site.timestep),
                            "subspace_dim": int(subspace_dim),
                            "lr": float(lr),
                            "lambda": float(lambda_scale),
                            "calibration": {
                                "sensitivity": float(sens),
                                "invariance": float(inv),
                                "combined": float(combined),
                            },
                            "rotator_state": {k: v.tolist() for k, v in rotator.state_dict().items()},
                        }
                        trials.append(trial)
                        if carry_best_key is None or key > carry_best_key:
                            carry_best_key = key
                            carry_best = trial
        best_by_carry[carry_index] = carry_best if carry_best is not None else {}

    per_carry = {}
    sens_vals = []
    inv_vals = []
    for carry_index, trial in best_by_carry.items():
        rotator = RotatedSubspace(model.hidden_size, int(trial["subspace_dim"]))
        rotator.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in trial["rotator_state"].items()})
        site = next(site for site in sites if site.key() == trial["site_key"])
        sens = _exact_match_rate(
            model,
            banks.test_positive_by_carry[carry_index],
            site,
            rotator,
            lambda_scale=float(trial["lambda"]),
            device=device,
            run_cache=run_cache,
        )
        inv = _exact_match_rate(
            model,
            banks.test_invariant_by_carry[carry_index],
            site,
            rotator,
            lambda_scale=float(trial["lambda"]),
            device=device,
            run_cache=run_cache,
        )
        combined = 0.5 * (sens + inv)
        sens_vals.append(sens)
        inv_vals.append(inv)
        per_carry[str(carry_index)] = {
            "sensitivity": float(sens),
            "invariance": float(inv),
            "combined": float(combined),
            "site_key": site.key(),
            "subspace_dim": int(trial["subspace_dim"]),
            "lambda": float(trial["lambda"]),
        }

    return {
        "das_config": config.as_dict(),
        "trials": trials,
        "best_by_carry": best_by_carry,
        "test": {
            "per_carry": per_carry,
            "sensitivity_mean": float(sum(sens_vals) / max(1, len(sens_vals))),
            "invariance_mean": float(sum(inv_vals) / max(1, len(inv_vals))),
            "combined_mean": float((sum(sens_vals) + sum(inv_vals)) / max(1, 2 * len(sens_vals))),
        },
    }
