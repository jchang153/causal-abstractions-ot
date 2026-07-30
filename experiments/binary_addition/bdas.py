from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .data import CarryPairRecord, ExhaustiveBanks
from .interventions import RunCache, factual_run
from .model import GRUAdder, examples_to_tensors
from .sites import FullStateSite


@dataclass(frozen=True)
class BDASConfig:
    """Boundless DAS defaults from the authors' released implementation.

    Candidate timesteps, target carries, data splits, and model/data seeds are
    deliberately supplied to ``run_bdas_sweep`` rather than hidden here.
    """

    rotation_learning_rate: float = 1e-3
    boundary_learning_rate: float = 1e-2
    boundary_init: float = 0.5
    boundary_penalty: float = 1.0
    boundary_penalty_power: float = 1.0
    boundary_penalty_warmup_fraction: float = 0.0
    temperature_start: float = 50.0
    temperature_end: float = 0.1
    epochs: int = 3
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    warmup_fraction: float = 0.1
    scheduler_horizon: str = "microsteps"
    changed_bit_weight: float = 1.0
    active_record_weight: float = 1.0
    selection_metric: str = "combined"
    eval_every_optimizer_steps: int = 200
    train_records_per_epoch: int | None = None
    restarts: int = 1
    seed: int = 42
    shuffle: bool = False

    def __post_init__(self) -> None:
        if self.rotation_learning_rate <= 0 or self.boundary_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if not 0 < self.boundary_init <= 1:
            raise ValueError("boundary_init must be in (0, 1]")
        if self.boundary_penalty < 0:
            raise ValueError("boundary_penalty must be nonnegative")
        if self.boundary_penalty_power <= 0:
            raise ValueError("boundary_penalty_power must be positive")
        if not 0 <= self.boundary_penalty_warmup_fraction < 1:
            raise ValueError("boundary_penalty_warmup_fraction must be in [0, 1)")
        if self.temperature_start <= 0 or self.temperature_end <= 0:
            raise ValueError("temperatures must be positive")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if self.scheduler_horizon not in {"microsteps", "optimizer_steps", "constant"}:
            raise ValueError("scheduler_horizon must be microsteps, optimizer_steps, or constant")
        if self.changed_bit_weight <= 0 or self.active_record_weight <= 0:
            raise ValueError("loss weights must be positive")
        if self.selection_metric not in {"combined", "sensitivity"}:
            raise ValueError("selection_metric must be combined or sensitivity")
        if self.eval_every_optimizer_steps <= 0 or self.restarts <= 0:
            raise ValueError("evaluation interval and restarts must be positive")
        if self.train_records_per_epoch is not None and self.train_records_per_epoch <= 0:
            raise ValueError("train_records_per_epoch must be positive when supplied")

    @property
    def effective_batch_size(self) -> int:
        return int(self.batch_size * self.gradient_accumulation_steps)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class BoundlessRotatedSubspace(nn.Module):
    """Full orthogonal rotation with a learned contiguous prefix boundary."""

    def __init__(self, hidden_size: int, *, boundary_init: float = 0.5) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if not 0 < float(boundary_init) <= 1:
            raise ValueError("boundary_init must be in (0, 1]")

        rotate_layer = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        nn.init.orthogonal_(rotate_layer.weight)
        self.rotate_layer = nn.utils.parametrizations.orthogonal(rotate_layer)
        self.boundary_fraction = nn.Parameter(torch.tensor([float(boundary_init)]))
        self.register_buffer("temperature", torch.tensor(50.0))
        self.register_buffer(
            "intervention_population",
            torch.arange(self.hidden_size, dtype=torch.float32),
        )

    def clamped_boundary(self) -> torch.Tensor:
        return torch.clamp(self.boundary_fraction, 1e-3, 1.0)

    def set_temperature(self, temperature: float | torch.Tensor) -> None:
        value = torch.as_tensor(temperature, device=self.temperature.device, dtype=self.temperature.dtype)
        if value.numel() != 1 or float(value.item()) <= 0:
            raise ValueError("temperature must be a positive scalar")
        self.temperature.copy_(value.reshape_as(self.temperature))

    def soft_mask(self) -> torch.Tensor:
        boundary = self.clamped_boundary()[0] * self.hidden_size
        population = self.intervention_population.to(dtype=self.boundary_fraction.dtype)
        temperature = self.temperature.to(dtype=self.boundary_fraction.dtype)
        return torch.sigmoid(population / temperature) * torch.sigmoid(
            (boundary - population) / temperature
        )

    def hard_dimension(self) -> int:
        fraction = float(self.clamped_boundary().detach().cpu().item())
        return max(1, min(self.hidden_size, int(math.ceil(fraction * self.hidden_size))))

    def hard_mask(self) -> torch.Tensor:
        mask = torch.zeros(
            self.hidden_size,
            device=self.boundary_fraction.device,
            dtype=self.boundary_fraction.dtype,
        )
        mask[: self.hard_dimension()] = 1.0
        return mask

    def rotation_weight(self) -> torch.Tensor:
        return self.rotate_layer.weight

    def intervene(
        self,
        base_h: torch.Tensor,
        source_h: torch.Tensor,
        *,
        hard: bool,
    ) -> torch.Tensor:
        """Apply a unit-strength interchange intervention; no lambda exists."""

        weight = self.rotation_weight().to(device=base_h.device, dtype=base_h.dtype)
        mask = (self.hard_mask() if hard else self.soft_mask()).to(
            device=base_h.device,
            dtype=base_h.dtype,
        )
        rotated_base = base_h @ weight
        rotated_source = source_h @ weight
        rotated_output = (1.0 - mask) * rotated_base + mask * rotated_source
        return rotated_output @ weight.transpose(0, 1)


def _rollout_batch(
    model: GRUAdder,
    records: Sequence[CarryPairRecord],
    site: FullStateSite,
    intervention: BoundlessRotatedSubspace,
    *,
    hard: bool,
    device: torch.device,
    run_cache: RunCache | None,
) -> torch.Tensor:
    if not records:
        return torch.empty((0, model.width + 1), device=device)

    if run_cache is None:
        base_x, _ = examples_to_tensors([record.base for record in records])
        base_x = base_x.to(device)
        source_states = torch.stack(
            [factual_run(model, record.source, device=device).hidden_states for record in records],
            dim=0,
        ).to(device)
    else:
        base_x = torch.cat([run_cache.get_input(record.base) for record in records], dim=0).to(device)
        source_states = torch.stack(
            [run_cache.get_run(record.source).hidden_states for record in records],
            dim=0,
        ).to(device)

    h = torch.zeros(base_x.size(0), model.hidden_size, device=device, dtype=base_x.dtype)
    sum_logits = []
    for step in range(model.width):
        h = model.cell(base_x[:, step, :], h)
        if step == int(site.timestep):
            source_h = source_states[:, step, :].to(device=device, dtype=h.dtype)
            h = intervention.intervene(h, source_h, hard=hard)
        sum_logits.append(model.sum_head(h))
    carry_logit = model.final_carry_head(h)
    return torch.cat(sum_logits + [carry_logit], dim=1)


@torch.no_grad()
def _exact_match_rate(
    model: GRUAdder,
    records: Sequence[CarryPairRecord],
    site: FullStateSite,
    intervention: BoundlessRotatedSubspace,
    *,
    batch_size: int,
    device: torch.device,
    run_cache: RunCache | None,
) -> float:
    if not records:
        return 0.0
    intervention.eval()
    hits = 0
    for start in range(0, len(records), int(batch_size)):
        batch = records[start : start + int(batch_size)]
        logits = _rollout_batch(
            model,
            batch,
            site,
            intervention,
            hard=True,
            device=device,
            run_cache=run_cache,
        )
        predictions = (torch.sigmoid(logits) >= 0.5).to(torch.int64).cpu()
        targets = torch.tensor(
            [record.counterfactual.output_bits_lsb for record in batch],
            dtype=torch.int64,
        )
        hits += int((predictions == targets).all(dim=1).sum().item())
    return float(hits / len(records))


def _calibration_metrics(
    model: GRUAdder,
    positive_records: Sequence[CarryPairRecord],
    invariant_records: Sequence[CarryPairRecord],
    site: FullStateSite,
    intervention: BoundlessRotatedSubspace,
    *,
    batch_size: int,
    device: torch.device,
    run_cache: RunCache | None,
) -> dict[str, float]:
    sensitivity = _exact_match_rate(
        model,
        positive_records,
        site,
        intervention,
        batch_size=batch_size,
        device=device,
        run_cache=run_cache,
    )
    invariance = _exact_match_rate(
        model,
        invariant_records,
        site,
        intervention,
        batch_size=batch_size,
        device=device,
        run_cache=run_cache,
    )
    return {
        "sensitivity": float(sensitivity),
        "invariance": float(invariance),
        "combined": float(0.5 * (sensitivity + invariance)),
    }


def _linear_warmup_decay(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    remaining = max(0, total_steps - step)
    decay_span = max(1, total_steps - warmup_steps)
    return float(remaining) / float(decay_span)


def _selection_key(
    metrics: Mapping[str, float],
    *,
    hard_dimension: int,
    selection_metric: str,
) -> tuple[float, float, float, int]:
    if selection_metric == "sensitivity":
        return (
            float(metrics["sensitivity"]),
            float(metrics["invariance"]),
            float(metrics["combined"]),
            -int(hard_dimension),
        )
    return (
        float(metrics["combined"]),
        float(metrics["sensitivity"]),
        float(metrics["invariance"]),
        -int(hard_dimension),
    )


def _train_candidate(
    model: GRUAdder,
    fit_records: Sequence[CarryPairRecord],
    calibration_positive: Sequence[CarryPairRecord],
    calibration_invariant: Sequence[CarryPairRecord],
    site: FullStateSite,
    config: BDASConfig,
    *,
    restart: int,
    device: torch.device,
    run_cache: RunCache | None,
) -> tuple[BoundlessRotatedSubspace, dict[str, object]]:
    candidate_seed = int(config.seed) + 1009 * int(restart) + 9176 * int(site.timestep)
    random.seed(candidate_seed)
    torch.manual_seed(candidate_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(candidate_seed)

    intervention = BoundlessRotatedSubspace(
        model.hidden_size,
        boundary_init=float(config.boundary_init),
    ).to(device)
    optimizer = torch.optim.Adam(
        [
            {
                "params": intervention.rotate_layer.parameters(),
                "lr": float(config.rotation_learning_rate),
            },
            {
                "params": [intervention.boundary_fraction],
                "lr": float(config.boundary_learning_rate),
            },
        ]
    )

    fit_records = list(fit_records)
    if not fit_records:
        raise ValueError("BDAS requires at least one fit record")
    epoch_record_count = len(fit_records)
    if config.train_records_per_epoch is not None:
        epoch_record_count = min(epoch_record_count, int(config.train_records_per_epoch))
    microbatches_per_epoch = max(1, math.ceil(epoch_record_count / int(config.batch_size)))
    optimizer_steps_per_epoch = max(
        1,
        math.ceil(microbatches_per_epoch / int(config.gradient_accumulation_steps)),
    )
    total_optimizer_steps = int(config.epochs * optimizer_steps_per_epoch)
    total_microsteps = int(config.epochs * microbatches_per_epoch)
    # Match the released Boundless DAS notebook: the scheduler horizon is
    # measured in dataloader microsteps even though scheduler.step() is called
    # only when gradient accumulation triggers an optimizer update.
    scheduler_total_steps = (
        int(total_optimizer_steps)
        if config.scheduler_horizon == "optimizer_steps"
        else int(total_microsteps)
    )
    warmup_steps = int(config.warmup_fraction * scheduler_total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=(
            (lambda _step: 1.0)
            if config.scheduler_horizon == "constant"
            else lambda step: _linear_warmup_decay(
                int(step),
                warmup_steps=warmup_steps,
                total_steps=scheduler_total_steps,
            )
        ),
    )
    rng = random.Random(candidate_seed)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_key: tuple[float, float, float, int] | None = None
    optimizer_step = 0
    microstep = 0
    optimizer.zero_grad(set_to_none=True)

    for _epoch in range(int(config.epochs)):
        if len(fit_records) > epoch_record_count:
            epoch_records = rng.sample(fit_records, epoch_record_count)
        else:
            epoch_records = list(fit_records)
        if config.shuffle and len(fit_records) <= epoch_record_count:
            rng.shuffle(epoch_records)
        for batch_index, start in enumerate(range(0, len(epoch_records), int(config.batch_size))):
            batch = epoch_records[start : start + int(config.batch_size)]
            progress = 0.0 if total_microsteps <= 1 else float(microstep) / float(total_microsteps - 1)
            temperature = float(config.temperature_start) + progress * (
                float(config.temperature_end) - float(config.temperature_start)
            )
            intervention.set_temperature(temperature)
            intervention.train()
            logits = _rollout_batch(
                model,
                batch,
                site,
                intervention,
                hard=False,
                device=device,
                run_cache=run_cache,
            )
            targets = torch.tensor(
                [record.counterfactual.output_bits_lsb for record in batch],
                dtype=torch.float32,
                device=device,
            )
            element_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            factual_targets = torch.tensor(
                [record.base.output_bits_lsb for record in batch],
                dtype=torch.float32,
                device=device,
            )
            changed = (targets != factual_targets).to(dtype=element_loss.dtype)
            element_weights = 1.0 + (float(config.changed_bit_weight) - 1.0) * changed
            active = torch.tensor(
                [float(getattr(record, "is_active", bool(changed[index].any().item()))) for index, record in enumerate(batch)],
                dtype=element_loss.dtype,
                device=device,
            ).unsqueeze(1)
            element_weights = element_weights * (
                1.0 + (float(config.active_record_weight) - 1.0) * active
            )
            prediction_loss = (element_loss * element_weights).sum() / element_weights.sum()
            penalty_active = progress >= float(config.boundary_penalty_warmup_fraction)
            boundary_loss = (
                float(config.boundary_penalty)
                * intervention.clamped_boundary().pow(float(config.boundary_penalty_power)).sum()
                if penalty_active
                else intervention.clamped_boundary().sum() * 0.0
            )
            loss = (prediction_loss + boundary_loss) / int(config.gradient_accumulation_steps)
            loss.backward()

            last_batch = batch_index + 1 == microbatches_per_epoch
            accumulation_boundary = (batch_index + 1) % int(config.gradient_accumulation_steps) == 0
            if accumulation_boundary or last_batch:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                should_evaluate = (
                    optimizer_step % int(config.eval_every_optimizer_steps) == 0
                    or optimizer_step == total_optimizer_steps
                )
                if should_evaluate:
                    metrics = _calibration_metrics(
                        model,
                        calibration_positive,
                        calibration_invariant,
                        site,
                        intervention,
                        batch_size=int(config.batch_size),
                        device=device,
                        run_cache=run_cache,
                    )
                    hard_dimension = intervention.hard_dimension()
                    key = _selection_key(
                        metrics,
                        hard_dimension=hard_dimension,
                        selection_metric=config.selection_metric,
                    )
                    history.append(
                        {
                            "optimizer_step": float(optimizer_step),
                            "temperature": float(temperature),
                            "prediction_loss": float(prediction_loss.detach().cpu().item()),
                            "boundary_fraction": float(intervention.clamped_boundary().detach().cpu().item()),
                            "hard_dimension": float(hard_dimension),
                            **metrics,
                        }
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best_metrics = dict(metrics)
                        best_state = copy.deepcopy(intervention.state_dict())
            microstep += 1

    if best_state is None:
        best_metrics = _calibration_metrics(
            model,
            calibration_positive,
            calibration_invariant,
            site,
            intervention,
            batch_size=int(config.batch_size),
            device=device,
            run_cache=run_cache,
        )
        best_state = copy.deepcopy(intervention.state_dict())
    intervention.load_state_dict(best_state)
    return intervention.cpu(), {
        "seed": int(candidate_seed),
        "restart": int(restart),
        "optimizer_steps": int(optimizer_step),
        "scheduler_total_steps": int(scheduler_total_steps),
        "scheduler_warmup_steps": int(warmup_steps),
        "temperature_total_microsteps": int(total_microsteps),
        "best_calibration": best_metrics,
        "history": history,
    }


def run_bdas_sweep(
    model: GRUAdder,
    banks: ExhaustiveBanks,
    *,
    sites: Sequence[FullStateSite],
    carry_indices: Sequence[int],
    config: BDASConfig | None = None,
    device: torch.device,
    run_cache: RunCache | None = None,
) -> dict[str, object]:
    """Run BDAS only after sites and target carries are explicitly supplied."""

    config = BDASConfig() if config is None else config
    sites = tuple(sites)
    carry_indices = tuple(int(index) for index in carry_indices)
    if not sites:
        raise ValueError("at least one candidate timestep site is required")
    if not carry_indices:
        raise ValueError("at least one target carry index is required")
    invalid_sites = [site.timestep for site in sites if not 0 <= int(site.timestep) < model.width]
    if invalid_sites:
        raise ValueError(f"candidate timesteps must be in [0, {model.width - 1}]: {invalid_sites}")
    invalid_carries = [index for index in carry_indices if not 1 <= index <= banks.width]
    if invalid_carries:
        raise ValueError(f"carry indices must be in [1, {banks.width}]: {invalid_carries}")

    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    started = perf_counter()
    selected: dict[str, dict[str, object]] = {}
    trials: list[dict[str, object]] = []

    for carry_index in carry_indices:
        best_key: tuple[float, float, float, int] | None = None
        best_intervention: BoundlessRotatedSubspace | None = None
        best_trial: dict[str, object] | None = None
        for site in sites:
            for restart in range(int(config.restarts)):
                intervention, training = _train_candidate(
                    model,
                    banks.fit_by_carry[carry_index],
                    banks.calib_positive_by_carry[carry_index],
                    banks.calib_invariant_by_carry[carry_index],
                    site,
                    config,
                    restart=restart,
                    device=device,
                    run_cache=run_cache,
                )
                intervention = intervention.to(device)
                calibration = dict(training["best_calibration"] or {})
                hard_dimension = intervention.hard_dimension()
                trial = {
                    "carry_index": int(carry_index),
                    "site_key": site.key(),
                    "site_timestep": int(site.timestep),
                    "restart": int(restart),
                    "lambda": 1.0,
                    "boundary_fraction": float(intervention.clamped_boundary().detach().cpu().item()),
                    "hard_dimension": int(hard_dimension),
                    "calibration": calibration,
                    "training": training,
                    "rotation_weight": intervention.rotation_weight().detach().cpu().tolist(),
                }
                trials.append(trial)
                key = _selection_key(
                    calibration,
                    hard_dimension=hard_dimension,
                    selection_metric=config.selection_metric,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_intervention = intervention
                    best_trial = trial

        if best_intervention is None or best_trial is None:
            raise RuntimeError(f"no BDAS candidate was trained for carry C{carry_index}")
        best_site = next(site for site in sites if site.key() == best_trial["site_key"])
        test = _calibration_metrics(
            model,
            banks.test_positive_by_carry[carry_index],
            banks.test_invariant_by_carry[carry_index],
            best_site,
            best_intervention,
            batch_size=int(config.batch_size),
            device=device,
            run_cache=run_cache,
        )
        selected[str(carry_index)] = {**best_trial, "test": test}

    return {
        "method": "boundless_das",
        "lambda": 1.0,
        "config": config.as_dict(),
        "candidate_sites": [site.key() for site in sites],
        "carry_indices": list(carry_indices),
        "trials": trials,
        "selected_by_carry": selected,
        "runtime_seconds": float(perf_counter() - started),
    }


def run_bdas_rows(
    model: GRUAdder,
    *,
    fit_by_row: Mapping[str, Sequence[CarryPairRecord]],
    calibration_positive_by_row: Mapping[str, Sequence[CarryPairRecord]],
    calibration_invariant_by_row: Mapping[str, Sequence[CarryPairRecord]],
    test_positive_by_row: Mapping[str, Sequence[CarryPairRecord]],
    test_invariant_by_row: Mapping[str, Sequence[CarryPairRecord]],
    sites: Sequence[FullStateSite],
    row_keys: Sequence[str],
    config: BDASConfig | None = None,
    device: torch.device,
    run_cache: RunCache | None = None,
) -> dict[str, object]:
    """Run Boundless DAS on the structured endogenous banks used by Full DAS."""

    config = BDASConfig() if config is None else config
    sites = tuple(sites)
    row_keys = tuple(str(key) for key in row_keys)
    if not sites:
        raise ValueError("at least one candidate timestep site is required")
    if not row_keys:
        raise ValueError("at least one target row is required")
    invalid_sites = [site.timestep for site in sites if not 0 <= int(site.timestep) < model.width]
    if invalid_sites:
        raise ValueError(f"candidate timesteps must be in [0, {model.width - 1}]: {invalid_sites}")
    bank_maps = (
        fit_by_row,
        calibration_positive_by_row,
        calibration_invariant_by_row,
        test_positive_by_row,
        test_invariant_by_row,
    )
    missing = sorted({key for key in row_keys for bank in bank_maps if key not in bank})
    if missing:
        raise ValueError(f"missing BDAS bank rows: {missing}")

    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    started = perf_counter()
    selected: dict[str, dict[str, object]] = {}
    trials: list[dict[str, object]] = []

    for row_offset, row_key in enumerate(row_keys):
        best_key: tuple[float, float, float, int] | None = None
        best_intervention: BoundlessRotatedSubspace | None = None
        best_trial: dict[str, object] | None = None
        for site in sites:
            for restart in range(int(config.restarts)):
                row_config = replace(config, seed=int(config.seed) + 104729 * row_offset)
                intervention, training = _train_candidate(
                    model,
                    fit_by_row[row_key],
                    calibration_positive_by_row[row_key],
                    calibration_invariant_by_row[row_key],
                    site,
                    row_config,
                    restart=restart,
                    device=device,
                    run_cache=run_cache,
                )
                intervention = intervention.to(device)
                calibration = dict(training["best_calibration"] or {})
                hard_dimension = intervention.hard_dimension()
                trial = {
                    "row_key": row_key,
                    "site_key": site.key(),
                    "site_timestep": int(site.timestep),
                    "restart": int(restart),
                    "lambda": 1.0,
                    "boundary_fraction": float(intervention.clamped_boundary().detach().cpu().item()),
                    "hard_dimension": int(hard_dimension),
                    "calibration": calibration,
                    "training": training,
                    "rotation_weight": intervention.rotation_weight().detach().cpu().tolist(),
                }
                trials.append(trial)
                key = _selection_key(
                    calibration,
                    hard_dimension=hard_dimension,
                    selection_metric=config.selection_metric,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_intervention = intervention
                    best_trial = trial

        if best_intervention is None or best_trial is None:
            raise RuntimeError(f"no BDAS candidate was trained for row {row_key}")
        best_site = next(site for site in sites if site.key() == best_trial["site_key"])
        test = _calibration_metrics(
            model,
            test_positive_by_row[row_key],
            test_invariant_by_row[row_key],
            best_site,
            best_intervention,
            batch_size=int(config.batch_size),
            device=device,
            run_cache=run_cache,
        )
        selected[row_key] = {**best_trial, "test": test}

    metrics = {
        name: float(sum(float(selected[key]["test"][name]) for key in row_keys) / len(row_keys))
        for name in ("sensitivity", "invariance", "combined")
    }
    return {
        "method": "boundless_das",
        "lambda": 1.0,
        "config": config.as_dict(),
        "candidate_sites": [site.key() for site in sites],
        "row_keys": list(row_keys),
        "trials": trials,
        "selected_by_row": selected,
        "test": metrics,
        "runtime_seconds": float(perf_counter() - started),
    }
