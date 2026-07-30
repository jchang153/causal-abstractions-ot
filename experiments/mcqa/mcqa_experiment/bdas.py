"""Boundless DAS search over MCQA residual-stream layers."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader

from .das import _ensure_model_on_device, _mini_bank_from_batch, _sync_if_cuda
from .data import MCQAPairBank, MCQAPairDataset
from .intervention import run_das_residual_intervention
from .metrics import cross_entropy_for_bank, das_metrics_from_logits, das_prediction_details_from_logits
from .sites import SiteLike, site_token_position_ids, site_total_width


@dataclass(frozen=True)
class BoundlessDASConfig:
    """Recommended bDAS settings, matched to the binary-addition baseline."""

    method_name: str = "boundless_das"
    batch_size: int = 64
    epochs: int = 12
    rotation_learning_rate: float = 1e-2
    boundary_learning_rate: float = 1e-4
    boundary_init: float = 0.5
    boundary_penalty: float = 1.0
    boundary_penalty_power: float = 1.0
    temperature_start: float = 1.0
    temperature_end: float = 0.1
    gradient_accumulation_steps: int = 1
    warmup_fraction: float = 0.1
    restarts: int = 1
    seed: int = 42
    shuffle: bool = False
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if self.rotation_learning_rate <= 0 or self.boundary_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if not 0 < self.boundary_init <= 1:
            raise ValueError("boundary_init must be in (0, 1]")
        if self.boundary_penalty < 0 or self.boundary_penalty_power <= 0:
            raise ValueError("boundary penalty must be nonnegative with positive power")
        if self.temperature_start <= 0 or self.temperature_end <= 0:
            raise ValueError("temperatures must be positive")
        if self.gradient_accumulation_steps <= 0 or self.restarts <= 0:
            raise ValueError("gradient accumulation and restarts must be positive")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class BoundlessDASIntervention(nn.Module):
    """Full orthogonal rotation with a learned contiguous prefix boundary."""

    def __init__(self, hidden_size: int, *, boundary_init: float = 0.5) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        rotate = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        nn.init.orthogonal_(rotate.weight)
        self.rotate = nn.utils.parametrizations.orthogonal(rotate)
        self.boundary_fraction = nn.Parameter(torch.tensor([float(boundary_init)]))
        self.register_buffer("temperature", torch.tensor(1.0))
        self.register_buffer("population", torch.arange(self.hidden_size, dtype=torch.float32))
        self.use_hard_mask = False

    def clamped_boundary(self) -> torch.Tensor:
        return torch.clamp(self.boundary_fraction, 1e-3, 1.0)

    def set_temperature(self, value: float) -> None:
        if float(value) <= 0:
            raise ValueError("temperature must be positive")
        self.temperature.fill_(float(value))

    def soft_mask(self) -> torch.Tensor:
        boundary = self.clamped_boundary()[0] * self.hidden_size
        population = self.population.to(dtype=self.boundary_fraction.dtype)
        temperature = self.temperature.to(dtype=self.boundary_fraction.dtype)
        return torch.sigmoid(population / temperature) * torch.sigmoid((boundary - population) / temperature)

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

    def forward(self, base_vectors: torch.Tensor, source_vectors: torch.Tensor) -> torch.Tensor:
        weight = self.rotate.weight
        compute_dtype = weight.dtype
        base = base_vectors.to(compute_dtype)
        source = source_vectors.to(compute_dtype)
        mask = self.hard_mask() if self.use_hard_mask else self.soft_mask()
        mask = mask.to(device=base.device, dtype=compute_dtype)
        rotated_base = base @ weight
        rotated_source = source @ weight
        updated = ((1.0 - mask) * rotated_base + mask * rotated_source) @ weight.transpose(0, 1)
        return updated.to(base_vectors.dtype)


def _linear_warmup_decay(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    return float(max(0, total_steps - step)) / float(max(1, total_steps - warmup_steps))


def _temperature(config: BoundlessDASConfig, step: int, total_steps: int) -> float:
    progress = 0.0 if total_steps <= 1 else float(step) / float(total_steps - 1)
    return float(config.temperature_start) + progress * (
        float(config.temperature_end) - float(config.temperature_start)
    )


def evaluate_boundless_candidate(
    *,
    model,
    bank: MCQAPairBank,
    site: SiteLike,
    intervention: BoundlessDASIntervention,
    batch_size: int,
    device: torch.device,
    tokenizer,
    return_details: bool = False,
) -> dict[str, object]:
    intervention.eval()
    intervention.use_hard_mask = True
    logits_all = []
    with torch.no_grad():
        for batch in DataLoader(MCQAPairDataset(bank), batch_size=int(batch_size), shuffle=False):
            logits = run_das_residual_intervention(
                model=model,
                base_input_ids=batch["base_input_ids"].to(device),
                base_attention_mask=batch["base_attention_mask"].to(device),
                source_input_ids=batch["source_input_ids"].to(device),
                source_attention_mask=batch["source_attention_mask"].to(device),
                site=site,
                intervention=intervention,
                base_position_by_id={key: value.to(device) for key, value in batch["base_positions"].items()},
                source_position_by_id={key: value.to(device) for key, value in batch["source_positions"].items()},
            )
            logits_all.append(logits.detach().cpu())
    logits = torch.cat(logits_all, dim=0)
    metrics = das_metrics_from_logits(logits, bank, tokenizer=tokenizer)
    if return_details:
        metrics["prediction_details"] = das_prediction_details_from_logits(logits, bank, tokenizer=tokenizer)
    return metrics


def _train_candidate(
    *,
    model,
    train_bank: MCQAPairBank,
    site: SiteLike,
    device: torch.device,
    config: BoundlessDASConfig,
    restart_index: int,
) -> tuple[BoundlessDASIntervention, dict[str, object]]:
    candidate_seed = int(config.seed) + 1009 * int(restart_index) + 9176 * int(site.layer)
    random.seed(candidate_seed)
    torch.manual_seed(candidate_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(candidate_seed)
    intervention = BoundlessDASIntervention(
        site_total_width(site, model_hidden_size=int(model.config.hidden_size)),
        boundary_init=float(config.boundary_init),
    ).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": intervention.rotate.parameters(), "lr": float(config.rotation_learning_rate)},
            {"params": [intervention.boundary_fraction], "lr": float(config.boundary_learning_rate)},
        ]
    )
    microbatches = max(1, math.ceil(int(train_bank.size) / int(config.batch_size)))
    optimizer_steps_per_epoch = max(1, math.ceil(microbatches / int(config.gradient_accumulation_steps)))
    total_optimizer_steps = int(config.epochs) * optimizer_steps_per_epoch
    total_microsteps = int(config.epochs) * microbatches
    warmup_steps = int(float(config.warmup_fraction) * total_optimizer_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _linear_warmup_decay(
            int(step), warmup_steps=warmup_steps, total_steps=total_optimizer_steps
        ),
    )
    history: list[dict[str, float]] = []
    microstep = 0
    optimizer_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(int(config.epochs)):
        loader = DataLoader(
            MCQAPairDataset(train_bank),
            batch_size=int(config.batch_size),
            shuffle=bool(config.shuffle),
        )
        loss_sum = 0.0
        example_count = 0
        for batch_index, batch in enumerate(loader):
            intervention.train()
            intervention.use_hard_mask = False
            intervention.set_temperature(_temperature(config, microstep, total_microsteps))
            logits = run_das_residual_intervention(
                model=model,
                base_input_ids=batch["base_input_ids"].to(device),
                base_attention_mask=batch["base_attention_mask"].to(device),
                source_input_ids=batch["source_input_ids"].to(device),
                source_attention_mask=batch["source_attention_mask"].to(device),
                site=site,
                intervention=intervention,
                base_position_by_id={key: value.to(device) for key, value in batch["base_positions"].items()},
                source_position_by_id={key: value.to(device) for key, value in batch["source_positions"].items()},
            )
            mini_bank = _mini_bank_from_batch(train_bank, batch)
            prediction_loss = cross_entropy_for_bank(logits, mini_bank)
            boundary_loss = float(config.boundary_penalty) * intervention.clamped_boundary().pow(
                float(config.boundary_penalty_power)
            ).sum()
            loss = (prediction_loss + boundary_loss) / int(config.gradient_accumulation_steps)
            loss.backward()
            batch_examples = int(batch["base_input_ids"].shape[0])
            loss_sum += float(prediction_loss.detach().cpu()) * batch_examples
            example_count += batch_examples
            last_batch = batch_index + 1 == microbatches
            if (batch_index + 1) % int(config.gradient_accumulation_steps) == 0 or last_batch:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
            microstep += 1
        history.append(
            {
                "epoch": float(epoch + 1),
                "prediction_loss": float(loss_sum / max(1, example_count)),
                "boundary_fraction": float(intervention.clamped_boundary().detach().cpu().item()),
                "hard_dimension": float(intervention.hard_dimension()),
                "temperature": float(intervention.temperature.detach().cpu().item()),
            }
        )
    return intervention, {
        "candidate_seed": int(candidate_seed),
        "optimizer_steps": int(optimizer_step),
        "history": history,
    }


def run_boundless_das_pipeline(
    *,
    model,
    train_bank: MCQAPairBank,
    calibration_bank: MCQAPairBank,
    holdout_bank: MCQAPairBank,
    sites: list[SiteLike],
    device: torch.device | str,
    tokenizer,
    config: BoundlessDASConfig,
) -> dict[str, object]:
    """Calibrate all layer candidates and evaluate only the selected one on test."""

    device = torch.device(device)
    _ensure_model_on_device(model, device, verbose=config.verbose, method_name=config.method_name)
    model.eval()
    model.requires_grad_(False)
    _sync_if_cuda(device)
    total_start = perf_counter()
    train_calibrate_seconds = 0.0
    trials: list[dict[str, object]] = []
    best_key: tuple[float, int, int] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_site: SiteLike | None = None
    best_trial: dict[str, object] | None = None
    for site in sites:
        for restart_index in range(max(1, int(config.restarts))):
            candidate_start = perf_counter()
            intervention, training = _train_candidate(
                model=model,
                train_bank=train_bank,
                site=site,
                device=device,
                config=config,
                restart_index=restart_index,
            )
            calibration = evaluate_boundless_candidate(
                model=model,
                bank=calibration_bank,
                site=site,
                intervention=intervention,
                batch_size=config.batch_size,
                device=device,
                tokenizer=tokenizer,
            )
            candidate_seconds = perf_counter() - candidate_start
            train_calibrate_seconds += float(candidate_seconds)
            hard_dimension = intervention.hard_dimension()
            trial = {
                "method": config.method_name,
                "variable": train_bank.target_var,
                "site_label": site.label,
                "layer": int(site.layer),
                "token_position_ids": list(site_token_position_ids(site)),
                "restart_index": int(restart_index),
                "boundary_fraction": float(intervention.clamped_boundary().detach().cpu().item()),
                "hard_dimension": int(hard_dimension),
                "calibration_exact_acc": float(calibration["exact_acc"]),
                "selection_exact_acc": float(calibration["exact_acc"]),
                "train_calibrate_seconds": float(candidate_seconds),
                "training": training,
            }
            trials.append(trial)
            key = (float(calibration["exact_acc"]), -int(hard_dimension), -int(site.layer))
            if best_key is None or key > best_key:
                best_key = key
                best_site = site
                best_trial = trial
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in intervention.state_dict().items()}
            del intervention
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if best_site is None or best_trial is None or best_state is None:
        raise RuntimeError(f"No bDAS candidate was trained for {train_bank.target_var}")
    selected = BoundlessDASIntervention(
        site_total_width(best_site, model_hidden_size=int(model.config.hidden_size)),
        boundary_init=float(config.boundary_init),
    ).to(device)
    selected.load_state_dict(best_state)
    selected_test_start = perf_counter()
    holdout = evaluate_boundless_candidate(
        model=model,
        bank=holdout_bank,
        site=best_site,
        intervention=selected,
        batch_size=config.batch_size,
        device=device,
        tokenizer=tokenizer,
        return_details=True,
    )
    selected_test_seconds = perf_counter() - selected_test_start
    result = {**best_trial, "split": holdout_bank.split, **holdout}
    total_seconds = perf_counter() - total_start
    return {
        "target_var": train_bank.target_var,
        "selection_split": "calibration",
        "test_used_for_selection": False,
        "test_evaluation_policy": "selected_calibration_candidate_only",
        "lambda": 1.0,
        "config": config.as_dict(),
        "runtime_seconds": float(total_seconds),
        "wall_runtime_seconds": float(total_seconds),
        "timing_seconds": {
            "t_bdas_train_calibrate_all_layers": float(train_calibrate_seconds),
            "t_final_holdout_eval": float(selected_test_seconds),
        },
        "search_records": {train_bank.target_var: trials},
        "results": [result],
    }
