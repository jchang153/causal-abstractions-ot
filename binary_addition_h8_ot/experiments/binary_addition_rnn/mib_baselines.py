from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Sequence

import torch
from torch import nn

from .interventions import RunCache
from .model import GRUAdder
from .sites import CoordinateGroupSite, FullStateSite, NeuronSite, OutputLogitSite, Site


@dataclass(frozen=True)
class MIBBaselineConfig:
    lambda_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    selection_rule: str = "combined"
    invariance_floor: float = 0.0
    mask_lrs: tuple[float, ...] = (0.03,)
    mask_l1s: tuple[float, ...] = (0.0,)
    mask_epochs: int = 16
    mask_batch_size: int = 64
    mask_train_records_per_epoch: int = 256
    sae_latent_mult: int = 4
    sae_lr: float = 0.01
    sae_l1: float = 1e-3
    sae_epochs: int = 250
    seed: int = 0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _site_dim(site: Site, hidden_size: int) -> int:
    if isinstance(site, FullStateSite):
        return int(hidden_size)
    if isinstance(site, CoordinateGroupSite):
        return int(len(site.coord_indices))
    if isinstance(site, NeuronSite):
        return 1
    if isinstance(site, OutputLogitSite):
        return 1
    raise TypeError(f"unsupported site type: {type(site)!r}")


def _site_chunk(h: torch.Tensor, site: Site) -> torch.Tensor:
    if isinstance(site, FullStateSite):
        return h
    if isinstance(site, CoordinateGroupSite):
        return h[:, list(site.coord_indices)]
    if isinstance(site, NeuronSite):
        return h[:, [int(site.neuron_index)]]
    raise TypeError(f"hidden-state chunk requested for unsupported site type: {type(site)!r}")


def _replace_site_chunk(h: torch.Tensor, site: Site, updated_chunk: torch.Tensor) -> torch.Tensor:
    if isinstance(site, FullStateSite):
        return updated_chunk
    if isinstance(site, CoordinateGroupSite):
        out = h.clone()
        out[:, list(site.coord_indices)] = updated_chunk
        return out
    if isinstance(site, NeuronSite):
        out = h.clone()
        out[:, int(site.neuron_index)] = updated_chunk[:, 0]
        return out
    raise TypeError(f"hidden-state chunk replacement requested for unsupported site type: {type(site)!r}")


def _calibration_key(*, combined: float, sensitivity: float, invariance: float, selection_rule: str, invariance_floor: float) -> tuple[float, float, float]:
    admissible = 1.0 if float(invariance) >= float(invariance_floor) else 0.0
    if selection_rule == "combined":
        return (admissible, float(combined), float(sensitivity))
    if selection_rule == "sensitivity_only":
        return (1.0, float(sensitivity), float(invariance))
    if selection_rule == "sensitivity_then_invariance":
        return (admissible, float(sensitivity), float(invariance))
    raise ValueError(f"unknown selection_rule: {selection_rule!r}")


class IdentityFeaturizer:
    def __init__(self, dim: int) -> None:
        self.feature_dim = int(dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def intervene(self, base: torch.Tensor, source: torch.Tensor, mask: torch.Tensor, lambda_scale: float) -> torch.Tensor:
        return base + float(lambda_scale) * mask.unsqueeze(0) * (source - base)


class PCAFeaturizer:
    def __init__(self, mean: torch.Tensor, components: torch.Tensor) -> None:
        self.mean = mean.to(torch.float32)
        self.components = components.to(torch.float32)
        self.feature_dim = int(self.components.size(0))

    @classmethod
    def fit(cls, vectors: torch.Tensor) -> "PCAFeaturizer":
        vectors = vectors.to(torch.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be rank-2")
        dim = int(vectors.size(1))
        if dim <= 1:
            return cls(torch.zeros(dim, dtype=torch.float32), torch.eye(dim, dtype=torch.float32))
        mean = vectors.mean(dim=0)
        centered = vectors - mean
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        return cls(mean, vh)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.unsqueeze(0)) @ self.components.transpose(0, 1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.mean.unsqueeze(0) + z @ self.components

    def intervene(self, base: torch.Tensor, source: torch.Tensor, mask: torch.Tensor, lambda_scale: float) -> torch.Tensor:
        z_base = self.encode(base)
        z_source = self.encode(source)
        z_mix = z_base + float(lambda_scale) * mask.unsqueeze(0) * (z_source - z_base)
        return self.decode(z_mix)


class _SparseAutoencoder(nn.Module):
    def __init__(self, dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return z, recon


class SAEFeaturizer:
    def __init__(self, mean: torch.Tensor, encoder_weight: torch.Tensor, encoder_bias: torch.Tensor, decoder_weight: torch.Tensor, decoder_bias: torch.Tensor) -> None:
        self.mean = mean.to(torch.float32)
        self.encoder_weight = encoder_weight.to(torch.float32)
        self.encoder_bias = encoder_bias.to(torch.float32)
        self.decoder_weight = decoder_weight.to(torch.float32)
        self.decoder_bias = decoder_bias.to(torch.float32)
        self.feature_dim = int(self.encoder_weight.size(0))

    @classmethod
    def fit(
        cls,
        vectors: torch.Tensor,
        *,
        latent_mult: int,
        lr: float,
        l1_coeff: float,
        epochs: int,
        seed: int,
    ) -> "SAEFeaturizer":
        vectors = vectors.to(torch.float32)
        dim = int(vectors.size(1))
        if dim <= 1:
            return cls(
                mean=torch.zeros(dim, dtype=torch.float32),
                encoder_weight=torch.eye(dim, dtype=torch.float32),
                encoder_bias=torch.zeros(dim, dtype=torch.float32),
                decoder_weight=torch.eye(dim, dtype=torch.float32),
                decoder_bias=torch.zeros(dim, dtype=torch.float32),
            )
        mean = vectors.mean(dim=0)
        centered = vectors - mean
        latent_dim = max(int(dim), int(dim) * int(latent_mult))
        model = _SparseAutoencoder(dim, latent_dim)
        torch.manual_seed(int(seed))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
        for _ in range(int(epochs)):
            optimizer.zero_grad(set_to_none=True)
            z, recon = model(centered)
            loss = torch.mean((recon - centered) ** 2) + float(l1_coeff) * z.abs().mean()
            loss.backward()
            optimizer.step()
        return cls(
            mean=mean.detach().cpu(),
            encoder_weight=model.encoder.weight.detach().cpu(),
            encoder_bias=model.encoder.bias.detach().cpu(),
            decoder_weight=model.decoder.weight.detach().cpu(),
            decoder_bias=model.decoder.bias.detach().cpu(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - self.mean.unsqueeze(0)
        return torch.relu(centered @ self.encoder_weight.transpose(0, 1) + self.encoder_bias.unsqueeze(0))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.mean.unsqueeze(0) + z @ self.decoder_weight.transpose(0, 1) + self.decoder_bias.unsqueeze(0)

    def intervene(self, base: torch.Tensor, source: torch.Tensor, mask: torch.Tensor, lambda_scale: float) -> torch.Tensor:
        z_base = self.encode(base)
        z_source = self.encode(source)
        z_mix = z_base + float(lambda_scale) * mask.unsqueeze(0) * (z_source - z_base)
        rec_base = self.decode(z_base)
        rec_mix = self.decode(z_mix)
        return rec_mix + (base - rec_base)


class DBMMask(nn.Module):
    def __init__(self, feature_dim: int, init_logit: float = -1.5) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.full((int(feature_dim),), float(init_logit), dtype=torch.float32))

    def values(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)


def collect_site_vectors(
    site: Site,
    examples,
    *,
    run_cache: RunCache,
    hidden_size: int,
) -> torch.Tensor:
    dim = _site_dim(site, int(hidden_size))
    vectors = []
    for example in examples:
        run = run_cache.get_run(example)
        if isinstance(site, OutputLogitSite):
            idx = int(site.output_index)
            vectors.append(run.output_logits[[idx]].to(torch.float32))
        else:
            step = int(site.timestep)
            chunk = _site_chunk(run.hidden_states[step].unsqueeze(0).to(torch.float32), site)[0]
            vectors.append(chunk)
    if not vectors:
        return torch.empty((0, dim), dtype=torch.float32)
    return torch.stack(vectors, dim=0)


def fit_featurizer(
    method: str,
    site: Site,
    examples,
    *,
    run_cache: RunCache,
    hidden_size: int,
    config: MIBBaselineConfig,
) -> object:
    dim = _site_dim(site, int(hidden_size))
    if method == "full_vector":
        return IdentityFeaturizer(dim)
    if method == "dbm":
        return IdentityFeaturizer(dim)
    if method == "dbm_pca":
        vectors = collect_site_vectors(site, examples, run_cache=run_cache, hidden_size=int(hidden_size))
        return PCAFeaturizer.fit(vectors)
    if method == "dbm_sae":
        vectors = collect_site_vectors(site, examples, run_cache=run_cache, hidden_size=int(hidden_size))
        return SAEFeaturizer.fit(
            vectors,
            latent_mult=int(config.sae_latent_mult),
            lr=float(config.sae_lr),
            l1_coeff=float(config.sae_l1),
            epochs=int(config.sae_epochs),
            seed=int(config.seed),
        )
    raise ValueError(f"unknown baseline method: {method!r}")


def _mask_tensor(mask: torch.Tensor | None, feature_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if mask is None:
        return torch.ones(feature_dim, device=device, dtype=dtype)
    return mask.to(device=device, dtype=dtype)


def run_featurized_intervention_batch(
    model: GRUAdder,
    records,
    site: Site,
    featurizer,
    *,
    mask: torch.Tensor | None,
    lambda_scale: float,
    device: torch.device,
    run_cache: RunCache,
) -> torch.Tensor:
    if not records:
        return torch.empty((0, model.width + 1), dtype=torch.float32, device=device)

    if isinstance(site, OutputLogitSite):
        base_logits = torch.stack([run_cache.get_run(rec.base).output_logits for rec in records], dim=0).to(device=device)
        source_logits = torch.stack([run_cache.get_run(rec.source).output_logits for rec in records], dim=0).to(device=device)
        idx = int(site.output_index)
        updated = featurizer.intervene(
            base_logits[:, [idx]],
            source_logits[:, [idx]],
            _mask_tensor(mask, featurizer.feature_dim, device, base_logits.dtype),
            lambda_scale=float(lambda_scale),
        )
        out = base_logits.clone()
        out[:, idx] = updated[:, 0]
        return out

    base_x = torch.cat([run_cache.get_input(rec.base) for rec in records], dim=0).to(device=device)
    source_states = torch.stack([run_cache.get_run(rec.source).hidden_states for rec in records], dim=0).to(device=device)
    h = torch.zeros(base_x.size(0), model.hidden_size, device=device, dtype=base_x.dtype)
    sum_logits = []
    target_step = int(site.timestep)
    mask_t = _mask_tensor(mask, featurizer.feature_dim, device, base_x.dtype)
    for step in range(model.width):
        h = model.cell(base_x[:, step, :], h)
        if step == target_step:
            base_chunk = _site_chunk(h, site)
            source_chunk = _site_chunk(source_states[:, step, :], site)
            updated = featurizer.intervene(base_chunk, source_chunk, mask_t, lambda_scale=float(lambda_scale))
            h = _replace_site_chunk(h, site, updated)
        sum_logits.append(model.sum_head(h))
    carry_logit = model.final_carry_head(h)
    return torch.cat(sum_logits + [carry_logit], dim=1)


def _sample_records(records, *, seed: int, count: int):
    if len(records) <= int(count):
        return list(records)
    rng = random.Random(int(seed))
    idx = rng.sample(range(len(records)), int(count))
    return [records[i] for i in idx]


def train_dbm_mask(
    model: GRUAdder,
    records,
    site: Site,
    featurizer,
    *,
    lr: float,
    l1_coeff: float,
    config: MIBBaselineConfig,
    device: torch.device,
    run_cache: RunCache,
    seed: int,
) -> torch.Tensor:
    mask_model = DBMMask(featurizer.feature_dim).to(device)
    optimizer = torch.optim.Adam(mask_model.parameters(), lr=float(lr))
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(int(config.mask_epochs)):
        sampled = _sample_records(
            records,
            seed=int(seed) + 1009 * epoch + int(round(lr * 1e5)) + int(round(l1_coeff * 1e6)),
            count=int(config.mask_train_records_per_epoch),
        )
        for start in range(0, len(sampled), int(config.mask_batch_size)):
            batch_records = sampled[start : start + int(config.mask_batch_size)]
            targets = torch.tensor([rec.counterfactual.output_bits_lsb for rec in batch_records], dtype=torch.float32, device=device)
            logits = run_featurized_intervention_batch(
                model,
                batch_records,
                site,
                featurizer,
                mask=mask_model.values(),
                lambda_scale=1.0,
                device=device,
                run_cache=run_cache,
            )
            loss = criterion(logits, targets) + float(l1_coeff) * mask_model.values().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return mask_model.values().detach().cpu()


def exact_match_rate(
    model: GRUAdder,
    records,
    site: Site,
    featurizer,
    *,
    mask: torch.Tensor | None,
    lambda_scale: float,
    device: torch.device,
    run_cache: RunCache,
    batch_size: int,
) -> float:
    if not records:
        return 0.0
    hits = 0
    for start in range(0, len(records), int(batch_size)):
        batch_records = records[start : start + int(batch_size)]
        logits = run_featurized_intervention_batch(
            model,
            batch_records,
            site,
            featurizer,
            mask=mask,
            lambda_scale=float(lambda_scale),
            device=device,
            run_cache=run_cache,
        )
        pred = (torch.sigmoid(logits) >= 0.5).to(torch.int64).cpu()
        tgt = torch.tensor([rec.counterfactual.output_bits_lsb for rec in batch_records], dtype=torch.int64)
        hits += int((pred == tgt).all(dim=1).sum().item())
    return float(hits / len(records))
