from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .interventions import RunCache
from .scm import BinaryAdditionExample


@dataclass(frozen=True)
class RotatedBasis:
    rotation: torch.Tensor
    mean: torch.Tensor
    scale: torch.Tensor
    variant: str


def fit_pca_rotations(
    *,
    fit_examples: Sequence[BinaryAdditionExample],
    run_cache: RunCache,
    width: int,
    hidden_size: int,
    variant: str = "uncentered",
) -> tuple[dict[int, RotatedBasis], dict[str, object]]:
    variant = str(variant)
    if variant not in {"uncentered", "centered", "whitened"}:
        raise ValueError(f"unknown PCA variant: {variant!r}")

    rotations: dict[int, RotatedBasis] = {}
    diagnostics: dict[str, object] = {}
    n_fit = int(len(fit_examples))
    for timestep in range(int(width)):
        states = torch.stack(
            [run_cache.get_run(example).hidden_states[int(timestep)] for example in fit_examples],
            dim=0,
        ).to(torch.float32)
        if states.shape != (int(len(fit_examples)), int(hidden_size)):
            raise ValueError(
                f"expected fit hidden matrix {(len(fit_examples), int(hidden_size))} at timestep {timestep}, got {tuple(states.shape)}"
            )
        mean = states.mean(dim=0)
        centered_states = states - mean.unsqueeze(0)
        svd_matrix = states if variant == "uncentered" else centered_states
        _u, singular_values, vh = torch.linalg.svd(svd_matrix, full_matrices=False)
        rotation = vh.transpose(0, 1).contiguous()
        gram = rotation.transpose(0, 1) @ rotation
        if variant == "whitened":
            scale = (singular_values / max(1.0, float(n_fit - 1)) ** 0.5).clamp_min(1e-6)
        else:
            scale = torch.ones_like(singular_values)
        if variant == "uncentered":
            basis_mean = torch.zeros_like(mean)
        else:
            basis_mean = mean
        rotations[int(timestep)] = RotatedBasis(
            rotation=rotation,
            mean=basis_mean,
            scale=scale,
            variant=variant,
        )
        diagnostics[f"h_{timestep}"] = {
            "fit_shape": [int(states.size(0)), int(states.size(1))],
            "variant": variant,
            "singular_values": singular_values.tolist(),
            "mean_norm": float(mean.norm().item()),
            "scale": scale.tolist(),
            "orthogonality_max_abs_error": float((gram - torch.eye(int(hidden_size))).abs().max().item()),
        }
    return rotations, diagnostics
