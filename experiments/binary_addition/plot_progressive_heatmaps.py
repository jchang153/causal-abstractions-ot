from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from experiments.binary_addition.data import enumerate_all_examples, stratified_base_split
from experiments.binary_addition.interventions import build_run_cache
from experiments.binary_addition.model import GRUAdder
from experiments.binary_addition.pca_basis import fit_pca_rotations


ROW_KEYS = ("C1", "C2", "C3")
METHOD_SETS = {
    "default": (
        ("stage_a", "PLOT\nlayers", "PC index"),
        ("plot_pca", "PLOT-PCA", "PC index"),
        ("das_full_timestep", "PLOT-DAS", "PC index"),
        ("das_pca", "PLOT-PCA-DAS", "PC index"),
        ("full_das", "Full DAS", "PC index"),
    ),
    "paper": (
        ("stage_a", "PLOT\nlayers", "PC index"),
        ("plot_canonical", "PLOT", "neuron"),
        ("plot_pca", "PLOT-PCA", "PC index"),
        ("das_full_timestep", "PLOT-DAS", "PC index"),
        ("full_das", "Full DAS", "PC index"),
    ),
}

METHODS = METHOD_SETS["default"]

MethodSpec = tuple[str, str, str]

METHOD_LOOKUP = (
    ("stage_a", "PLOT\nlayers", "PC index"),
    ("plot_canonical", "PLOT", "neuron"),
    ("plot_pca", "PLOT-PCA", "PC index"),
    ("das_full_timestep", "PLOT-DAS", "PC index"),
    ("das_pca", "PLOT-PCA-DAS", "PC index"),
    ("full_das", "Full DAS", "PC index"),
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot progressive PLOT/DAS heatmaps for the GRU binary-addition runs.")
    ap.add_argument(
        "--run-dir",
        type=str,
        default=str(Path("eval") / "codex_progressive_plot_10seed"),
        help="Directory containing h8/h16 seed summaries.",
    )
    ap.add_argument("--hidden-size", type=int, required=True, choices=[8, 16])
    ap.add_argument("--seeds", type=str, default="0", help="Comma-separated seed list, or 'all'.")
    ap.add_argument("--out-dir", type=str, default="", help="Output directory. Defaults to <run-dir>/figures.")
    ap.add_argument("--formats", type=str, default="png,pdf")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--layout", type=str, default="standard", choices=["standard", "square"])
    ap.add_argument("--method-set", type=str, default="default", choices=sorted(METHOD_SETS))
    ap.add_argument(
        "--stage-a-source-seed",
        type=str,
        default="",
        help="Optional seed to use only for the leftmost Stage-A PLOT layers panel.",
    )
    ap.add_argument("--cell-mm", type=float, default=3.0, help="Cell size for --layout square.")
    ap.add_argument("--row-gap-mm", type=float, default=1.5, help="Row gap for --layout square.")
    ap.add_argument("--col-gap-mm", type=float, default=2.0, help="Column gap for --layout square.")
    ap.add_argument("--no-title", action="store_true")
    ap.add_argument(
        "--plot-canonical-run-dir",
        type=str,
        default="",
        help=(
            "Optional run directory from run_progressive_plot_stage_b_resolution_sweep.py. "
            "When set, the PLOT column shows those post-calibration handles instead of "
            "the canonical handles stored in the base progressive summary."
        ),
    )
    ap.add_argument(
        "--plot-canonical-resolutions",
        type=str,
        default="",
        help="Optional comma-separated resolution subset to use from --plot-canonical-run-dir, e.g. 1,2.",
    )
    ap.add_argument(
        "--das-display-basis",
        type=str,
        default="pca",
        choices=["pca", "canonical", "das_rotated"],
        help="Basis used to visualize canonical DAS projectors.",
    )
    return ap.parse_args()


def _parse_seeds(text: str, seed_root: Path) -> tuple[int, ...]:
    if str(text).strip().lower() == "all":
        seeds = []
        for path in seed_root.glob("seed_*"):
            match = re.fullmatch(r"seed_(\d+)", path.name)
            if match and (path / "progressive_seed_summary.json").exists():
                seeds.append(int(match.group(1)))
        return tuple(sorted(seeds))
    return tuple(int(x.strip()) for x in str(text).split(",") if x.strip())


def _timestep_from_site(site_key: str) -> int | None:
    if not site_key.startswith("h_"):
        return None
    return int(site_key.split("[", 1)[0].split("_", 1)[1])


def _indices_from_site(site_key: str, hidden_size: int, *, pca: bool) -> tuple[int, ...]:
    if "[" not in site_key:
        return tuple(range(hidden_size))
    inner = site_key.split("[", 1)[1].rstrip("]")
    if pca and ":" in inner:
        inner = inner.split(":", 1)[1]
    if not inner:
        return tuple()
    return tuple(int(part) for part in inner.split(",") if part)


def _stage_a_matrix(summary: dict[str, object], row_key: str, hidden_size: int) -> np.ndarray:
    row = summary["stages"]["stage_a"]["best_trial"]["test"]["per_row"][row_key]
    mass = np.asarray(row["row_mass"], dtype=float)
    mat = np.zeros((4, hidden_size), dtype=float)
    for t, value in enumerate(mass[:4]):
        mat[t, :] = float(value)
    return mat


def _plot_pca_matrix(summary: dict[str, object], row_key: str, hidden_size: int) -> np.ndarray:
    stage = summary["stages"]["stage_b_pca_ot"]
    row = stage["best_trial"]["test"]["per_row"][row_key]
    mat = np.zeros((4, hidden_size), dtype=float)
    selected_sites = list(row.get("selected_sites", []))
    if selected_sites:
        for site_record in selected_sites:
            site_key = str(site_record["site_key"])
            timestep = _timestep_from_site(site_key)
            if timestep is None:
                continue
            indices = _indices_from_site(site_key, hidden_size, pca=True)
            if not indices:
                continue
            for idx in indices:
                mat[int(timestep), int(idx)] += float(site_record["weight"])
        return mat

    for site_key, mass in zip(stage["sites"], row["row_mass"]):
        timestep = _timestep_from_site(str(site_key))
        if timestep is None:
            continue
        indices = _indices_from_site(str(site_key), hidden_size, pca=True)
        if not indices:
            continue
        share = float(mass) / float(len(indices))
        for idx in indices:
            mat[int(timestep), int(idx)] += share
    return mat


def _plot_canonical_matrix(summary: dict[str, object], row_key: str, hidden_size: int) -> np.ndarray:
    stage = summary["stages"]["stage_b_canonical_ot"]
    row = stage["best_trial"]["test"]["per_row"][row_key]
    mat = np.zeros((4, hidden_size), dtype=float)
    selected_sites = list(row.get("selected_sites", []))
    if selected_sites:
        for site_record in selected_sites:
            site_key = str(site_record["site_key"])
            timestep = _timestep_from_site(site_key)
            if timestep is None:
                continue
            indices = _indices_from_site(site_key, hidden_size, pca=False)
            if not indices:
                continue
            for idx in indices:
                mat[int(timestep), int(idx)] += float(site_record["weight"])
        return mat

    for site_key, mass in zip(stage["sites"], row["row_mass"]):
        timestep = _timestep_from_site(str(site_key))
        if timestep is None:
            continue
        indices = _indices_from_site(str(site_key), hidden_size, pca=False)
        if not indices:
            continue
        share = float(mass) / float(len(indices))
        for idx in indices:
            mat[int(timestep), int(idx)] += share
    return mat


def _parse_resolution_subset(text: str) -> tuple[str, ...] | None:
    if not str(text).strip():
        return None
    return tuple(str(int(chunk.strip())) for chunk in str(text).split(",") if chunk.strip())


def _resolution_sweep_summary_path(run_dir: Path, hidden_size: int, seed: int) -> Path:
    candidates = (
        run_dir / f"h{hidden_size}" / f"seed_{seed}" / "resolution_separate_topk_seed_summary.json",
        run_dir / f"seed_{seed}" / "resolution_separate_topk_seed_summary.json",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(candidates[0])


def _best_calibrated_resolution_row(
    summary: dict[str, object],
    row_key: str,
    *,
    resolutions: tuple[str, ...] | None,
) -> dict[str, object]:
    allowed = set(resolutions) if resolutions is not None else None
    best_row = None
    best_key = None
    for resolution, stage in summary["resolution_results"].items():
        if allowed is not None and str(resolution) not in allowed:
            continue
        for trial in stage["trials"]:
            row = trial["test"]["per_row"][row_key]
            calibration = row["calibration"]
            key = (
                float(calibration["combined"]),
                float(calibration["sensitivity"]),
                float(calibration["invariance"]),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_row = row
    if best_row is None:
        raise RuntimeError(f"no calibrated row found for {row_key} with resolutions={resolutions}")
    return best_row


def _plot_canonical_matrix_from_resolution_sweep(
    summary: dict[str, object],
    row_key: str,
    hidden_size: int,
    *,
    resolutions: tuple[str, ...] | None,
) -> np.ndarray:
    row = _best_calibrated_resolution_row(summary, row_key, resolutions=resolutions)
    mat = np.zeros((4, hidden_size), dtype=float)
    for site_record in row.get("selected_sites", []):
        site_key = str(site_record["site_key"])
        timestep = _timestep_from_site(site_key)
        if timestep is None:
            continue
        indices = _indices_from_site(site_key, hidden_size, pca=False)
        if not indices:
            continue
        for idx in indices:
            mat[int(timestep), int(idx)] += float(site_record["weight"])
    return mat


def _canonical_resolution_sweep_matrices(
    run_dir: Path,
    seeds: Sequence[int],
    hidden_size: int,
    *,
    resolutions: tuple[str, ...] | None,
) -> list[np.ndarray]:
    per_seed = []
    for seed in seeds:
        path = _resolution_sweep_summary_path(run_dir, hidden_size, int(seed))
        summary = json.loads(path.read_text(encoding="utf-8"))
        per_seed.append(
            [
                _plot_canonical_matrix_from_resolution_sweep(
                    summary,
                    row_key,
                    hidden_size,
                    resolutions=resolutions,
                )
                for row_key in ROW_KEYS
            ]
        )
    if len(per_seed) == 1:
        return per_seed[0]
    return [np.mean(np.stack([item[row_idx] for item in per_seed], axis=0), axis=0) for row_idx in range(len(ROW_KEYS))]


def _best_das_trial(stage: dict[str, object], row_info: dict[str, object]) -> dict[str, object]:
    target_support = row_info["support"]
    for trial in stage["trials"]:
        if str(trial["row_key"]) != str(row_info.get("row_key", "")) and "row_key" in row_info:
            continue
        if trial["support"] != target_support:
            continue
        if int(trial["subspace_dim"]) != int(row_info["subspace_dim"]):
            continue
        if abs(float(trial["lr"]) - float(row_info["lr"])) > 1e-12:
            continue
        if abs(float(trial["lambda"]) - float(row_info["lambda"])) > 1e-12:
            continue
        return trial
    for trial in stage["trials"]:
        if trial["support"] == target_support and int(trial["subspace_dim"]) == int(row_info["subspace_dim"]):
            return trial
    raise RuntimeError(f"could not find matching DAS trial for {target_support}")


def _das_loadings_from_trial(trial: dict[str, object]) -> np.ndarray:
    proj = np.asarray(trial["rotator_state"]["proj"], dtype=float)
    if proj.ndim != 2:
        raise ValueError("DAS projection matrix must be 2D")
    subspace_dim = int(trial["subspace_dim"])
    q, _ = np.linalg.qr(proj, mode="reduced")
    q = q[:, :subspace_dim]
    loadings = np.sum(q * q, axis=1)
    total = float(loadings.sum())
    if total > 0:
        loadings = loadings / total
    return loadings


def _das_projector_from_trial(trial: dict[str, object]) -> np.ndarray:
    proj = np.asarray(trial["rotator_state"]["proj"], dtype=float)
    if proj.ndim != 2:
        raise ValueError("DAS projection matrix must be 2D")
    subspace_dim = int(trial["subspace_dim"])
    q, _ = np.linalg.qr(proj, mode="reduced")
    q = q[:, :subspace_dim]
    return q @ q.T


def _pca_rotations_for_summary(summary: dict[str, object]) -> dict[int, np.ndarray]:
    config = summary["config"]
    checkpoint = Path(str(config["checkpoint"]))
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model = GRUAdder(
        width=int(config["width"]),
        hidden_size=int(config["hidden_size"]),
    )
    model.load_state_dict(state)
    model.eval()

    examples = enumerate_all_examples(width=int(config["width"]))
    split = stratified_base_split(
        examples,
        fit_count=int(config["fit_bases"]),
        calib_count=int(config["calib_bases"]),
        test_count=int(config["test_bases"]),
        seed=int(config["seed"]),
    )
    run_cache = build_run_cache(model, examples, device=torch.device("cpu"))
    rotation_map, _diagnostics = fit_pca_rotations(
        fit_examples=split.fit,
        run_cache=run_cache,
        width=int(config["width"]),
        hidden_size=int(config["hidden_size"]),
        variant=str(config.get("pca_variant", "centered")),
    )
    return {int(t): basis.rotation.detach().cpu().numpy() for t, basis in rotation_map.items()}


def _das_matrix(
    summary: dict[str, object],
    row_key: str,
    hidden_size: int,
    stage_key: str,
    *,
    pca_rotations: dict[int, np.ndarray] | None,
    display_basis: str,
) -> np.ndarray:
    stage = summary["stages"][stage_key]
    row_info = dict(stage["test"]["per_row"][row_key])
    row_info["row_key"] = row_key
    trial = _best_das_trial(stage, row_info)
    support = row_info["support"]
    timestep = int(support["timestep"])
    indices = tuple(int(i) for i in support["indices"])
    mat = np.zeros((4, hidden_size), dtype=float)
    if str(display_basis) == "das_rotated":
        active = min(int(row_info["subspace_dim"]), hidden_size)
        mat[timestep, :active] = 1.0
        return mat

    if str(display_basis) == "pca":
        support_basis = str(support["basis"])
        projector = _das_projector_from_trial(trial)
        if support_basis == "pca":
            pc_projector = np.zeros((hidden_size, hidden_size), dtype=float)
            for local_i, coord_i in enumerate(indices):
                for local_j, coord_j in enumerate(indices):
                    if coord_i < hidden_size and coord_j < hidden_size:
                        pc_projector[coord_i, coord_j] = projector[local_i, local_j]
            loadings = np.diag(pc_projector)
        elif support_basis == "canonical":
            if pca_rotations is None or timestep not in pca_rotations:
                raise RuntimeError(f"missing PCA rotation for timestep h_{timestep}")
            canonical_projector = np.zeros((hidden_size, hidden_size), dtype=float)
            for local_i, coord_i in enumerate(indices):
                for local_j, coord_j in enumerate(indices):
                    if coord_i < hidden_size and coord_j < hidden_size:
                        canonical_projector[coord_i, coord_j] = projector[local_i, local_j]
            rotation = pca_rotations[timestep]
            loadings = np.diag(rotation.T @ canonical_projector @ rotation)
        else:
            raise ValueError(f"unknown support basis: {support_basis!r}")
        total = float(loadings.sum())
        if total > 0:
            loadings = loadings / total
        mat[timestep, :] = np.maximum(loadings, 0.0)
        return mat

    loadings = _das_loadings_from_trial(trial)
    for local_idx, coord in enumerate(indices):
        if coord < hidden_size and local_idx < len(loadings):
            mat[timestep, coord] = float(loadings[local_idx])
    return mat


def _matrices_for_summary(
    summary: dict[str, object],
    hidden_size: int,
    *,
    display_basis: str,
) -> dict[str, list[np.ndarray]]:
    pca_rotations = _pca_rotations_for_summary(summary) if str(display_basis) == "pca" else None
    return {
        "stage_a": [_stage_a_matrix(summary, row_key, hidden_size) for row_key in ROW_KEYS],
        "plot_canonical": [_plot_canonical_matrix(summary, row_key, hidden_size) for row_key in ROW_KEYS],
        "plot_pca": [_plot_pca_matrix(summary, row_key, hidden_size) for row_key in ROW_KEYS],
        "das_full_timestep": [
            _das_matrix(
                summary,
                row_key,
                hidden_size,
                "stage_b_das_full_timestep",
                pca_rotations=pca_rotations,
                display_basis=display_basis,
            )
            for row_key in ROW_KEYS
        ],
        "das_pca": [
            _das_matrix(
                summary,
                row_key,
                hidden_size,
                "stage_b_das_pca_support",
                pca_rotations=pca_rotations,
                display_basis=display_basis,
            )
            for row_key in ROW_KEYS
        ],
        "full_das": [
            _das_matrix(
                summary,
                row_key,
                hidden_size,
                "full_das",
                pca_rotations=pca_rotations,
                display_basis=display_basis,
            )
            for row_key in ROW_KEYS
        ],
    }


def _average_matrices(
    items: Sequence[dict[str, list[np.ndarray]]],
    *,
    methods: Sequence[MethodSpec],
) -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = {}
    for method, _, _ in methods:
        out[method] = []
        for row_idx in range(len(ROW_KEYS)):
            stack = np.stack([item[method][row_idx] for item in items], axis=0)
            out[method].append(np.mean(stack, axis=0))
    return out


def _plot_grid(
    matrices: dict[str, list[np.ndarray]],
    *,
    methods: Sequence[MethodSpec],
    hidden_size: int,
    das_display_basis: str,
    title: str,
    out_base: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        nrows=len(ROW_KEYS),
        ncols=len(methods),
        figsize=(18.5 if hidden_size == 8 else 22.5, 7.2),
        sharey=True,
        constrained_layout=False,
    )
    fig.suptitle(title, y=0.985, fontsize=16)
    plt.subplots_adjust(left=0.055, right=0.955, top=0.90, bottom=0.09, wspace=0.36, hspace=0.34)

    col_vmax = {}
    for method, _, _ in methods:
        vmax = max(float(np.max(mat)) for mat in matrices[method])
        col_vmax[method] = vmax if vmax > 0 else 1.0

    ims = []
    xticks = list(range(hidden_size)) if hidden_size <= 8 else list(range(0, hidden_size, 2)) + [hidden_size - 1]
    xticks = sorted(set(xticks))
    for col, (method, label, xlabel) in enumerate(methods):
        if method.startswith("das") or method == "full_das":
            xlabel = "rotated index" if str(das_display_basis) == "das_rotated" else xlabel
        ims.append(None)
        for row, row_key in enumerate(ROW_KEYS):
            ax = axes[row, col]
            im = ax.imshow(
                matrices[method][row],
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
                vmin=0.0,
                vmax=col_vmax[method],
            )
            ims[col] = im
            if row == 0:
                ax.set_title(label, fontsize=12, pad=8)
            if col == 0:
                ax.set_yticks(range(4))
                ax.set_yticklabels([rf"$h_{t}$" for t in range(4)], fontsize=10)
                ax.set_ylabel(rf"${row_key}$", fontsize=16, rotation=0, labelpad=24, va="center")
            else:
                ax.set_yticks(range(4))
                ax.tick_params(axis="y", labelleft=False)
            ax.set_xticks(xticks)
            if row == len(ROW_KEYS) - 1:
                ax.set_xlabel(xlabel, fontsize=10)
                ax.set_xticklabels([str(i) for i in xticks], fontsize=8)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

    for col, im in enumerate(ims):
        if im is None:
            continue
        top = axes[0, col].get_position().y1
        bottom = axes[-1, col].get_position().y0
        right = axes[0, col].get_position().x1
        cax = fig.add_axes([right + 0.006, bottom, 0.0045, top - bottom])
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=9)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_base.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_square_grid(
    matrices: dict[str, list[np.ndarray]],
    *,
    methods: Sequence[MethodSpec],
    hidden_size: int,
    das_display_basis: str,
    title: str,
    out_base: Path,
    formats: Sequence[str],
    dpi: int,
    cell_mm: float,
    row_gap_mm: float,
    col_gap_mm: float,
) -> None:
    mm = 1.0 / 25.4
    n_rows = len(ROW_KEYS)
    n_cols = len(methods)
    matrix_rows = 4

    cell_w = float(cell_mm) * mm
    tile_w = int(hidden_size) * cell_w
    tile_h = matrix_rows * cell_w
    row_gap = float(row_gap_mm) * mm
    col_gap = float(col_gap_mm) * mm
    cbar_pad = 1.5 * mm
    cbar_w = 1.0 * mm

    left = 20.0 * mm
    right = 4.0 * mm
    bottom = 11.0 * mm
    top = (17.0 if title else 11.0) * mm
    group_w = tile_w
    stack_h = n_rows * tile_h + (n_rows - 1) * row_gap
    fig_w = left + n_cols * group_w + (n_cols - 1) * col_gap + cbar_pad + cbar_w + right
    fig_h = bottom + stack_h + top

    fig = plt.figure(figsize=(fig_w, fig_h))
    if title:
        fig.suptitle(title, y=0.99, fontsize=13)

    col_vmax = {}
    for method, _, _ in methods:
        vmax = max(float(np.max(mat)) for mat in matrices[method])
        col_vmax[method] = vmax if vmax > 0 else 1.0

    xticks = list(range(hidden_size)) if hidden_size <= 8 else list(range(0, hidden_size, 2)) + [hidden_size - 1]
    xticks = sorted(set(xticks))
    last_im = None
    for col, (method, label, xlabel) in enumerate(methods):
        if method.startswith("das") or method == "full_das":
            xlabel = "rotated index" if str(das_display_basis) == "das_rotated" else xlabel
        x0 = left + col * (group_w + col_gap)
        for row, row_key in enumerate(ROW_KEYS):
            y0 = bottom + (n_rows - 1 - row) * (tile_h + row_gap)
            ax = fig.add_axes([x0 / fig_w, y0 / fig_h, tile_w / fig_w, tile_h / fig_h])
            im = ax.imshow(
                matrices[method][row],
                aspect="equal",
                interpolation="nearest",
                cmap="viridis",
                vmin=0.0,
                vmax=col_vmax[method],
            )
            last_im = im
            if row == 0:
                ax.set_title(label, fontsize=9.5, pad=3)
            if col == 0:
                ax.set_yticks(range(4))
                ax.set_yticklabels([rf"$h_{t}$" for t in range(4)], fontsize=7)
                ax.set_ylabel(rf"${row_key}$", fontsize=9.5, rotation=0, labelpad=17, va="center")
            else:
                ax.set_yticks(range(4))
                ax.tick_params(axis="y", labelleft=False)
            ax.set_xticks(xticks)
            if row == n_rows - 1:
                ax.set_xlabel(xlabel, fontsize=8, labelpad=2)
                ax.set_xticklabels([str(i) for i in xticks], fontsize=6.5)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            ax.tick_params(length=2.5, width=0.6)
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)

    if last_im is not None:
        cax = fig.add_axes(
            [
                (left + n_cols * group_w + (n_cols - 1) * col_gap + cbar_pad) / fig_w,
                bottom / fig_h,
                cbar_w / fig_w,
                stack_h / fig_h,
            ]
        )
        legend_mappable = mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=0.0, vmax=0.8),
            cmap="viridis",
        )
        legend_mappable.set_array([])
        cbar = fig.colorbar(legend_mappable, cax=cax)
        cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8])
        cbar.ax.tick_params(labelsize=6.5, length=2.0, width=0.5)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_base.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    seed_root = run_dir / f"h{args.hidden_size}"
    seeds = _parse_seeds(args.seeds, seed_root)
    if not seeds:
        raise RuntimeError(f"no seeds found under {seed_root}")

    summaries = []
    for seed in seeds:
        path = seed_root / f"seed_{seed}" / "progressive_seed_summary.json"
        if not path.exists():
            raise FileNotFoundError(path)
        summaries.append(json.loads(path.read_text(encoding="utf-8")))

    methods = METHOD_SETS[str(args.method_set)]
    all_matrices = [
        _matrices_for_summary(summary, int(args.hidden_size), display_basis=str(args.das_display_basis))
        for summary in summaries
    ]
    matrices = all_matrices[0] if len(all_matrices) == 1 else _average_matrices(all_matrices, methods=METHOD_LOOKUP)

    stage_a_source_seed = str(args.stage_a_source_seed).strip()
    if stage_a_source_seed:
        stage_a_path = seed_root / f"seed_{int(stage_a_source_seed)}" / "progressive_seed_summary.json"
        if not stage_a_path.exists():
            raise FileNotFoundError(stage_a_path)
        stage_a_summary = json.loads(stage_a_path.read_text(encoding="utf-8"))
        stage_a_matrices = _matrices_for_summary(
            stage_a_summary,
            int(args.hidden_size),
            display_basis=str(args.das_display_basis),
        )
        matrices["stage_a"] = stage_a_matrices["stage_a"]

    plot_canonical_run_dir = str(args.plot_canonical_run_dir).strip()
    if plot_canonical_run_dir:
        matrices["plot_canonical"] = _canonical_resolution_sweep_matrices(
            Path(plot_canonical_run_dir),
            seeds,
            int(args.hidden_size),
            resolutions=_parse_resolution_subset(str(args.plot_canonical_resolutions)),
        )

    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else run_dir / "figures"
    suffix = f"seed{seeds[0]}" if len(seeds) == 1 else f"mean{len(seeds)}seeds"
    if stage_a_source_seed:
        suffix += f"_stageAseed{int(stage_a_source_seed)}"
    basis_suffix = {
        "pca": "das-pca-view",
        "canonical": "das-canonical-view",
        "das_rotated": "das-rotated-view",
    }[str(args.das_display_basis)]
    layout_suffix = "_square" if str(args.layout) == "square" else ""
    method_suffix = "" if str(args.method_set) == "default" else f"_{args.method_set}"
    out_base = out_dir / f"progressive_heatmaps_h{args.hidden_size}_{suffix}_{basis_suffix}{layout_suffix}"
    if method_suffix:
        out_base = out_dir / f"progressive_heatmaps_h{args.hidden_size}_{suffix}_{basis_suffix}{layout_suffix}{method_suffix}"
    title = (
        f"GRU binary addition, h={args.hidden_size}, seed {seeds[0]}"
        if len(seeds) == 1
        else f"GRU binary addition, h={args.hidden_size}, mean over {len(seeds)} seeds"
    )
    if bool(args.no_title):
        title = ""
    formats = tuple(fmt.strip().lstrip(".") for fmt in str(args.formats).split(",") if fmt.strip())
    if str(args.layout) == "square":
        _plot_square_grid(
            matrices,
            methods=methods,
            hidden_size=int(args.hidden_size),
            das_display_basis=str(args.das_display_basis),
            title=title,
            out_base=out_base,
            formats=formats,
            dpi=int(args.dpi),
            cell_mm=float(args.cell_mm),
            row_gap_mm=float(args.row_gap_mm),
            col_gap_mm=float(args.col_gap_mm),
        )
    else:
        _plot_grid(
            matrices,
            methods=methods,
            hidden_size=int(args.hidden_size),
            das_display_basis=str(args.das_display_basis),
            title=title,
            out_base=out_base,
            formats=formats,
            dpi=int(args.dpi),
        )
    print(out_base)


if __name__ == "__main__":
    main()
