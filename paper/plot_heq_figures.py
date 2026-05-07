from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import FuncNorm
from PIL import Image


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_ot_epsilon_sweep(epsilon_payload: dict, output_path: Path) -> None:
    records = [record for record in epsilon_payload["sweep_records"] if record["method"] == "ot"]
    epsilons = np.array([float(record["epsilon"]) for record in records], dtype=float)
    wx = np.array([float(record["per_variable_exact_acc"]["WX"]) for record in records], dtype=float)
    yz = np.array([float(record["per_variable_exact_acc"]["YZ"]) for record in records], dtype=float)
    avg = np.array([float(record["average_exact_acc"]) for record in records], dtype=float)

    fig, ax = plt.subplots(figsize=(4.1, 2.8), constrained_layout=True)
    ax.plot(epsilons, avg, marker="o", linewidth=2.0, color="#2f2f2f", label="Average")
    ax.plot(epsilons, wx, marker="o", linewidth=1.9, color="#4e79a7", label="$z_{WX}$")
    ax.plot(epsilons, yz, marker="o", linewidth=1.9, color="#e15759", label="$z_{YZ}$")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("Exact accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _extract_strategy(strategy_payload: dict, strategy_name: str) -> dict:
    for strategy in strategy_payload["strategies"]:
        if strategy["name"] == strategy_name:
            return strategy
    raise KeyError(f"Strategy {strategy_name!r} not found")


def _compute_das_projector_cache(repo_dir: Path, strategy_payload: dict, cache_path: Path) -> dict:
    if cache_path.exists():
        return load_json(cache_path)

    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    import torch
    from pyvene import RotatedSpaceIntervention

    from equality_calibration_strategy_sweep import (
        BATCH_SIZE,
        DAS_LEARNING_RATE,
        DAS_MAX_EPOCHS,
        DAS_MIN_EPOCHS,
        DAS_PLATEAU_PATIENCE,
        DAS_PLATEAU_REL_DELTA,
        SEED,
        build_fixed_train_bank,
        ensure_backbone,
        load_equality_problem,
    )
    from equality_experiment.das import train_rotated_intervention
    from equality_experiment.pair_bank import PairBankVariableDataset
    from equality_experiment.pyvene_utils import DASSearchSpec, build_intervenable

    strategy = _extract_strategy(strategy_payload, "shared_balanced_wx_yz_only")
    selected_specs = {}
    for record in strategy["methods"]["das"]["raw_payload"]["results"]:
        selected_specs[str(record["variable"])] = {
            "layer": int(record["layer"]),
            "subspace_dim": int(record["subspace_dim"]),
        }

    torch.manual_seed(SEED)
    device = torch.device("cpu")
    problem = load_equality_problem(num_entities=100, embedding_dim=4)
    model, _, _ = ensure_backbone(problem, device)
    train_bank = build_fixed_train_bank(problem)

    cache_payload = {}
    for variable, spec_meta in selected_specs.items():
        layer = int(spec_meta["layer"])
        subspace_dim = int(spec_meta["subspace_dim"])
        intervention = RotatedSpaceIntervention(embed_dim=int(model.config.hidden_dims[layer]))
        spec = DASSearchSpec(layer=layer, subspace_dim=subspace_dim, component=f"h[{layer}].output")
        intervenable = build_intervenable(
            model=model,
            layer=spec.layer,
            component=spec.component,
            intervention=intervention,
            device=device,
            unit=spec.unit,
            max_units=spec.max_units,
            freeze_model=True,
            freeze_intervention=False,
            use_fast=False,
        )
        dataset = PairBankVariableDataset(train_bank, variable)
        losses = train_rotated_intervention(
            intervenable=intervenable,
            dataset=dataset,
            spec=spec,
            max_epochs=DAS_MAX_EPOCHS,
            learning_rate=DAS_LEARNING_RATE,
            batch_size=BATCH_SIZE,
            device=device,
            plateau_patience=DAS_PLATEAU_PATIENCE,
            plateau_rel_delta=DAS_PLATEAU_REL_DELTA,
            min_epochs=DAS_MIN_EPOCHS,
        )
        selected_intervention = next(iter(intervenable.interventions.values()))
        rotation = selected_intervention.rotate_layer.weight.detach().cpu()
        projector_diag = (rotation[:, :subspace_dim] ** 2).sum(dim=1)
        projector_diag = projector_diag / projector_diag.sum().clamp_min(1e-12)
        cache_payload[variable] = {
            "layer": layer,
            "subspace_dim": subspace_dim,
            "epochs": len(losses),
            "projector_diag": projector_diag.tolist(),
        }

    cache_path.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
    return cache_payload


def _make_tail_detail_norm(exponent: float = 0.68) -> FuncNorm:
    # Allocate more color resolution near both 0 and 1 so weak mass and near-selected peaks
    # separate better than under a one-sided power-law normalization.
    def forward(values):
        values = np.asarray(values, dtype=float)
        clipped = np.clip(values, 0.0, 1.0)
        mapped = np.empty_like(clipped, dtype=float)
        lower = clipped <= 0.5
        mapped[lower] = 0.5 * np.power(2.0 * clipped[lower], exponent)
        mapped[~lower] = 1.0 - 0.5 * np.power(2.0 * (1.0 - clipped[~lower]), exponent)
        return mapped

    def inverse(values):
        values = np.asarray(values, dtype=float)
        clipped = np.clip(values, 0.0, 1.0)
        mapped = np.empty_like(clipped, dtype=float)
        lower = clipped <= 0.5
        mapped[lower] = 0.5 * np.power(2.0 * clipped[lower], 1.0 / exponent)
        mapped[~lower] = 1.0 - 0.5 * np.power(2.0 * (1.0 - clipped[~lower]), 1.0 / exponent)
        return mapped

    return FuncNorm((forward, inverse), vmin=0.0, vmax=1.0)


def plot_handle_summary(strategy_payload: dict, output_path: Path) -> None:
    strategy = _extract_strategy(strategy_payload, "shared_balanced_wx_yz_only")
    methods = ["ot", "uot", "das"]
    method_labels = {"ot": "OT", "uot": "UOT", "das": "DAS"}
    variables = ["WX", "YZ"]
    layer_labels = ["L1", "L2", "L3"]
    variable_colors = {"WX": "#4e79a7", "YZ": "#e15759"}

    layer_mass = {variable: {method: np.zeros(len(layer_labels), dtype=float) for method in methods} for variable in variables}
    transport_candidate_heatmaps = {method: {variable: np.zeros((3, 16), dtype=float) for variable in variables} for method in ["ot", "uot"]}
    transport_topk_heatmaps = {method: {variable: np.zeros((3, 16), dtype=float) for variable in variables} for method in ["ot", "uot"]}
    das_canonical_heatmaps = {variable: np.zeros((3, 16), dtype=float) for variable in variables}
    das_rotated_heatmaps = {variable: np.zeros((3, 16), dtype=float) for variable in variables}

    cache_path = output_path.parent / "heq_das_projector_cache.json"
    repo_dir = Path(__file__).resolve().parent.parent
    das_projector_cache = _compute_das_projector_cache(repo_dir, strategy_payload, cache_path)

    for method in methods:
        method_payload = strategy["methods"][method]["raw_payload"]
        if method in {"ot", "uot"}:
            transport_matrix = np.asarray(method_payload["transport"], dtype=float)
            target_vars = [str(variable) for variable in method_payload["target_vars"]]
            site_labels = [str(label) for label in method_payload["sites"]]
            for target_idx, variable in enumerate(target_vars):
                candidate_values = transport_matrix[target_idx]
                total_mass = float(candidate_values.sum())
                for site_idx, site_label in enumerate(site_labels):
                    layer_idx = int(site_label[1])
                    dim_idx = int(site_label.split("-d", 1)[1])
                    if 0 <= layer_idx < 3 and 0 <= dim_idx < 16:
                        value = float(candidate_values[site_idx])
                        transport_candidate_heatmaps[method][variable][layer_idx, dim_idx] = value
                        if total_mass > 0.0:
                            layer_mass[variable][method][layer_idx] += value / total_mass
        for record in method_payload["results"]:
            variable = str(record["variable"])
            if method == "das":
                layer_index = int(record["layer"])
                layer_mass[variable][method][layer_index] = 1.0
                das_canonical_heatmaps[variable][layer_index, :] = np.asarray(
                    das_projector_cache[variable]["projector_diag"],
                    dtype=float,
                )
                das_rotated_heatmaps[variable][layer_index, : int(record["subspace_dim"])] = 1.0
            else:
                for layer_name, mask_values in method_payload["layer_masks_by_variable"][variable].items():
                    layer_idx = int(layer_name[1:])
                    transport_topk_heatmaps[method][variable][layer_idx, :] = np.asarray(mask_values, dtype=float)

    fig = plt.figure(figsize=(18.3, 3.35), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.12, 3.58], wspace=0.22)
    left_grid = outer[0, 0].subgridspec(3, 1, height_ratios=[0.079, 0.92, 0.001], hspace=0.0)
    ax = fig.add_subplot(left_grid[1, 0])
    right_outer = outer[0, 1].subgridspec(
        4,
        5,
        height_ratios=[0.11, 1.0, 1.0, 1.0],
        width_ratios=[3.55, 0.21, 3.55, 0.08, 0.11],
        wspace=0.006,
        hspace=0.0,
    )

    ordered_pairs = [(method, variable) for method in methods for variable in variables]
    x = np.arange(len(ordered_pairs))
    width = 0.6

    layer_bar_width = 0.18
    min_visible_bar_height = 0.006
    layer_display_labels = ["L1", "L2", "L3"]
    layer_offsets = np.array([-layer_bar_width, 0.0, layer_bar_width], dtype=float)
    for layer_idx, layer_name in enumerate(layer_labels):
        heights = np.array(
            [layer_mass[variable][method][layer_idx] for method, variable in ordered_pairs],
            dtype=float,
        )
        display_heights = np.where(heights > 0.0, heights, min_visible_bar_height)
        facecolors = [variable_colors[variable] for _, variable in ordered_pairs]
        ax.bar(
            x + layer_offsets[layer_idx],
            display_heights,
            width=layer_bar_width,
            color=facecolors,
            alpha=0.35 + 0.28 * layer_idx,
            edgecolor=facecolors,
            linewidth=0.7,
        )
    ax.set_xticks(
        x,
        [rf"{method_labels[method]}" + "\n" + rf"$z_{{{variable}}}$" for method, variable in ordered_pairs],
    )
    ax.set_ylim(0.0, 1.03)
    ax.set_ylabel("Normalized layer mass")
    ax.set_title("Layer weight distribution pre-calibration")
    ax.grid(True, axis="y", alpha=0.2)
    for boundary in [1.5, 3.5]:
        ax.axvline(boundary, color="#999999", linewidth=0.8, alpha=0.5)
    label_x_offsets_by_pair = {
        0: np.array([-0.62 * layer_bar_width, 0.0, 0.5 * layer_bar_width], dtype=float),
        1: np.array([-0.72 * layer_bar_width, -0.22 * layer_bar_width, 0.0], dtype=float),
    }
    label_gap = 0.024
    for pair_idx in [0, 1]:
        group_color = variable_colors[ordered_pairs[pair_idx][1]]
        label_x_offsets = label_x_offsets_by_pair[pair_idx]
        for layer_idx, layer_text in enumerate(layer_display_labels):
            base_height = float(layer_mass[ordered_pairs[pair_idx][1]][ordered_pairs[pair_idx][0]][layer_idx])
            label_height = base_height + label_gap
            ax.text(
                x[pair_idx] + layer_offsets[layer_idx] + label_x_offsets[layer_idx],
                label_height,
                layer_text,
                color=group_color,
                fontsize=8.5,
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    cmap = plt.get_cmap("viridis")
    norm = _make_tail_detail_norm()
    for variable, grid_col in zip(variables, [0, 2]):
        title_ax = fig.add_subplot(right_outer[0, grid_col])
        title_ax.axis("off")
        title_ax.text(
            0.5,
            0.76,
            rf"$z_{{{variable}}}$",
            ha="center",
            va="center",
            fontsize=12,
        )

    image = None
    variable_blocks = {
        "WX": right_outer[1:, 0].subgridspec(3, 2, wspace=0.01, hspace=0.0),
        "YZ": right_outer[1:, 2].subgridspec(3, 2, wspace=0.01, hspace=0.0),
    }
    for row_idx, method in enumerate(methods):
        for variable in variables:
            pair_grid = variable_blocks[variable]
            first_ax = fig.add_subplot(pair_grid[row_idx, 0])
            second_ax = fig.add_subplot(pair_grid[row_idx, 1], sharey=first_ax)

            if method in {"ot", "uot"}:
                first_values = np.asarray(transport_candidate_heatmaps[method][variable], dtype=float)
                second_values = np.asarray(transport_topk_heatmaps[method][variable], dtype=float)
                first_title = "pre-top$K$"
                second_title = "top-$K$"
            else:
                first_values = np.asarray(das_canonical_heatmaps[variable], dtype=float)
                second_values = np.asarray(das_rotated_heatmaps[variable], dtype=float)
                first_title = "canonical"
                second_title = "rotated"

            def rescale_panel(values: np.ndarray) -> np.ndarray:
                values = np.asarray(values, dtype=float)
                vmax = float(values.max())
                vmin = float(values.min())
                if vmax <= 0.0:
                    return values
                if vmax > vmin:
                    return (values - vmin) / (vmax - vmin)
                return values / vmax

            first_values = rescale_panel(first_values)
            second_values = rescale_panel(second_values)
            box_aspect = first_values.shape[0] / first_values.shape[1]
            first_ax.set_box_aspect(box_aspect)
            second_ax.set_box_aspect(box_aspect)

            image = first_ax.imshow(
                first_values,
                cmap=cmap,
                norm=norm,
                aspect="auto",
                interpolation="nearest",
            )
            second_ax.imshow(
                second_values,
                cmap=cmap,
                norm=norm,
                aspect="auto",
                interpolation="nearest",
            )

            for heat_ax in (first_ax, second_ax):
                heat_ax.set_xticks(np.arange(-0.5, 16, 1), minor=True)
                heat_ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
                heat_ax.grid(which="minor", color="white", linewidth=0.32)
                heat_ax.tick_params(which="minor", bottom=False, left=False)
                for spine in heat_ax.spines.values():
                    spine.set_linewidth(0.9)
                    spine.set_edgecolor("#333333")

            if variable == "WX":
                first_ax.set_yticks([0, 1, 2], layer_labels)
                first_ax.set_ylabel(method_labels[method], rotation=0, labelpad=12, va="center", fontsize=11)
            else:
                first_ax.set_yticks([0, 1, 2], [])
            second_ax.set_yticks([0, 1, 2], [])

            if row_idx == len(methods) - 1:
                first_ax.set_xticks([0, 5, 10, 15], ["0", "5", "10", "15"])
                second_ax.set_xticks([0, 5, 10, 15], ["0", "5", "10", "15"])
                first_ax.set_xlabel("Neuron index")
            else:
                first_ax.set_xticks([0, 5, 10, 15], [])
                second_ax.set_xticks([0, 5, 10, 15], [])

            first_ax.set_title(first_title, fontsize=8.5, pad=2.0, color="#444444")
            second_ax.set_title(second_title, fontsize=8.5, pad=2.0, color="#444444")

    cax = fig.add_subplot(right_outer[1:, 4])
    tick_values = [0.0, 0.05, 0.15, 0.5, 0.85, 0.95, 1.0]
    cbar = fig.colorbar(image, cax=cax, ticks=tick_values)
    cbar.ax.set_yticklabels([f"{tick:.2f}".rstrip("0").rstrip(".") for tick in tick_values])
    cbar.ax.tick_params(labelsize=8.5)
    cbar.set_label("Relative site strength", rotation=90)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Export separate crops so the paper can include true LaTeX subfigures.
    image = Image.open(output_path)
    width, height = image.size
    # The combined figure consists of a narrow layer-profile panel on the left
    # and the full heatmap block on the right. Keep the crops disjoint so that
    # subfigure (a) contains only the layer profile.
    left_end = int(round(0.229 * width))
    right_start = int(round(0.244 * width))
    image.crop((0, 0, left_end, height)).save(output_path.with_name("heq_handle_summary_left.png"))
    image.crop((right_start, 0, width, height)).save(output_path.with_name("heq_handle_summary_right.png"))


def plot_handle_heatmaps_stacked(strategy_payload: dict, output_path: Path) -> None:
    strategy = _extract_strategy(strategy_payload, "shared_balanced_wx_yz_only")
    methods = ["ot", "das"]
    method_labels = {"ot": "OT", "das": "DAS"}
    variables = ["WX", "YZ"]
    layer_labels = ["L1", "L2", "L3"]

    transport_candidate_heatmaps = {variable: np.zeros((3, 16), dtype=float) for variable in variables}
    transport_topk_heatmaps = {variable: np.zeros((3, 16), dtype=float) for variable in variables}
    das_canonical_heatmaps = {variable: np.zeros((3, 16), dtype=float) for variable in variables}
    das_rotated_heatmaps = {variable: np.zeros((3, 16), dtype=float) for variable in variables}

    cache_path = output_path.parent / "heq_das_projector_cache.json"
    repo_dir = Path(__file__).resolve().parent.parent
    das_projector_cache = _compute_das_projector_cache(repo_dir, strategy_payload, cache_path)

    ot_payload = strategy["methods"]["ot"]["raw_payload"]
    transport_matrix = np.asarray(ot_payload["transport"], dtype=float)
    target_vars = [str(variable) for variable in ot_payload["target_vars"]]
    site_labels = [str(label) for label in ot_payload["sites"]]
    for target_idx, variable in enumerate(target_vars):
        for site_idx, site_label in enumerate(site_labels):
            layer_idx = int(site_label[1])
            dim_idx = int(site_label.split("-d", 1)[1])
            if 0 <= layer_idx < 3 and 0 <= dim_idx < 16:
                transport_candidate_heatmaps[variable][layer_idx, dim_idx] = float(transport_matrix[target_idx, site_idx])
    for variable, layer_masks in ot_payload["layer_masks_by_variable"].items():
        for layer_name, mask_values in layer_masks.items():
            layer_idx = int(layer_name[1:])
            transport_topk_heatmaps[str(variable)][layer_idx, :] = np.asarray(mask_values, dtype=float)

    das_payload = strategy["methods"]["das"]["raw_payload"]
    for record in das_payload["results"]:
        variable = str(record["variable"])
        layer_index = int(record["layer"])
        das_canonical_heatmaps[variable][layer_index, :] = np.asarray(
            das_projector_cache[variable]["projector_diag"],
            dtype=float,
        )
        das_rotated_heatmaps[variable][layer_index, : int(record["subspace_dim"])] = 1.0

    def rescale_panel(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        vmax = float(values.max())
        vmin = float(values.min())
        if vmax <= 0.0:
            return values
        if vmax > vmin:
            return (values - vmin) / (vmax - vmin)
        return values / vmax

    cmap = plt.get_cmap("viridis")
    norm = _make_tail_detail_norm()
    title_pad_points = 2.85  # 1 mm at 72.27 points per inch.
    fig = plt.figure(figsize=(7.35, 3.72), constrained_layout=True)
    outer = fig.add_gridspec(
        2,
        5,
        width_ratios=[0.25, 0.28, 1.0, 1.0, 0.08],
        wspace=0.08,
        hspace=0.48,
    )
    image = None
    row_axes = {}

    for var_row, variable in enumerate(variables):
        variable_ax = fig.add_subplot(outer[var_row, 0])
        variable_ax.axis("off")

        method_grid = outer[var_row, 1].subgridspec(2, 1, hspace=0.0)
        heat_grid = outer[var_row, 2:4].subgridspec(2, 2, wspace=0.03, hspace=0.02)

        for row_idx, method in enumerate(methods):
            label_ax = fig.add_subplot(method_grid[row_idx, 0])
            label_ax.axis("off")

            first_ax = fig.add_subplot(heat_grid[row_idx, 0])
            second_ax = fig.add_subplot(heat_grid[row_idx, 1], sharey=first_ax)
            row_axes[(variable, method)] = (first_ax, second_ax)
            if method == "ot":
                first_values = transport_candidate_heatmaps[variable]
                second_values = transport_topk_heatmaps[variable]
                first_title = "pre-top-$K$"
                second_title = "top-$K$"
            else:
                first_values = das_canonical_heatmaps[variable]
                second_values = das_rotated_heatmaps[variable]
                first_title = "canonical"
                second_title = "rotated"

            first_values = rescale_panel(first_values)
            second_values = rescale_panel(second_values)
            box_aspect = first_values.shape[0] / first_values.shape[1]
            first_ax.set_box_aspect(box_aspect)
            second_ax.set_box_aspect(box_aspect)
            image = first_ax.imshow(first_values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
            second_ax.imshow(second_values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

            for heat_ax in (first_ax, second_ax):
                heat_ax.set_xticks(np.arange(-0.5, 16, 1), minor=True)
                heat_ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
                heat_ax.grid(which="minor", color="white", linewidth=0.32)
                heat_ax.tick_params(which="minor", bottom=False, left=False)
                for spine in heat_ax.spines.values():
                    spine.set_linewidth(0.8)
                    spine.set_edgecolor("#333333")

            first_ax.set_yticks([0, 1, 2], layer_labels, fontsize=8)
            second_ax.set_yticks([0, 1, 2], [])
            if variable == variables[-1] and row_idx == len(methods) - 1:
                first_ax.set_xticks([0, 5, 10, 15], ["0", "5", "10", "15"], fontsize=8)
                second_ax.set_xticks([0, 5, 10, 15], ["0", "5", "10", "15"], fontsize=8)
                first_ax.set_xlabel("Neuron index", fontsize=8)
            else:
                first_ax.set_xticks([0, 5, 10, 15], [])
                second_ax.set_xticks([0, 5, 10, 15], [])
                first_ax.tick_params(axis="x", bottom=False, labelbottom=False)
                second_ax.tick_params(axis="x", bottom=False, labelbottom=False)

            first_ax.set_title(first_title, fontsize=8, pad=title_pad_points, color="#444444")
            second_ax.set_title(second_title, fontsize=8, pad=title_pad_points, color="#444444")

    cax = fig.add_subplot(outer[:, 4])
    cax.axis("off")
    fig.canvas.draw()

    wx_das_shift = 0.035
    for heat_ax in row_axes[("WX", "das")]:
        position = heat_ax.get_position()
        heat_ax.set_position([position.x0, position.y0 + wx_das_shift, position.width, position.height])

    reference_columns = row_axes[("WX", "ot")]
    for variable in variables:
        for method in methods:
            for col_idx, heat_ax in enumerate(row_axes[(variable, method)]):
                ref_position = reference_columns[col_idx].get_position()
                position = heat_ax.get_position()
                heat_ax.set_position([ref_position.x0, position.y0, ref_position.width, position.height])

    first_heat_x0 = row_axes[(variables[0], methods[0])][0].get_position().x0
    method_x = first_heat_x0 - 0.035
    variable_x = method_x - 0.078
    for variable in variables:
        row_centers = []
        for method in methods:
            row_box = row_axes[(variable, method)][0].get_position()
            row_center = 0.5 * (row_box.y0 + row_box.y1)
            row_centers.append(row_center)
            fig.text(method_x, row_center, method_labels[method], ha="right", va="center", fontsize=10)
        fig.text(variable_x, 0.5 * sum(row_centers), rf"$z_{{{variable}}}$", ha="center", va="center", fontsize=11)

    cax_position = cax.get_position()
    colorbar_top = row_axes[(variables[0], methods[0])][0].get_position().y1
    colorbar_bottom = row_axes[(variables[-1], methods[-1])][0].get_position().y0
    cax.set_position([cax_position.x0 + 0.028, colorbar_bottom, cax_position.width, colorbar_top - colorbar_bottom])
    cax.set_axis_on()
    tick_values = [0.0, 0.05, 0.15, 0.5, 0.85, 0.95, 1.0]
    cbar = fig.colorbar(image, cax=cax, ticks=tick_values)
    cbar.ax.set_yticklabels([f"{tick:.2f}".rstrip("0").rstrip(".") for tick in tick_values])
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Relative site strength", rotation=90, fontsize=9)

    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    paper_dir = Path(__file__).resolve().parent
    repo_dir = paper_dir.parent
    results_dir = repo_dir / "results"
    plot_dir = paper_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    epsilon_payload = load_json(results_dir / "20260404_173020_equality_clean_epsilon_sweep" / "equality_clean_epsilon_sweep.json")
    strategy_payload = load_json(results_dir / "20260402_123611_equality_calibration_strategy_sweep" / "equality_calibration_strategy_sweep.json")

    plot_ot_epsilon_sweep(epsilon_payload, plot_dir / "heq_ot_epsilon_sweep.png")
    plot_handle_summary(strategy_payload, plot_dir / "heq_handle_summary.png")
    plot_handle_heatmaps_stacked(strategy_payload, plot_dir / "heq_handle_heatmaps_stacked.png")


if __name__ == "__main__":
    main()
