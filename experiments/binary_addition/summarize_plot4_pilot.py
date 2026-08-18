from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the four-method binary-addition PLOT pilot.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"missing pilot result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _method_row(name: str, payload: dict[str, object]) -> dict[str, object]:
    accuracy = float(payload["mean_combined"])
    runtime = float(payload["runtime_seconds"])
    if not math.isfinite(accuracy) or not math.isfinite(runtime):
        raise ValueError(f"non-finite result for {name}: accuracy={accuracy}, runtime={runtime}")
    return {
        "method": str(name),
        "average_accuracy": accuracy,
        "serial_runtime_seconds": runtime,
    }


def summarize(run_dir: Path, *, hidden_size: int, seed: int) -> dict[str, object]:
    single_root = run_dir / "single_stage" / f"h{int(hidden_size)}" / f"seed_{int(seed)}"
    progressive_path = (
        run_dir
        / "progressive_ot"
        / f"h{int(hidden_size)}"
        / f"seed_{int(seed)}"
        / "progressive_seed_summary.json"
    )
    single_native = _load(single_root / "single_stage_plot_seed_summary.json")
    single_pca = _load(single_root / "single_stage_plot_pca_seed_summary.json")
    progressive = _load(progressive_path)

    provenance_records = [
        single_native.get("protocol", {}),
        single_pca.get("protocol", {}),
        progressive.get("protocol", {}),
    ]
    required_provenance = ("checkpoint_sha256", "split_manifest_sha256", "rows")
    for key in required_provenance:
        values = [json.dumps(record.get(key), sort_keys=True) for record in provenance_records]
        if any(value == "null" for value in values) or len(set(values)) != 1:
            raise ValueError(f"pilot methods do not share one {key}: {values}")

    progressive_methods = progressive["methods"]
    methods = [
        _method_row("PLOT (single-stage)", single_native["method"]),
        _method_row("PLOT-native (two-stage)", progressive_methods["plot_in_timestep"]),
        _method_row("PLOT-PCA (single-stage)", single_pca["method"]),
        _method_row("PLOT-PCA (two-stage)", progressive_methods["plot_pca_in_timestep"]),
    ]
    result = {
        "config": {
            "run_dir": str(run_dir.resolve()),
            "hidden_size": int(hidden_size),
            "seed": int(seed),
        },
        "protocol": provenance_records[0],
        "methods": methods,
    }
    output_path = run_dir / "plot4_summary.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "| Method | Average accuracy | Serial runtime |",
        "|---|---:|---:|",
    ]
    for method in methods:
        lines.append(
            f"| {method['method']} | {method['average_accuracy']:.4f} | "
            f"{method['serial_runtime_seconds']:.2f}s |"
        )
    markdown_path = run_dir / "plot4_summary.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {**result, "summary": str(output_path), "table": str(markdown_path)}


def main() -> None:
    args = parse_args()
    result = summarize(
        Path(args.run_dir).resolve(),
        hidden_size=int(args.hidden_size),
        seed=int(args.seed),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
