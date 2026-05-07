from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from experiments.binary_addition.data import enumerate_all_examples, stratified_base_split
from experiments.binary_addition.model import TrainConfig, exact_accuracy, train_backbone


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a factual 4-bit GRU binary addition backbone.")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--hidden-size", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--learning-rate", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--fit-bases", type=int, default=128)
    ap.add_argument("--calib-bases", type=int, default=64)
    ap.add_argument("--test-bases", type=int, default=64)
    ap.add_argument("--train-on", type=str, default="all", choices=["all", "fit_only"])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = enumerate_all_examples(width=args.width)
    split = stratified_base_split(
        examples,
        fit_count=int(args.fit_bases),
        calib_count=int(args.calib_bases),
        test_count=int(args.test_bases),
        seed=int(args.seed),
    )

    train_examples = examples if args.train_on == "all" else split.fit
    eval_examples = examples
    cfg = TrainConfig(
        width=int(args.width),
        hidden_size=int(args.hidden_size),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        device=str(args.device),
    )
    model, train_summary = train_backbone(cfg, train_examples=train_examples, eval_examples=eval_examples)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    result = {
        "config": vars(args),
        "split_sizes": {"fit": len(split.fit), "calib": len(split.calib), "test": len(split.test)},
        "factual_exact": {
            "all": exact_accuracy(model, examples, device=device),
            "fit": exact_accuracy(model, split.fit, device=device),
            "calib": exact_accuracy(model, split.calib, device=device),
            "test": exact_accuracy(model, split.test, device=device),
        },
        "training": train_summary,
    }

    checkpoint_path = out_dir / "gru_adder.pt"
    summary_path = out_dir / "train_summary.json"
    torch.save({"model_state_dict": model.state_dict(), "train_config": cfg.as_dict()}, checkpoint_path)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps({"checkpoint": str(checkpoint_path), "summary": str(summary_path), "factual_exact_all": result["factual_exact"]["all"]}, indent=2))


if __name__ == "__main__":
    main()
