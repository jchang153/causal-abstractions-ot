from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .scm import BinaryAdditionExample


@dataclass(frozen=True)
class TrainConfig:
    width: int = 4
    hidden_size: int = 16
    input_size: int = 2
    batch_size: int = 64
    epochs: int = 250
    learning_rate: float = 1e-2
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "cpu"
    early_stop_exact: float = 1.0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class GRUAdder(nn.Module):
    def __init__(self, width: int = 4, input_size: int = 2, hidden_size: int = 16) -> None:
        super().__init__()
        self.width = int(width)
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.cell = nn.GRUCell(self.input_size, self.hidden_size)
        self.sum_head = nn.Linear(self.hidden_size, 1)
        self.final_carry_head = nn.Linear(self.hidden_size, 1)

    def forward(self, x_bits: torch.Tensor) -> dict[str, torch.Tensor]:
        if x_bits.ndim != 3 or x_bits.size(1) != self.width or x_bits.size(2) != self.input_size:
            raise ValueError(
                f"x_bits must have shape [batch, {self.width}, {self.input_size}], got {tuple(x_bits.shape)}"
            )

        batch = x_bits.size(0)
        h = torch.zeros(batch, self.hidden_size, device=x_bits.device, dtype=x_bits.dtype)
        hidden_states = []
        sum_logits = []
        for step in range(self.width):
            h = self.cell(x_bits[:, step, :], h)
            hidden_states.append(h)
            sum_logits.append(self.sum_head(h))
        hidden_table = torch.stack(hidden_states, dim=1)
        sum_logits = torch.cat(sum_logits, dim=1)
        carry_logit = self.final_carry_head(hidden_states[-1])
        output_logits = torch.cat([sum_logits, carry_logit], dim=1)
        return {
            "hidden_states": hidden_table,
            "sum_logits": sum_logits,
            "carry_logit": carry_logit,
            "output_logits": output_logits,
        }


def examples_to_tensors(examples: Sequence[BinaryAdditionExample]) -> tuple[torch.Tensor, torch.Tensor]:
    x_rows = []
    y_rows = []
    for ex in examples:
        x_rows.append([[float(ex.a_bits_lsb[t]), float(ex.b_bits_lsb[t])] for t in range(ex.width)])
        y_rows.append([float(bit) for bit in ex.output_bits_lsb])
    x = torch.tensor(x_rows, dtype=torch.float32)
    y = torch.tensor(y_rows, dtype=torch.float32)
    return x, y


def predict_bits(model: GRUAdder, examples: Sequence[BinaryAdditionExample], device: torch.device) -> torch.Tensor:
    model.eval()
    x, _ = examples_to_tensors(examples)
    with torch.no_grad():
        logits = model(x.to(device))["output_logits"]
    return (torch.sigmoid(logits) >= 0.5).to(torch.int64).cpu()


def exact_accuracy(model: GRUAdder, examples: Sequence[BinaryAdditionExample], device: torch.device) -> float:
    if not examples:
        return 0.0
    preds = predict_bits(model, examples, device=device)
    _, y = examples_to_tensors(examples)
    correct = (preds == y.to(torch.int64)).all(dim=1).float().mean().item()
    return float(correct)


def bit_accuracy(model: GRUAdder, examples: Sequence[BinaryAdditionExample], device: torch.device) -> float:
    if not examples:
        return 0.0
    preds = predict_bits(model, examples, device=device)
    _, y = examples_to_tensors(examples)
    correct = (preds == y.to(torch.int64)).float().mean().item()
    return float(correct)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def train_backbone(
    config: TrainConfig,
    train_examples: Sequence[BinaryAdditionExample],
    eval_examples: Sequence[BinaryAdditionExample] | None = None,
) -> tuple[GRUAdder, dict[str, object]]:
    _seed_everything(config.seed)
    device = torch.device("cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu")

    if eval_examples is None:
        eval_examples = train_examples

    x_train, y_train = examples_to_tensors(train_examples)
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = GRUAdder(width=config.width, input_size=config.input_size, hidden_size=config.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_items = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)["output_logits"]
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            batch_size = xb.size(0)
            epoch_loss += float(loss.item()) * batch_size
            n_items += batch_size

        train_exact = exact_accuracy(model, train_examples, device=device)
        eval_exact = exact_accuracy(model, eval_examples, device=device)
        train_bit = bit_accuracy(model, train_examples, device=device)
        eval_bit = bit_accuracy(model, eval_examples, device=device)
        rec = {
            "epoch": float(epoch),
            "loss": epoch_loss / max(1, n_items),
            "train_exact": train_exact,
            "eval_exact": eval_exact,
            "train_bit": train_bit,
            "eval_bit": eval_bit,
        }
        history.append(rec)
        if eval_exact >= config.early_stop_exact:
            break

    summary = {
        "config": config.as_dict(),
        "device": str(device),
        "epochs_completed": len(history),
        "final": history[-1] if history else {},
        "history": history,
    }
    return model, summary
