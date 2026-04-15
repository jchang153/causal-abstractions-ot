"""ARC task definitions, dataset loading, and factual filtering."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import random
from typing import Callable

from datasets import get_dataset_split_names, load_dataset
import torch
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from .runtime import resolve_device


DATASET_PATH = os.environ.get("ARC_DATASET_PATH", "mib-bench/arc")
DATASET_NAME = os.environ.get("ARC_DATASET_CONFIG")
_DATASET_CONFIG_UNSET = object()

CANONICAL_ANSWER_STRINGS = (" A", " B", " C", " D")
CANONICAL_ANSWER_LABELS = ("A", "B", "C", "D")
ALPHABET_LABELS = tuple(chr(codepoint) for codepoint in range(ord("A"), ord("Z") + 1))


class ARCCausalModel:
    def __init__(self) -> None:
        self.variables = (
            "question",
            "raw_input",
            "symbol0",
            "symbol1",
            "symbol2",
            "symbol3",
            "choice0",
            "choice1",
            "choice2",
            "choice3",
            "answer_key",
            "answer_pointer",
            "answer",
            "raw_output",
        )

    def run_forward(self, input_dict: dict[str, object]) -> dict[str, object]:
        output = dict(input_dict)
        symbols = [str(output[f"symbol{index}"]) for index in range(4)]
        answer_key = str(output.get("answer_key", symbols[0])).strip().upper()
        pointer = None
        for index, symbol in enumerate(symbols):
            if symbol.strip().upper() == answer_key:
                pointer = index
                break
        if pointer is None:
            raise ValueError(f"Could not resolve answer_pointer from answer_key={answer_key} and symbols={symbols}")
        answer = " " + symbols[pointer]
        output["answer_key"] = answer_key
        output["answer_pointer"] = int(pointer)
        output["answer"] = answer
        output["raw_output"] = answer
        return output


@dataclass(frozen=True)
class TokenPosition:
    resolver: Callable[[dict[str, object], object], list[int]]
    id: str

    def resolve(self, input_dict: dict[str, object], tokenizer) -> int:
        positions = self.resolver(input_dict, tokenizer)
        if not positions:
            raise ValueError(f"Token position {self.id} returned no indices")
        return int(positions[0])


@dataclass(frozen=True)
class ARCPairBank:
    split: str
    target_var: str
    dataset_names: tuple[str, ...]
    base_input_ids: torch.Tensor
    base_attention_mask: torch.Tensor
    source_input_ids: torch.Tensor
    source_attention_mask: torch.Tensor
    labels: torch.Tensor
    base_inputs: list[dict[str, object]]
    source_inputs: list[dict[str, object]]
    base_outputs: list[dict[str, object]]
    source_outputs: list[dict[str, object]]
    base_position_by_id: dict[str, torch.Tensor]
    source_position_by_id: dict[str, torch.Tensor]
    symbol_token_ids: torch.Tensor
    symbol_variant_token_ids: torch.Tensor
    source_symbol_token_ids: torch.Tensor
    source_symbol_variant_token_ids: torch.Tensor
    alphabet_token_ids: torch.Tensor
    alphabet_variant_token_ids: torch.Tensor
    canonical_answer_token_ids: torch.Tensor
    answer_token_ids: torch.Tensor
    base_answer_token_ids: torch.Tensor
    changed_mask: torch.Tensor
    expected_answer_texts: list[str]

    @property
    def size(self) -> int:
        return int(self.labels.shape[0])

    def metadata(self) -> dict[str, object]:
        return {
            "split": self.split,
            "target_var": self.target_var,
            "size": self.size,
            "dataset_names": list(self.dataset_names),
            "changed_count": int(self.changed_mask.sum().item()),
            "changed_rate": float(self.changed_mask.float().mean().item()) if self.size else 0.0,
        }


class ARCPairDataset(torch.utils.data.Dataset):
    def __init__(self, bank: ARCPairBank):
        self.bank = bank

    def __len__(self) -> int:
        return self.bank.size

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "base_input_ids": self.bank.base_input_ids[index],
            "base_attention_mask": self.bank.base_attention_mask[index],
            "source_input_ids": self.bank.source_input_ids[index],
            "source_attention_mask": self.bank.source_attention_mask[index],
            "labels": self.bank.labels[index],
            "symbol_token_ids": self.bank.symbol_token_ids[index],
            "symbol_variant_token_ids": self.bank.symbol_variant_token_ids[index],
            "source_symbol_token_ids": self.bank.source_symbol_token_ids[index],
            "source_symbol_variant_token_ids": self.bank.source_symbol_variant_token_ids[index],
            "answer_token_id": self.bank.answer_token_ids[index],
            "base_answer_token_id": self.bank.base_answer_token_ids[index],
            "base_positions": {key: value[index] for key, value in self.bank.base_position_by_id.items()},
            "source_positions": {key: value[index] for key, value in self.bank.source_position_by_id.items()},
            "expected_answer_text": self.bank.expected_answer_texts[index],
        }


def _normalize_label(value: object) -> str:
    text = str(value).strip().upper()
    return text[0] if text else "A"


def _canonical_option_symbols() -> list[str]:
    return list(CANONICAL_ANSWER_LABELS)


def _extract_question_stem(row: dict[str, object]) -> str:
    question = row.get("question")
    if isinstance(question, dict) and "stem" in question:
        return str(question["stem"]).strip()
    if isinstance(row.get("question"), str):
        return str(row["question"]).strip()
    if "prompt" in row:
        return str(row["prompt"]).strip()
    return ""


def _extract_choices(row: dict[str, object]) -> tuple[list[str], list[str]]:
    question = row.get("question")
    if isinstance(question, dict) and isinstance(question.get("choices"), list):
        labels = [_normalize_label(choice.get("label", CANONICAL_ANSWER_LABELS[index])) for index, choice in enumerate(question["choices"])]
        texts = [str(choice.get("text", "")).strip() for choice in question["choices"]]
        return labels[:4], texts[:4]
    choices = row.get("choices")
    if isinstance(choices, dict) and "label" in choices and "text" in choices:
        labels = [_normalize_label(item) for item in list(choices["label"])[:4]]
        texts = [str(item).strip() for item in list(choices["text"])[:4]]
        return labels, texts
    raise ValueError("Could not parse ARC choices from row")


def _format_prompt(stem: str, labels: list[str], texts: list[str]) -> str:
    lines = [f"Question: {stem}", "Choices:"]
    lines.extend([f"{label}. {text}" for label, text in zip(labels, texts)])
    lines.append("Answer:")
    return "\n".join(lines)


def parse_arc_example(row: dict[str, object]) -> dict[str, object]:
    if "raw_input" in row and all(f"symbol{index}" in row for index in range(4)):
        return dict(row)

    stem = _extract_question_stem(row)
    labels, texts = _extract_choices(row)
    if len(labels) < 4 or len(texts) < 4:
        raise ValueError(f"Expected >=4 choices, got labels={len(labels)} texts={len(texts)}")
    answer_key = _normalize_label(row.get("answerKey", row.get("answer_key", labels[0])))
    label_to_index = {label: index for index, label in enumerate(labels[:4])}
    answer_index = label_to_index.get(answer_key, 0)
    canonical_symbols = _canonical_option_symbols()

    result: dict[str, object] = {
        "question": stem,
        "raw_input": _format_prompt(stem, labels, texts),
        "answer_key": canonical_symbols[answer_index],
    }
    for index in range(4):
        result[f"symbol{index}"] = canonical_symbols[index]
        result[f"choice{index}"] = texts[index]
    return result


def get_token_positions(tokenizer, causal_model: ARCCausalModel) -> list[TokenPosition]:
    del causal_model

    def last_token(input_dict: dict[str, object], current_tokenizer) -> list[int]:
        prompt = str(input_dict["raw_input"])
        tokenized = current_tokenizer(prompt, add_special_tokens=True, return_attention_mask=False)["input_ids"]
        return [len(tokenized) - 1]

    return [TokenPosition(last_token, "last_token")]


def _load_counterfactual_rows(
    *,
    split: str,
    size: int | None = None,
    hf_token: str | None = None,
    dataset_path: str | None = None,
    dataset_name: str | None | object = _DATASET_CONFIG_UNSET,
) -> dict[str, list[dict[str, object]]]:
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    resolved_dataset_path = dataset_path or DATASET_PATH
    resolved_dataset_name = DATASET_NAME if dataset_name is _DATASET_CONFIG_UNSET else dataset_name
    dataset_path_obj = Path(resolved_dataset_path)
    if dataset_path_obj.exists():
        split_file = dataset_path_obj / f"{split}.jsonl"
        if not split_file.exists():
            raise FileNotFoundError(f"Expected local ARC dataset file at {split_file}")
        dataset = load_dataset("json", data_files={split: str(split_file)}, split=split)
    else:
        if resolved_dataset_name:
            dataset = load_dataset(resolved_dataset_path, resolved_dataset_name, split=split, token=token)
        else:
            dataset = load_dataset(resolved_dataset_path, split=split, token=token)
    if size is not None:
        dataset = dataset.select(range(min(int(size), len(dataset))))

    sample = dataset[0]
    if "input" in sample and "counterfactual_inputs" in sample:
        return {f"paired_{split}": [dict(row) for row in dataset]}

    counterfactual_names = [key for key in sample.keys() if key.endswith("_counterfactual")]
    if not counterfactual_names:
        raise ValueError(
            "ARC loader expected MIB-style counterfactual columns '*_counterfactual' or prepaired rows "
            "with keys ['input', 'counterfactual_inputs']."
        )

    datasets: dict[str, list[dict[str, object]]] = {}
    for counterfactual_name in counterfactual_names:
        dataset_name = counterfactual_name.replace("_counterfactual", f"_{split}")
        rows: list[dict[str, object]] = []
        for row in dataset:
            base_input = parse_arc_example(row)
            source_input = parse_arc_example(row[counterfactual_name])
            rows.append({"input": base_input, "counterfactual_inputs": [source_input]})
        datasets[dataset_name] = rows
    return datasets


def load_public_arc_datasets(
    *,
    size: int | None = None,
    hf_token: str | None = None,
    dataset_path: str | None = None,
    dataset_name: str | None | object = _DATASET_CONFIG_UNSET,
) -> dict[str, list[dict[str, object]]]:
    datasets: dict[str, list[dict[str, object]]] = {}
    resolved_dataset_path = dataset_path or DATASET_PATH
    resolved_dataset_name = DATASET_NAME if dataset_name is _DATASET_CONFIG_UNSET else dataset_name
    dataset_path_obj = Path(resolved_dataset_path)
    if dataset_path_obj.exists():
        candidate_splits = tuple(split_file.stem for split_file in sorted(dataset_path_obj.glob("*.jsonl")) if split_file.stem)
        if not candidate_splits:
            raise FileNotFoundError(f"No .jsonl splits found under local dataset path {dataset_path_obj}")
    else:
        if resolved_dataset_name:
            candidate_splits = tuple(get_dataset_split_names(resolved_dataset_path, resolved_dataset_name, token=hf_token))
        else:
            candidate_splits = tuple(get_dataset_split_names(resolved_dataset_path, token=hf_token))
    for split in candidate_splits:
        datasets.update(
            _load_counterfactual_rows(
                split=split,
                size=size,
                hf_token=hf_token,
                dataset_path=resolved_dataset_path,
                dataset_name=resolved_dataset_name,
            )
        )
    return datasets


def _infer_next_token_ids(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits
    last_indices = torch.full((logits.shape[0],), logits.shape[1] - 1, dtype=torch.long, device=logits.device)
    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch_indices, last_indices].argmax(dim=-1)


def normalize_answer_text(text: str) -> str:
    return str(text).strip()


def filter_correct_examples(
    *,
    model,
    tokenizer,
    causal_model: ARCCausalModel,
    datasets_by_name: dict[str, list[dict[str, object]]],
    batch_size: int,
    device: torch.device,
) -> dict[str, list[dict[str, object]]]:
    filtered: dict[str, list[dict[str, object]]] = {}
    for dataset_name, rows in datasets_by_name.items():
        print(f"[filter] dataset={dataset_name} total_rows={len(rows)}")
        prompts = [str(row["input"]["raw_input"]) for row in rows]
        expected_answers = [normalize_answer_text(str(causal_model.run_forward(row["input"])["answer"])) for row in rows]
        expected_answer_variant_ids = [_encode_symbol_token_variants(expected, tokenizer) for expected in expected_answers]
        keep_mask: list[bool] = []
        batch_starts = range(0, len(rows), batch_size)
        batch_iterator = batch_starts
        if tqdm is not None:
            batch_iterator = tqdm(
                batch_starts,
                desc=f"Filtering {dataset_name}",
                leave=False,
                total=(len(rows) + batch_size - 1) // batch_size,
            )
        for start in batch_iterator:
            end = min(start + batch_size, len(rows))
            batch_prompts = prompts[start:end]
            encoded = tokenizer(batch_prompts, padding=True, return_tensors="pt", add_special_tokens=True)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            predicted_ids = _infer_next_token_ids(model, input_ids, attention_mask)
            for predicted_id, expected, expected_variants in zip(
                predicted_ids.detach().cpu().tolist(),
                expected_answers[start:end],
                expected_answer_variant_ids[start:end],
            ):
                decoded = normalize_answer_text(tokenizer.decode([int(predicted_id)]))
                keep_mask.append(int(predicted_id) in expected_variants or expected == decoded)
        filtered[dataset_name] = [row for row, keep in zip(rows, keep_mask) if keep]
        print(f"[filter] dataset={dataset_name} kept={len(filtered[dataset_name])}/{len(rows)}")
    return filtered


def _validate_answer_tokenization(tokenizer) -> torch.Tensor:
    token_ids = []
    for token_text in CANONICAL_ANSWER_STRINGS:
        ids = tokenizer.encode(token_text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Expected {token_text!r} to map to a single token, but got token ids {ids}. "
                "ARC v1 requires single-token answer letters."
            )
        token_ids.append(int(ids[0]))
    return torch.tensor(token_ids, dtype=torch.long)


def _encode_symbol_token(symbol: str, tokenizer) -> int:
    ids = tokenizer.encode(" " + str(symbol), add_special_tokens=False)
    if len(ids) != 1:
        ids = tokenizer.encode(str(symbol), add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Expected symbol {symbol!r} to map to one token, but got ids {ids}")
    return int(ids[0])


def _encode_symbol_token_variants(symbol: str, tokenizer) -> tuple[int, int]:
    symbol = normalize_answer_text(symbol)
    variant_ids = []
    for candidate in (" " + symbol, symbol):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            variant_ids.append(int(ids[0]))
    if not variant_ids:
        raise ValueError(f"Expected symbol {symbol!r} to have at least one single-token encoding")
    if len(variant_ids) == 1:
        variant_ids.append(variant_ids[0])
    return (variant_ids[0], variant_ids[1])


def _alphabet_index(symbol: str) -> int:
    normalized = normalize_answer_text(symbol)
    if len(normalized) != 1 or normalized not in ALPHABET_LABELS:
        raise ValueError(f"Expected uppercase alphabet symbol, got {symbol!r}")
    return ALPHABET_LABELS.index(normalized)


def _compute_row_change_masks(
    rows: list[dict[str, object]],
    causal_model: ARCCausalModel,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, list[bool]]]:
    base_inputs = [row["input"] for row in rows]
    source_inputs = [row["counterfactual_inputs"][0] for row in rows]
    base_outputs = [causal_model.run_forward(base_input) for base_input in base_inputs]
    source_outputs = [causal_model.run_forward(source_input) for source_input in source_inputs]
    changed_masks = {
        "answer_pointer": [
            int(base_output["answer_pointer"]) != int(source_output["answer_pointer"])
            for base_output, source_output in zip(base_outputs, source_outputs)
        ],
        "answer": [
            str(base_output["answer"]) != str(source_output["answer"])
            for base_output, source_output in zip(base_outputs, source_outputs)
        ],
    }
    return base_outputs, source_outputs, changed_masks


def build_pair_banks(
    *,
    tokenizer,
    causal_model: ARCCausalModel,
    token_positions: list[TokenPosition],
    datasets_by_name: dict[str, list[dict[str, object]]],
    counterfactual_names: tuple[str, ...],
    target_vars: tuple[str, ...],
    split_seed: int = 0,
    train_pool_size: int | None = None,
    calibration_pool_size: int | None = None,
    test_pool_size: int | None = None,
    pooled_total_examples: int | None = None,
) -> tuple[dict[str, dict[str, ARCPairBank]], dict[str, object]]:
    canonical_answer_token_ids = _validate_answer_tokenization(tokenizer)

    def make_bank(output_split: str, split_dataset_names: list[str], combined_rows: list[dict[str, object]]) -> dict[str, ARCPairBank]:
        base_inputs = [row["input"] for row in combined_rows]
        source_inputs = [row["counterfactual_inputs"][0] for row in combined_rows]
        base_outputs = [causal_model.run_forward(base_input) for base_input in base_inputs]
        source_outputs = [causal_model.run_forward(source_input) for source_input in source_inputs]

        base_prompts = [str(base_input["raw_input"]) for base_input in base_inputs]
        source_prompts = [str(source_input["raw_input"]) for source_input in source_inputs]
        base_encoded = tokenizer(base_prompts, padding=True, return_tensors="pt", add_special_tokens=True)
        source_encoded = tokenizer(source_prompts, padding=True, return_tensors="pt", add_special_tokens=True)

        base_position_by_id = {
            token_position.id: torch.tensor([token_position.resolve(base_input, tokenizer) for base_input in base_inputs], dtype=torch.long)
            for token_position in token_positions
        }
        source_position_by_id = {
            token_position.id: torch.tensor([token_position.resolve(source_input, tokenizer) for source_input in source_inputs], dtype=torch.long)
            for token_position in token_positions
        }
        symbol_token_ids = torch.tensor(
            [[_encode_symbol_token(str(base_input[f"symbol{index}"]), tokenizer) for index in range(4)] for base_input in base_inputs],
            dtype=torch.long,
        )
        symbol_variant_token_ids = torch.tensor(
            [[_encode_symbol_token_variants(str(base_input[f"symbol{index}"]), tokenizer) for index in range(4)] for base_input in base_inputs],
            dtype=torch.long,
        )
        source_symbol_token_ids = torch.tensor(
            [[_encode_symbol_token(str(source_input[f"symbol{index}"]), tokenizer) for index in range(4)] for source_input in source_inputs],
            dtype=torch.long,
        )
        source_symbol_variant_token_ids = torch.tensor(
            [[_encode_symbol_token_variants(str(source_input[f"symbol{index}"]), tokenizer) for index in range(4)] for source_input in source_inputs],
            dtype=torch.long,
        )
        alphabet_variant_token_ids = torch.tensor(
            [[_encode_symbol_token_variants(letter, tokenizer) for letter in ALPHABET_LABELS] for _ in base_inputs],
            dtype=torch.long,
        )
        alphabet_token_ids = alphabet_variant_token_ids[:, :, 0]
        answer_token_ids = torch.tensor(
            [_encode_symbol_token(str(source_output["answer"]).strip(), tokenizer) for source_output in source_outputs],
            dtype=torch.long,
        )
        base_answer_token_ids = torch.tensor(
            [_encode_symbol_token(str(base_output["answer"]).strip(), tokenizer) for base_output in base_outputs],
            dtype=torch.long,
        )
        answer_label_indices = torch.tensor(
            [_alphabet_index(str(source_output["answer"])) for source_output in source_outputs],
            dtype=torch.long,
        )
        changed_pointer = torch.tensor(
            [int(base_output["answer_pointer"]) != int(source_output["answer_pointer"]) for base_output, source_output in zip(base_outputs, source_outputs)],
            dtype=torch.bool,
        )
        changed_answer = torch.tensor(
            [str(base_output["answer"]) != str(source_output["answer"]) for base_output, source_output in zip(base_outputs, source_outputs)],
            dtype=torch.bool,
        )

        banks: dict[str, ARCPairBank] = {}
        for target_var in target_vars:
            if target_var == "answer_pointer":
                labels = answer_label_indices
                changed_mask = changed_pointer
            elif target_var == "answer":
                labels = answer_label_indices
                changed_mask = changed_answer
            else:
                raise ValueError(f"Unsupported ARC target variable {target_var}")
            banks[target_var] = ARCPairBank(
                split=output_split,
                target_var=target_var,
                dataset_names=tuple(split_dataset_names),
                base_input_ids=base_encoded["input_ids"].to(torch.long),
                base_attention_mask=base_encoded["attention_mask"].to(torch.long),
                source_input_ids=source_encoded["input_ids"].to(torch.long),
                source_attention_mask=source_encoded["attention_mask"].to(torch.long),
                labels=labels,
                base_inputs=base_inputs,
                source_inputs=source_inputs,
                base_outputs=base_outputs,
                source_outputs=source_outputs,
                base_position_by_id=base_position_by_id,
                source_position_by_id=source_position_by_id,
                symbol_token_ids=symbol_token_ids,
                symbol_variant_token_ids=symbol_variant_token_ids,
                source_symbol_token_ids=source_symbol_token_ids,
                source_symbol_variant_token_ids=source_symbol_variant_token_ids,
                alphabet_token_ids=alphabet_token_ids,
                alphabet_variant_token_ids=alphabet_variant_token_ids,
                canonical_answer_token_ids=canonical_answer_token_ids,
                answer_token_ids=answer_token_ids,
                base_answer_token_ids=base_answer_token_ids,
                changed_mask=changed_mask,
                expected_answer_texts=[normalize_answer_text(str(source_output["answer"])) for source_output in source_outputs],
            )
        return banks

    banks_by_split: dict[str, dict[str, ARCPairBank]] = {"train": {}, "calibration": {}, "test": {}}
    pooled_dataset_names = []
    pooled_rows: list[dict[str, object]] = []
    for dataset_name in sorted(datasets_by_name):
        counterfactual_name, _, _split_name = dataset_name.rpartition("_")
        if counterfactual_names and counterfactual_name not in counterfactual_names:
            continue
        pooled_dataset_names.append(dataset_name)
        pooled_rows.extend(datasets_by_name[dataset_name])

    if not pooled_rows:
        raise ValueError("No ARC rows found for pooled bank construction")

    rng = random.Random(int(split_seed))
    shuffled_rows = list(pooled_rows)
    rng.shuffle(shuffled_rows)
    if pooled_total_examples is not None:
        shuffled_rows = shuffled_rows[: min(int(pooled_total_examples), len(shuffled_rows))]
    total = len(shuffled_rows)

    resolved_train_pool_size = total if train_pool_size is None else int(train_pool_size)
    resolved_calibration_pool_size = 0 if calibration_pool_size is None else int(calibration_pool_size)
    resolved_test_pool_size = 0 if test_pool_size is None else int(test_pool_size)
    if resolved_train_pool_size < 0 or resolved_calibration_pool_size < 0 or resolved_test_pool_size < 0:
        raise ValueError("train_pool_size, calibration_pool_size, and test_pool_size must be non-negative")
    if resolved_train_pool_size > total:
        raise ValueError(
            f"Requested train_pool_size={resolved_train_pool_size}, but only {total} filtered ARC rows are available"
        )

    train_rows = shuffled_rows[:resolved_train_pool_size]
    holdout_candidate_rows = shuffled_rows[resolved_train_pool_size:]
    if not train_rows:
        raise ValueError("No ARC rows found for pooled train split")

    train_rng = random.Random(f"{int(split_seed)}:train:shared")
    shared_train_rows = list(train_rows)
    train_rng.shuffle(shared_train_rows)
    train_banks = make_bank("train", pooled_dataset_names, shared_train_rows)
    for target_var in target_vars:
        banks_by_split["train"][target_var] = train_banks[target_var]

    _base_outputs, _source_outputs, holdout_changed_masks = _compute_row_change_masks(holdout_candidate_rows, causal_model)
    for target_var in target_vars:
        changed_mask = holdout_changed_masks[target_var]
        positive_rows = [row for row, changed in zip(holdout_candidate_rows, changed_mask) if changed]
        local_rng = random.Random(f"{int(split_seed)}:holdout:{target_var}")
        local_rng.shuffle(positive_rows)
        required = resolved_calibration_pool_size + resolved_test_pool_size
        if len(positive_rows) < required:
            raise ValueError(
                f"Requested calibration_pool_size={resolved_calibration_pool_size} and "
                f"test_pool_size={resolved_test_pool_size} for target_var={target_var}, "
                f"but only {len(positive_rows)} sensitive rows are available after train allocation"
            )
        calibration_rows = positive_rows[:resolved_calibration_pool_size]
        test_rows = positive_rows[resolved_calibration_pool_size:required]
        if resolved_calibration_pool_size > 0:
            banks_by_split["calibration"][target_var] = make_bank("calibration", pooled_dataset_names, calibration_rows)[target_var]
        if resolved_test_pool_size > 0:
            banks_by_split["test"][target_var] = make_bank("test", pooled_dataset_names, test_rows)[target_var]

    if resolved_calibration_pool_size == 0:
        banks_by_split["calibration"] = {}
    if resolved_test_pool_size == 0:
        banks_by_split["test"] = {}

    metadata = {
        split: {target_var: bank.metadata() for target_var, bank in banks.items()}
        for split, banks in banks_by_split.items()
    }
    return banks_by_split, metadata


def load_filtered_arc_pipeline(
    *,
    model_name: str,
    device: str | None = None,
    batch_size: int = 16,
    dataset_size: int | None = None,
    hf_token: str | None = None,
    dataset_path: str | None = None,
    dataset_name: str | None | object = _DATASET_CONFIG_UNSET,
) -> tuple[object, object, ARCCausalModel, list[TokenPosition], dict[str, list[dict[str, object]]]]:
    import transformers

    torch_device = resolve_device(device)
    print(f"[load] device={torch_device} model={model_name}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.float16 if torch_device.type in {"cuda", "mps"} else torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        token=hf_token,
        attn_implementation="eager",
    )
    model.to(torch_device)
    model.eval()

    causal_model = ARCCausalModel()
    token_positions = get_token_positions(tokenizer, causal_model)

    resolved_dataset_path = dataset_path or DATASET_PATH
    resolved_dataset_name = DATASET_NAME if dataset_name is _DATASET_CONFIG_UNSET else dataset_name
    print(f"[load] loading ARC datasets path={resolved_dataset_path} config={resolved_dataset_name!r} size_cap={dataset_size}")
    public_datasets = load_public_arc_datasets(
        size=dataset_size,
        hf_token=hf_token,
        dataset_path=resolved_dataset_path,
        dataset_name=resolved_dataset_name,
    )
    print(f"[load] loaded datasets={sorted(public_datasets.keys())}")

    filtered_datasets = filter_correct_examples(
        model=model,
        tokenizer=tokenizer,
        causal_model=causal_model,
        datasets_by_name=public_datasets,
        batch_size=batch_size,
        device=torch_device,
    )
    return model, tokenizer, causal_model, token_positions, filtered_datasets
