from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.ioi.causal import LinearCausalModel, fit_linear_causal_model
from experiments.ioi.data import (
    FAMILY_SPECS,
    IOIExample,
    deterministic_public_split,
    family_signals,
    mib_name_token_id,
    row_to_examples,
)
from experiments.ioi.interventions import (
    DASTrainConfig,
    HeadRotations,
    evaluate_mse,
    freeze_model,
    train_joint_das,
    all_gpt2_heads,
)
from experiments.ioi.plot import (
    SignatureBank,
    bruteforce_coupling_from_cost,
    build_cost_matrix,
    calibrate_uot_grid,
    choose_k,
    sinkhorn_one_sided_uot,
    top_k_heads,
)


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"Alice": [11], "Bob": [12], "Carol": [13]}[text]


class PipelineTokenizer(TinyTokenizer):
    pad_token_id = 0

    def __call__(self, prompts, **_kwargs):
        ids = [[1, 1] if prompt.startswith("base") else [2, 2] for prompt in prompts]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones((len(ids), 2), dtype=torch.long),
        }


class TinyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c_proj = torch.nn.Linear(4, 4, bias=False)
        self.c_proj.weight.data.copy_(torch.eye(4))


class TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = TinyAttention()


class TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(n_embd=4, n_head=1)
        self.transformer = SimpleNamespace(h=torch.nn.ModuleList([TinyBlock()]))
        self.embedding = torch.nn.Embedding(3, 4)
        self.output = torch.nn.Linear(4, 20, bias=False)
        torch.manual_seed(4)
        torch.nn.init.normal_(self.embedding.weight)
        torch.nn.init.normal_(self.output.weight)

    def forward(self, input_ids, attention_mask, position_ids, use_cache=False):
        del attention_mask, position_ids, use_cache
        hidden = self.embedding(input_ids)
        hidden = self.transformer.h[0].attn.c_proj(hidden)
        return SimpleNamespace(logits=self.output(hidden))


def _released_row():
    row = {"prompt": "base", "choices": ["Alice", "Bob"], "answerKey": 0}
    for family, (column, _position, _token) in FAMILY_SPECS.items():
        row[column] = {
            "prompt": family,
            "choices": ["Bob", "Alice"],
            "answerKey": 0,
        }
    return row


def test_counterfactual_signals_and_row_expansion():
    assert family_signals("s1_io_flip") == (-1, 1)
    assert family_signals("s2_io_flip") == (-1, -1)
    assert family_signals("s1_ioi_flip_s2_ioi_flip") == (1, -1)
    examples = row_to_examples(_released_row(), row_id=7, split="fit")
    assert set(examples) == set(FAMILY_SPECS)
    assert all(example.base_io == "Alice" for example in examples.values())
    assert all(example.source_io == "Bob" for example in examples.values())


def test_deterministic_split_is_disjoint_and_pre_filter():
    calibration, test = deterministic_public_split(10_000, calibration_rows=2_000, seed=0)
    assert len(calibration) == 2_000
    assert len(test) == 8_000
    assert not set(calibration).intersection(test)
    assert sorted(calibration + test) == list(range(10_000))
    assert (calibration, test) == deterministic_public_split(
        10_000, calibration_rows=2_000, seed=0
    )


def test_mib_operational_token_id_requires_one_token():
    tokenizer = TinyTokenizer()
    assert mib_name_token_id(tokenizer, "Alice") == 11
    tokenizer.encode = lambda *_args, **_kwargs: [1, 2]
    with pytest.raises(ValueError, match="one token"):
        mib_name_token_id(tokenizer, "Alice")


def test_regression_fit_and_serialization(tmp_path):
    expected = LinearCausalModel(0.05, 2.0, 0.75)
    same = [expected.predict_signals(1, 1)] * 5
    patched = {
        family: [expected.predict_signals(*family_signals(family))] * 5
        for family in FAMILY_SPECS
    }
    fitted, diagnostics = fit_linear_causal_model(same, patched)
    assert fitted.bias == pytest.approx(expected.bias)
    assert fitted.position_coeff == pytest.approx(expected.position_coeff)
    assert fitted.token_coeff == pytest.approx(expected.token_coeff)
    assert fitted.r2 == pytest.approx(1.0)
    path = tmp_path / "causal.json"
    path.write_text(json.dumps(fitted.as_dict()))
    assert LinearCausalModel.from_dict(json.loads(path.read_text())) == fitted
    assert diagnostics["n_observations"] == 20


def test_full_rank_rotation_is_full_vector_patch_and_source_equal_is_identity():
    torch.manual_seed(0)
    rotations = HeadRotations(((0, 0),), head_dim=4, subspace_dim=4)
    base = torch.randn(2, 3, 4)
    source = torch.randn(2, 3, 4)
    assert torch.allclose(
        rotations.intervene((0, 0), base, source), source, atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(rotations.intervene((0, 0), base, base), base, atol=1e-6)
    assert rotations.metadata()["heads"] == ["L0H0"]
    assert rotations.parameter_count == 16


def test_joint_rotation_count_matches_selected_heads():
    heads = all_gpt2_heads()
    rotations = HeadRotations(heads, subspace_dim=32)
    assert len(rotations.projectors) == 144
    assert rotations.parameter_count == 144 * 64 * 32


def test_plot_cost_uot_and_deterministic_ranking():
    heads = ((0, 0), (0, 1), (0, 2))
    model = LinearCausalModel(0.0, 2.0, 1.0)
    effects = {}
    for family in FAMILY_SPECS:
        position = model.effect("position", family)
        token = model.effect("token", family)
        effects[family] = np.asarray([[position, token, 99.0], [position, token, 99.0]])
    signatures = SignatureBank(
        heads=heads,
        factual={family: (3.0, 3.0) for family in FAMILY_SPECS},
        neural_effects=effects,
        runtime_seconds=0.0,
    )
    cost, _diagnostics = build_cost_matrix(signatures, model)
    assert cost.shape == (2, 3)
    assert cost[0, 0] == pytest.approx(0.0)
    assert cost[1, 1] == pytest.approx(0.0)
    coupling = sinkhorn_one_sided_uot(cost, epsilon=0.5, beta_neural=1.0)
    assert coupling.shape == (2, 3)
    assert np.isfinite(coupling).all()
    assert np.allclose(coupling.sum(axis=1), 1.0)
    assert top_k_heads(coupling, heads, "position", 1) == ((0, 0),)
    tied = np.ones((2, 3))
    assert top_k_heads(tied, heads, "token", 2) == ((0, 0), (0, 1))


def test_bruteforce_coupling_ranks_macro_family_mse_without_tuning():
    cost = np.asarray([[0.3, 0.1, 0.2], [5.0, 5.0, 4.0]], dtype=np.float64)
    coupling = bruteforce_coupling_from_cost(cost)
    heads = ((0, 0), (0, 1), (0, 2))
    assert coupling.shape == (2, 3)
    assert np.isfinite(coupling).all()
    assert np.array_equal(coupling, -cost)
    assert top_k_heads(coupling, heads, "position", 3) == (
        (0, 1), (0, 2), (0, 0)
    )
    assert top_k_heads(coupling, heads, "token", 1) == ((0, 2),)


def test_staged_calibration_never_receives_heldout_groups():
    seen = []

    def evaluator(_model, _tokenizer, groups, *, variable, heads, **_kwargs):
        seen.append(groups["sentinel"])
        return {"macro_mse": float(len(heads)), "variable": variable}

    selected, trials = calibrate_uot_grid(
        None,
        None,
        {"sentinel": "calibration-only"},
        causal_model=LinearCausalModel(0.0, 1.0, 1.0),
        cost=np.zeros((2, 4)),
        heads=((0, 0), (0, 1), (0, 2), (0, 3)),
        epsilons=(0.5, 1.0),
        beta_neurals=(0.1,),
        k_values=(1, 2, 3),
        device=torch.device("cpu"),
        batch_size=2,
        evaluator=evaluator,
    )
    assert len(trials) == 2
    assert selected["epsilon"] == 0.5
    assert all(float(trial["runtime_seconds"]) >= 0.0 for trial in trials)
    assert float(selected["runtime_seconds"]) >= 0.0
    assert seen and set(seen) == {"calibration-only"}
    assert choose_k([{"k": 2, "macro_mse": 1.0}, {"k": 1, "macro_mse": 1.0}])["k"] == 1


def test_offline_mocked_training_and_evaluation_pipeline():
    model = TinyCausalLM()
    freeze_model(model)
    tokenizer = PipelineTokenizer()
    groups = {
        family: tuple(
            IOIExample(
                row_id=index,
                split="fit",
                family=family,
                base_prompt=f"base{index}",
                source_prompt=f"source{index}",
                base_io="Alice",
                base_subject="Bob",
                source_io="Bob",
                source_subject="Alice",
            )
            for index in range(2)
        )
        for family in FAMILY_SPECS
    }
    causal = LinearCausalModel(0.0, 1.0, 0.5)
    rotations, training = train_joint_das(
        model,
        tokenizer,
        groups,
        variable="position",
        causal_model=causal,
        heads=((0, 0),),
        device=torch.device("cpu"),
        config=DASTrainConfig(
            subspace_dim=2,
            epochs=1,
            learning_rate=0.1,
            effective_batch_size=4,
            micro_batch_size=2,
        ),
    )
    metrics = evaluate_mse(
        model,
        tokenizer,
        groups,
        variable="position",
        causal_model=causal,
        heads=((0, 0),),
        rotations=rotations,
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert training["optimizer_steps"] == 2
    assert len(training["epoch_mse"]) == 1
    assert np.isfinite(metrics["macro_mse"])


@pytest.mark.skipif(
    not bool(__import__("os").environ.get("IOI_NETWORK_SMOKE")),
    reason="Set IOI_NETWORK_SMOKE=1 for the opt-in cached/network GPT-2 smoke test",
)
def test_opt_in_gpt2_smoke(tmp_path):
    from experiments.ioi.run import main

    assert main(
        [
            "--methods", "blind,plot,bruteforce,oracle", "--quick-rows", "100", "--calibration-rows", "50",
            "--signature-bank-size", "8", "--uot-epsilons", "1",
            "--uot-beta-neural", "1", "--plot-k", "1",
            "--microbatch-size", "4", "--effective-batch-size", "32",
            "--eval-batch-size", "8", "--das-dimension", "1",
            "--allow-coefficient-mismatch", "--output-dir", str(tmp_path / "smoke"),
            "--allow-version-mismatch",
        ]
    ) == 0
