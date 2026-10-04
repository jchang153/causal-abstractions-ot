from types import SimpleNamespace

import pytest
import torch

from mcqa_experiment import das
from mcqa_experiment.sites import ResidualSite


@pytest.mark.parametrize("evaluate_holdout", [False, True])
def test_calibration_selects_handle_and_controls_test_access(monkeypatch, tmp_path, evaluate_holdout):
    model = torch.nn.Linear(4, 4)
    model.config = SimpleNamespace(hidden_size=4)
    banks = {name: SimpleNamespace(split=name, target_var="answer_token")
             for name in ("train", "calibration", "test")}
    sites = [ResidualSite(layer, "last_token", 0, 4) for layer in (0, 1)]
    calls = []

    def train(**kwargs):
        return torch.nn.Linear(4, 4), [1.0]

    def evaluate(**kwargs):
        bank, site = kwargs["bank"], kwargs["site"]
        calls.append((bank.split, site.layer))
        # The calibration winner differs from the hypothetical test winner.
        score = (0.9 if site.layer == 1 else 0.2) if bank.split == "calibration" else 0.1
        return {"iia_acc": score}

    monkeypatch.setattr(das, "train_das_candidate", train)
    monkeypatch.setattr(das, "evaluate_das_candidate", evaluate)
    checkpoint = tmp_path / "selected.pt"
    payload = das.run_das_pipeline(
        model=model, train_bank=banks["train"], calibration_bank=banks["calibration"],
        holdout_bank=banks["test"], sites=sites, device="cpu", tokenizer=None,
        config=das.DASConfig(subspace_dims=(2,), verbose=False,
                             evaluate_holdout=evaluate_holdout,
                             selected_checkpoint_path=checkpoint),
    )
    assert payload["test_evaluated"] is evaluate_holdout
    assert payload["test_used_for_selection"] is False
    assert payload["results"][0]["layer"] == 1
    assert [call for call in calls if call[0] == "test"] == (
        [("test", 1)] if evaluate_holdout else []
    )
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["selected_record"]["layer"] == 1
    assert saved["state_dict"]


def test_calibration_only_blocks_candidate_test_diagnostics(monkeypatch, tmp_path):
    model = torch.nn.Linear(4, 4)
    model.config = SimpleNamespace(hidden_size=4)
    banks = {name: SimpleNamespace(split=name, target_var="answer_token")
             for name in ("train", "calibration", "test")}

    def evaluate(**kwargs):
        assert kwargs["bank"].split == "calibration"
        return {"iia_acc": 0.5}

    monkeypatch.setattr(das, "train_das_candidate", lambda **kwargs: (torch.nn.Linear(4, 4), [1.0]))
    monkeypatch.setattr(das, "evaluate_das_candidate", evaluate)
    payload = das.run_das_pipeline(
        model=model, train_bank=banks["train"], calibration_bank=banks["calibration"],
        holdout_bank=banks["test"], sites=[ResidualSite(0, "last_token", 0, 4)],
        device="cpu", tokenizer=None,
        config=das.DASConfig(subspace_dims=(2,), verbose=False, evaluate_holdout=False,
                             store_candidate_holdout_metrics=True),
    )
    assert payload["test_evaluated"] is False
    assert payload["timing_seconds"]["t_candidate_holdout_eval"] == 0.0
