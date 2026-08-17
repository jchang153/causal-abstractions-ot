from __future__ import annotations

import argparse

from experiments.binary_addition.run_mib_baselines import (
    artifact_protocol,
    can_resume,
    resume_protocol,
)


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        rows="C1,C2,C3",
        timesteps="0,1,2,3",
        width=4,
        hidden_size=16,
        fit_bases=128,
        calib_bases=64,
        test_bases=64,
        source_policy="structured_26_top3carry_c2x5_c3x7_no_random",
        batch_size=64,
        eval_batch_size=512,
        epochs=8,
        learning_rate=0.01,
        temperature_start=1.0,
        temperature_end=0.01,
        regularization_coefficient=0.0,
        regularization_coefficients="0,1e-5,1e-4,1e-3",
        pca_rank=None,
        train_on="all",
        train_epochs=120,
        train_batch_size=64,
        train_lr=0.02,
    )


def test_resume_protocol_rejects_stale_hyperparameters(tmp_path) -> None:
    args = _args()
    checkpoint = tmp_path / "gru.pt"
    expected = artifact_protocol(
        resume_protocol(args, seed=3, checkpoint=str(checkpoint)),
        artifact="candidate",
        method="dbm-canonical",
        row="C2",
        timestep=1,
    )
    assert can_resume({"resume_protocol": expected}, expected)
    assert not can_resume({}, expected)

    changed = _args()
    changed.epochs = 9
    stale = artifact_protocol(
        resume_protocol(changed, seed=3, checkpoint=str(checkpoint)),
        artifact="candidate",
        method="dbm-canonical",
        row="C2",
        timestep=1,
    )
    assert not can_resume({"resume_protocol": stale}, expected)


def test_resume_protocol_distinguishes_candidate_grid_entries(tmp_path) -> None:
    base = resume_protocol(_args(), seed=0, checkpoint=str(tmp_path / "gru.pt"))
    c1_t0 = artifact_protocol(
        base,
        artifact="candidate",
        method="full-state",
        row="C1",
        timestep=0,
    )
    c1_t1 = artifact_protocol(
        base,
        artifact="candidate",
        method="full-state",
        row="C1",
        timestep=1,
    )
    assert c1_t0 != c1_t1
    assert not can_resume({"resume_protocol": c1_t0}, c1_t1)


def test_resume_protocol_distinguishes_regularization_candidates(tmp_path) -> None:
    base = resume_protocol(_args(), seed=0, checkpoint=str(tmp_path / "gru.pt"))
    unregularized = artifact_protocol(
        base,
        artifact="candidate",
        method="dbm-canonical",
        row="C1",
        timestep=0,
        regularization_coefficient=0.0,
    )
    regularized = artifact_protocol(
        base,
        artifact="candidate",
        method="dbm-canonical",
        row="C1",
        timestep=0,
        regularization_coefficient=1e-4,
    )
    assert unregularized != regularized
    assert not can_resume({"resume_protocol": unregularized}, regularized)
