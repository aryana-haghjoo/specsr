"""Every run must leave behind enough to reproduce it, locally *and* on W&B.

Until 2026-08-14 nothing in the package called ``wandb.save`` or
``log_artifact``, so W&B held the metrics, config and validation images of every
run ever trained here and **not one set of weights**. A run page had numbers you
could read and no model you could load, and the only copy of the weights sat on
a shared workstation whose single partition has hit 100% full with other users'
data.

The reproducibility half is not hypothetical either:
``runs/finetune_20260730_003724/sr1/`` -- the released SR1 -- holds two ``.pth``
files and no config at all, so its configuration had to be recovered by loading
candidates until one matched.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from specsr.training.common import state_fingerprint, write_run_manifest  # noqa: E402


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 2)


def test_fingerprint_is_stable_and_detects_a_changed_weight():
    """The guard SR2 uses to prove it left the redshift head alone."""
    module = _Tiny()
    before = state_fingerprint(module)

    assert state_fingerprint(module) == before, "fingerprint is not deterministic"

    with torch.no_grad():
        module.fc.weight[0, 0] += 1e-6
    assert state_fingerprint(module) != before, (
        "a modified weight did not change the fingerprint, so the guard that "
        "asserts SR2 never retunes the redshift head cannot fire"
    )


def test_sr2_freezes_the_redshift_head_by_default():
    """SR2 refines a reconstruction; it must not retune the estimator.

    Reopening `attn_score` and `z_logits` at 0.1x the SR2 learning rate was
    measured to take the held-out catastrophic-outlier rate from 10.84% frozen
    to 14.69% after one epoch and 17.83% by epoch 30, consistently across three
    independent runs -- and the adapted head was discarded at save time anyway,
    so the released pipeline pairs SR2 with the pre-adaptation head regardless.
    """
    from specsr.config import load_config
    from specsr.training.sr2 import DEFAULTS

    assert DEFAULTS["zhead_unfreeze_last_n"] == 0
    assert load_config("configs/sr2.yaml")["zhead_unfreeze_last_n"] == 0


def test_manifest_records_what_a_checkpoint_cannot(tmp_path):
    """Dataset, split and upstream checkpoints must be written beside the weights.

    A checkpoint does not say which split produced it. Two SR2 evaluations once
    disagreed (0.864069 vs 0.865459) purely because they sat on different SR1s,
    which is the class of ambiguity this file removes.
    """
    written = write_run_manifest(
        tmp_path,
        {"lr": 1e-4, "hidden_dim": 120},
        stage="sr2",
        dataset="data/paired_DR4_logR.npz",
        split_path="splits/groupsplit3_80_20_deadbeef.npz",
        upstream={"sr1_ckpt": "runs/x/best_superres_model.pth"},
    )

    assert {p.name for p in written} == {"config_resolved.yaml", "run_manifest.json"}

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["stage"] == "sr2"
    assert manifest["dataset"] == "data/paired_DR4_logR.npz"
    assert manifest["split_file"] == "splits/groupsplit3_80_20_deadbeef.npz"
    assert manifest["upstream"]["sr1_ckpt"] == "runs/x/best_superres_model.pth"

    # The *resolved* config, i.e. after any sweep overrides -- which is what
    # actually trained, and differs from the file passed to --config.
    cfg = yaml.safe_load((tmp_path / "config_resolved.yaml").read_text())
    assert cfg["lr"] == pytest.approx(1e-4)
    assert cfg["hidden_dim"] == 120


def test_artifact_logging_is_a_no_op_without_an_active_run(tmp_path):
    """Must not raise when W&B is disabled, or a smoke run would crash at the end."""
    from specsr.training.common import log_checkpoint_artifact

    ckpt = tmp_path / "best.pth"
    ckpt.write_bytes(b"not really a checkpoint")
    log_checkpoint_artifact([ckpt], kind="sr1", metadata={"lr": 1e-4})


def test_artifact_logging_survives_a_wandb_directory_shadowing_the_library(
    tmp_path, monkeypatch
):
    """An importable `wandb` that is not the library must still be a no-op.

    W&B writes its run data to a `wandb/` directory in the working directory. If
    the library itself is not installed, that directory is importable as a
    *namespace package*: `import wandb` succeeds and hands back a module object
    with no `run` attribute. A bare `wandb.run` then raises AttributeError from
    inside the one function whose contract is that it never raises -- at the end
    of training, with the checkpoint already on disk.

    Found on a Python 3.10 environment where wandb was absent but a leftover
    run directory was not.
    """
    import sys
    import types

    from specsr.training.common import log_checkpoint_artifact

    shadow = types.ModuleType("wandb")          # no `run`, no `Artifact`
    monkeypatch.setitem(sys.modules, "wandb", shadow)

    ckpt = tmp_path / "best.pth"
    ckpt.write_bytes(b"not really a checkpoint")
    log_checkpoint_artifact([ckpt], kind="sr1", metadata={"lr": 1e-4})


def test_artifact_logging_survives_wandb_not_being_installed(tmp_path, monkeypatch):
    """`wandb` is optional; a missing import must be a no-op, not an exception.

    This is not hypothetical -- CI installs the package without wandb, and the
    import originally sat *above* the guard, so the function whose contract is
    "never raises" raised `ModuleNotFoundError` at the end of training. It could
    not be caught on a developer machine, where wandb is always installed, which
    is why the failure only appeared on CI.
    """
    import sys

    from specsr.training.common import log_checkpoint_artifact, write_run_manifest

    # A `None` entry in sys.modules makes `import wandb` raise ImportError,
    # which is what an environment without the package does.
    monkeypatch.setitem(sys.modules, "wandb", None)

    ckpt = tmp_path / "best.pth"
    ckpt.write_bytes(b"not really a checkpoint")
    log_checkpoint_artifact([ckpt], kind="sr1", metadata={"lr": 1e-4})

    # The manifest is written from the same code path and must survive too.
    written = write_run_manifest(tmp_path, {"lr": 1e-4}, stage="sr1")
    assert all(p.exists() for p in written)
