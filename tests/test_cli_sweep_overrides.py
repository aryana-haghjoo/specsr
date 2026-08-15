"""`specsr train` must accept the arguments a W&B sweep agent appends.

A sweep agent runs the `command:` block with every searched parameter appended
as ``--name=value`` (the ``${args}`` macro). The CLI used to call
``parse_args``, which rejects any flag it has not declared -- and the searched
names cannot be declared in advance, since they differ per sweep. So **every
trial of every sweep would have died at argument parsing**, before importing
torch. No sweep had yet been launched against the package CLI, which is why the
break went unnoticed for so long.

The second half is subtler and would have been much harder to notice: the
trainers build their config from ``DEFAULTS`` updated by ``--config``, and never
consulted ``wandb.config``. Had the parser been made permissive on its own, the
agent's parameters would have been recorded by W&B and silently ignored by the
model. Every trial would have trained the same network, the sweep would have
completed, and its parameter-importance plot -- the entire deliverable -- would
have been noise with a plausible shape.

Both halves are asserted here.
"""
from __future__ import annotations

import pytest

from specsr.cli import main
from specsr.training.runner import run_train


class _Args:
    """Stand-in for the parsed namespace `run_train` consumes."""

    def __init__(self, **kw):
        self.stage = kw.pop("stage")
        self.config = kw.pop("config")
        self.overrides = kw.pop("overrides", {})
        self.dataset = None
        self.out_dir = None
        self.source = None
        for name in ("sr1_ckpt", "sr1_config", "sr2_ckpt", "zhead_ckpt",
                     "zhead_bootstrap_ckpt", "init_ckpt"):
            setattr(self, name, None)
        self.__dict__.update(kw)


def test_cli_accepts_sweep_style_arguments(monkeypatch, tmp_path):
    """`--name=value` must parse, and must arrive typed rather than as strings."""
    seen = {}

    def fake_train_sr1(cfg, **kw):
        seen.update(cfg)
        return tmp_path

    monkeypatch.setattr("specsr.training.sr1.train_sr1", fake_train_sr1)

    rc = main([
        "train", "sr1",
        "--config", "configs/finetune/sr1.yaml",
        "--lr=0.0001",
        "--hidden_dim=96",
        "--use_var_clamp=false",
    ])

    assert rc == 0
    # Typed by the YAML scalar parser, not left as strings -- "false" as a string
    # is truthy, which would silently invert a boolean switch.
    assert seen["lr"] == pytest.approx(1e-4)
    assert seen["hidden_dim"] == 96
    assert seen["use_var_clamp"] is False


def test_sweep_overrides_beat_the_config_file(monkeypatch, tmp_path):
    """The agent's value must win over `--config`, or the sweep varies nothing.

    `configs/finetune/sr1.yaml` sets hidden_dim 120. If the file won, every
    trial would train an identical network while W&B recorded 60 distinct
    configurations.
    """
    seen = {}
    monkeypatch.setattr(
        "specsr.training.sr1.train_sr1",
        lambda cfg, **kw: (seen.update(cfg), tmp_path)[1],
    )

    main(["train", "sr1", "--config", "configs/finetune/sr1.yaml", "--hidden_dim=144"])

    assert seen["hidden_dim"] == 144, (
        "the config file overrode the sweep parameter; every trial would train "
        "the same model"
    )


def test_override_naming_a_dead_key_is_rejected():
    """A name the stage does not read must fail loudly, at trial 1.

    Silently accepting it reproduces the earlier SR2 sweep failure, where eight of
    eleven search axes were names nothing read: the trials run, the optimiser
    attributes their non-effect to noise, and the search burns its budget on
    axes with no effect while appearing to explore them.
    """
    args = _Args(stage="sr1", config="configs/finetune/sr1.yaml",
                 overrides={"lam_shape_max": 1.0})

    with pytest.raises(SystemExit, match="not read by this stage"):
        run_train(args)


def test_unknown_flags_still_error_outside_train():
    """Only `train` is permissive. Elsewhere an unknown flag is a typo."""
    with pytest.raises(SystemExit):
        main(["evaluate", "residuals", "--not-a-real-flag=1"])
