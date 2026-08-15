"""Every sweep must optimise a metric the training loop actually logs.

Both redshift-head sweeps optimised ``val_med_abs_dz_over_1pz`` while the loop
logged nothing of the sort. W&B does not complain: the agent runs, trials
complete, and the Bayes optimiser simply has no signal to rank them by. Seventy-
five trials would have produced a "best" configuration chosen at random.

A sweep is also useless if its ``command:`` names a program that does not exist,
which every sweep did after the training loops moved into the package.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]
SWEEPS = sorted((REPO / "configs" / "sweeps").glob("*.yaml"))

# Which stage each sweep drives, and hence which module logs its metrics.
_STAGE_FOR = {"sr1": "sr1", "sr2": "sr2", "zhead": "zhead"}

# Run length, not model. A sweep runs shorter trials than a from-scratch config
# on purpose -- hyperband kills them earlier still -- so these may differ.
_BUDGET_KEYS = {"epochs", "min_epochs", "patience"}

# Divergences that are real and *unresolved*, listed rather than hidden so the
# test still catches anything new.
#
# The SR1 sweep pins the line-mask parameters of the released SR1
# (`mask_dilate: 11`, `mask_min_width: 7`, matching
# `checkpoints/checkpoints_run5_20260728/config_logR.yaml` and
# `configs/finetune/sr1.yaml`). `configs/sr1.yaml` carries 35 and 15, which look
# like the old values scaled by the 6671/2500 change in grid length. Whether
# that rescale was ever *measured* is unknown -- the equivalent arithmetic for
# SR2's presence mask was wrong when it was finally checked, flagging 1.1% of
# bright lines. Needs a measurement on the val split, not a decision here.
#
# Empty, and worth keeping that way. The three archived sweep configs that
# needed entries here were deleted on 2026-08-14: their replacements carry the
# same history in their headers, and git keeps the originals. A new entry means
# a live sweep disagrees with the config it passes, which needs a reason.
_KNOWN_DIVERGENCES: dict[str, set[str]] = {}


def _stage(path: Path) -> str:
    return _STAGE_FOR[path.stem.split("_")[0]]


def float_ne(a, b) -> bool:
    """True when a and b differ, tolerating 1e-4 vs 0.0001 and 1 vs 1.0."""
    try:
        return float(a) != float(b)
    except (TypeError, ValueError):
        return str(a) != str(b)


def _logged_metric_keys(stage: str) -> set[str]:
    """Keys the stage passes to ``wandb.log``, including the expanded metric dict.

    Parsed from source rather than executed: importing is cheap but *running* a
    stage is not, and the point is to check the contract statically.
    """
    src = (REPO / "src" / "specsr" / "training" / f"{stage}.py").read_text()
    body = src.split("wandb.log(")[1][:800]
    keys = set(re.findall(r'"([a-zA-Z0-9_/]+)"\s*:', body))

    # zhead splats redshift_metrics() in under a `val_` prefix.
    if 'f"val_{k}"' in body:
        from specsr.metrics import redshift_metrics

        keys |= {f"val_{k}" for k in redshift_metrics([1.0, 2.0], [1.1, 2.1])}
    return keys


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_sweep_metric_is_actually_logged(path: Path):
    cfg = yaml.safe_load(path.read_text())
    metric = cfg["metric"]["name"]
    logged = _logged_metric_keys(_stage(path))

    assert metric in logged, (
        f"{path.name} optimises {metric!r}, which {_stage(path)}.py never logs. "
        f"The sweep will run to completion and rank nothing. Logged keys: "
        f"{sorted(logged)}"
    )


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_sweep_parameters_are_read_by_the_trainer(path: Path):
    """Every swept name must reach a hyperparameter the stage actually uses.

    W&B will happily vary a name nothing reads. The trial runs, the value is
    logged, and the Bayes optimiser attributes the difference it did not make to
    noise -- so the search burns its budget on axes with no effect while looking
    like it is exploring.

    An earlier SR2 sweep config held 28 such names, and *eight of its eleven swept
    axes* were among them: teacher forcing, an anti-hallucination ramp and an
    entire line-shape schedule, all belonging to the retired
    `train/sr2_best/train_sr2_attention.py`.
    """
    import importlib

    params = yaml.safe_load(path.read_text()).get("parameters", {})
    defaults = importlib.import_module(f"specsr.training.{_stage(path)}").DEFAULTS
    # Consumed via `cfg.get(...)` or handed to wandb.init rather than defaulted.
    allowed = set(defaults) | {"dataset_npz", "wandb_name", "wandb_project", "source"}

    dead = sorted(set(params) - allowed)
    assert not dead, (
        f"{path.name} names {dead}, which {_stage(path)}.py never reads. Any of "
        "these that carry `values:` are search axes with no effect on the model."
    )


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_sweep_does_not_contradict_the_config_it_passes(path: Path):
    """A pinned sweep value must agree with the `--config` file it overrides.

    W&B appends `${args}` *after* `--config`, so anything the sweep pins wins.
    An earlier SR2 sweep config pinned `cnn_dim: 64`, `num_cnn_blocks: 4` and
    `window_half: 25` while passing a config that says 96, 6 and 80 -- so every
    trial trained a smaller network than the released model, and the two files
    read as agreeing unless you know which one takes precedence.

    Only single-valued (`value:`) entries are checked. A `values:` list is a
    search axis and is *supposed* to override the config.
    """
    sweep = yaml.safe_load(path.read_text())
    command = [str(c) for c in sweep.get("command", [])]
    if "--config" not in command:
        pytest.skip(f"{path.name} passes no --config")
    cfg_rel = command[command.index("--config") + 1]
    cfg_path = REPO / cfg_rel
    if not cfg_path.exists():
        pytest.skip(f"{cfg_rel} not available")

    from specsr.config import load_config

    cfg = load_config(cfg_path)
    ignore = _BUDGET_KEYS | _KNOWN_DIVERGENCES.get(path.name, set())
    clashes = [
        (k, spec["value"], cfg[k])
        for k, spec in sweep.get("parameters", {}).items()
        if isinstance(spec, dict) and "value" in spec and k in cfg
        and k not in ignore
        and str(spec["value"]) != str(cfg[k])
        and float_ne(spec["value"], cfg[k])
    ]
    assert not clashes, (
        f"{path.name} pins values that silently override {cfg_rel}: "
        + "; ".join(f"{k} sweep={s!r} config={c!r}" for k, s, c in clashes)
    )


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_sweep_invokes_the_package_cli(path: Path):
    """The program must be `specsr.cli`, not a loose script path.

    Sweeps used to name `train_sr1.py` and friends, which no longer exist. A
    sweep pointing at a missing file fails on every trial.
    """
    command = yaml.safe_load(path.read_text()).get("command")
    assert command, f"{path.name} has no command: block"
    joined = " ".join(str(c) for c in command)

    assert "specsr.cli" in joined, (
        f"{path.name} does not invoke the package CLI: {joined!r}"
    )
    stray = [c for c in command if isinstance(c, str) and c.endswith(".py")]
    assert not stray, (
        f"{path.name} still names loose scripts {stray}; training lives in "
        "specsr.training and is reached through `specsr train`"
    )


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_sweep_does_not_reference_the_leaked_dataset(path: Path):
    """No sweep may point at a pre-leak-fix product.

    `resolve_dataset` refuses these at runtime, but a sweep that dies on trial 1
    still costs a launch and a confusing W&B page.
    """
    text = path.read_text()
    assert "spectra_dataset_2500" not in text, (
        f"{path.name} references the pre-leak-fix dataset, where ~99% of "
        "galaxies straddle the split"
    )


def _referenced_paths(path: Path) -> list[str]:
    command = yaml.safe_load(path.read_text()).get("command", [])
    return [
        c for c in command
        if isinstance(c, str) and ("/" in c) and c.endswith((".pth", ".yaml"))
    ]


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_sweep_paths_are_repo_relative(path: Path):
    """Pinned paths must be relative to the repo, not to somebody's home directory.

    `configs/sr2.yaml` was found holding
    an absolute path into another group member's home directory --
    another machine, and a directory that no longer exists. An absolute path
    also silently breaks every sweep agent that is not the machine it was
    written on, which is the normal case for a sweep.

    This runs everywhere, including CI, because it needs no artifacts.
    """
    absolute = [c for c in _referenced_paths(path) if c.startswith(("/", "~"))]
    assert not absolute, (
        f"{path.name} pins absolute paths {absolute}; use repo-relative paths so "
        "any agent on any machine resolves them the same way"
    )


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_sweep_referenced_paths_exist(path: Path):
    """Pinned upstream checkpoints must actually be there.

    A sweep names its upstream so every trial refines the same frozen model. A
    path that has moved makes all N trials fail identically, which reads like a
    code bug rather than a stale config.

    **Skips when the artifacts are absent**, which is the normal state in CI:
    checkpoints are gitignored and live on the Hub. This test earns its keep on
    a developer machine, immediately before a sweep is launched -- the only
    moment the check matters. An earlier version asserted unconditionally and
    turned CI red for a repo that is behaving correctly.
    """
    referenced = _referenced_paths(path)

    # Split by whether git is expected to carry the file. Anything under
    # `configs/` is tracked and must exist everywhere, including CI. Checkpoint
    # archives are gitignored and live on the Hub, so their absence is normal
    # rather than a defect -- an earlier version of this test lumped the two
    # together and turned CI red for a correctly-behaving repo.
    tracked = [c for c in referenced if c.startswith("configs/")]
    artifacts = [c for c in referenced if not c.startswith("configs/")]

    missing_tracked = [c for c in tracked if not (REPO / c).exists()]
    assert not missing_tracked, (
        f"{path.name} references tracked files that do not exist: {missing_tracked}"
    )

    missing_artifacts = [c for c in artifacts if not (REPO / c).exists()]
    if missing_artifacts:
        if len(missing_artifacts) == len(artifacts):
            pytest.skip("pinned checkpoints not available locally (gitignored)")
        # A partially-present archive is a real problem: it means the directory
        # exists but has been pruned or renamed under the sweep's feet.
        pytest.fail(
            f"{path.name} pins {len(artifacts)} artifacts but {missing_artifacts} "
            "are missing while others are present -- the archive has moved or been "
            "pruned, and every trial in this sweep would fail identically."
        )
