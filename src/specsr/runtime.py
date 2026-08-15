"""Environment-variable overrides for the training scripts.

The SR1 and SR2 training scripts carry their hyperparameters in a literal dict
with no command-line interface, so there was no way to run them briefly. That
makes an overnight chain risky: a typo or a missing checkpoint in stage 3 only
surfaces hours in, after stages 1 and 2 have already burned the night.

This module lets any stage be run in a shortened form **through exactly the same
code path**, driven by environment variables so no call site has to change:

``SPECSR_EPOCHS``
    Override the epoch count.
``SPECSR_LIMIT_TRAIN_BATCHES`` / ``SPECSR_LIMIT_VAL_BATCHES``
    Stop each epoch after this many batches. This is what makes a smoke run take
    seconds rather than an hour — an epoch over the DR4 product is ~1,300 steps.
``SPECSR_OUT_DIR``
    Redirect checkpoints and artefacts, so a smoke run cannot overwrite real
    weights.
``SPECSR_DATASET``
    Override the dataset path.
``SPECSR_WANDB_MODE``
    Set to ``disabled`` or ``offline`` for smoke runs that should not create
    cloud runs.

A smoke run must exercise the real code path. Reimplementing a shortened
version of the loop would test the reimplementation, not the thing that runs
overnight.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from typing import Any

__all__ = [
    "apply_env_overrides",
    "limit_batches",
    "is_smoke_run",
    "env_int",
    "describe_overrides",
    "resolve_dataset",
    "wandb_mode",
]


def env_int(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def is_smoke_run() -> bool:
    """True when any batch limit is set, i.e. this is not a full run."""
    return bool(
        os.environ.get("SPECSR_LIMIT_TRAIN_BATCHES")
        or os.environ.get("SPECSR_LIMIT_VAL_BATCHES")
    )


def apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Return ``config`` with environment overrides applied.

    Called immediately after a stage builds its config dict and before that dict
    reaches W&B, so the run record shows what actually ran.
    """
    cfg = dict(config)

    epochs = env_int("SPECSR_EPOCHS")
    if epochs is not None:
        cfg["epochs"] = epochs
        # Several stages refuse to checkpoint before a minimum epoch count;
        # leaving those at their full-run values means a short run produces
        # nothing to hand to the next stage.
        for key in ("min_epochs", "early_stop_min_epochs", "sharp_wd_start_epoch"):
            if key in cfg:
                cfg[key] = min(int(cfg[key]), epochs)
        # A warmup is a phase boundary, not a floor: clamping it to `epochs`
        # would leave a short run entirely inside warmup, so the ramp it exists
        # to test never switches on. Keep it strictly inside the run instead.
        for key in ("lam_sparse_warmup", "lam_z_warmup"):
            if key in cfg:
                cfg[key] = max(1, min(int(cfg[key]), epochs // 4))

    dataset = os.environ.get("SPECSR_DATASET")
    if dataset:
        for key in ("dataset_npz", "dataset", "data"):
            if key in cfg:
                cfg[key] = dataset

    return cfg


def limit_batches(loader: Iterable, limit: int | None = None, *, kind: str = "train") -> Iterator:
    """Yield at most ``limit`` batches from ``loader``.

    ``limit`` defaults to ``SPECSR_LIMIT_TRAIN_BATCHES`` or
    ``SPECSR_LIMIT_VAL_BATCHES`` depending on ``kind``. With no limit set this is
    a transparent pass-through, so wrapping a loop costs nothing in a real run.
    """
    if limit is None:
        var = "SPECSR_LIMIT_TRAIN_BATCHES" if kind == "train" else "SPECSR_LIMIT_VAL_BATCHES"
        limit = env_int(var)
    if limit is None or limit <= 0:
        yield from loader
        return
    for i, batch in enumerate(loader):
        if i >= limit:
            break
        yield batch


def describe_overrides() -> str:
    """One-line summary of active overrides, for logging at stage start."""
    keys = (
        "SPECSR_EPOCHS",
        "SPECSR_LIMIT_TRAIN_BATCHES",
        "SPECSR_LIMIT_VAL_BATCHES",
        "SPECSR_OUT_DIR",
        "SPECSR_DATASET",
        "SPECSR_WANDB_MODE",
    )
    active = {k: os.environ[k] for k in keys if os.environ.get(k)}
    if not active:
        return "FULL RUN: no environment overrides"
    body = ", ".join(f"{k}={v}" for k, v in active.items())
    # The chain script always sets the path variables, so their presence alone
    # does not mean this is a rehearsal. Only the batch limits shorten a run,
    # and mislabelling a real run as a smoke test in its own log is exactly the
    # kind of thing that gets a good run discarded later.
    kind = "SMOKE RUN" if is_smoke_run() else "FULL RUN"
    return f"{kind}: {body}"


def resolve_dataset(default: str) -> str:
    """Return ``SPECSR_DATASET`` if set, else ``default``.

    The SR1 and SR2 scripts hold their dataset path in a local variable rather
    than in the config dict, so :func:`apply_env_overrides` never reached it.
    That was not cosmetic: a preflight check could validate one dataset while the
    stage silently trained on another, which a smoke run caught doing exactly
    that — validating DR4 and then training on DR3.

    The resolved path is echoed on a machine-readable line so the chain script
    can assert every stage agrees on the dataset. A stage training on the wrong
    product is invisible in the metrics and expensive to discover late.
    """
    path = os.environ.get("SPECSR_DATASET") or default
    _reject_pre_leak_fix_dataset(path)
    print(f"[specsr] dataset: {os.path.abspath(path)}", flush=True)
    return path


# Products built before the augmentation leak was fixed. 99.2% of galaxies had
# rows on both sides of the split, so any metric measured against these is
# meaningless -- and they are on the superseded 2,500-point linear grid besides.
# They are kept only because a separate, retired analysis still reads them.
_PRE_LEAK_FIX = ("spectra_dataset_2500",)


def _reject_pre_leak_fix_dataset(path: str) -> None:
    """Refuse to train on a dataset built before the leak fix.

    Several stage configs still named these products long after the fix, and
    nothing complained: training completes, the metrics look normal, and the
    numbers are worthless. Set ``SPECSR_ALLOW_LEAKED_DATASET=1`` to override,
    which is only legitimate for reproducing a historical result.
    """
    if os.environ.get("SPECSR_ALLOW_LEAKED_DATASET"):
        return
    name = os.path.basename(str(path))
    if any(bad in name for bad in _PRE_LEAK_FIX):
        raise RuntimeError(
            f"Refusing to use {name}: it was built before the augmentation leak was "
            "fixed, so ~99% of galaxies have rows in both train and test and every "
            "metric from it is invalid. Use data/paired_DR4_logR.npz, or set "
            "SPECSR_ALLOW_LEAKED_DATASET=1 if you genuinely mean it."
        )


def wandb_mode(default: str = "online") -> str:
    """W&B mode, overridable via ``SPECSR_WANDB_MODE``.

    Smoke runs set this to ``offline`` so rehearsals do not litter the project
    with cloud runs that look like real experiments.
    """
    return os.environ.get("SPECSR_WANDB_MODE") or default
