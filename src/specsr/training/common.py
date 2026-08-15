"""Pieces every training stage needs: seeding, parameter groups, run directories.

Small and dull on purpose. These used to be copied into each of the six training
scripts, where they drifted -- ``get_activation`` alone existed in three
versions, one of which silently ignored ``alpha`` for ELU.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "set_seed",
    "build_param_groups",
    "resolve_out_dir",
    "clip_grad_or_skip",
    "log_checkpoint_artifact",
    "state_fingerprint",
    "write_run_manifest",
    "EMA",
]


def state_fingerprint(module) -> str:
    """SHA-256 over a module's state dict, for asserting weights did not move.

    Used to prove that a stage left a frozen upstream module untouched. Checking
    ``requires_grad`` is not enough on its own: a module can still be mutated by
    something that writes to the tensors directly, and the question we care
    about -- did these exact weights change -- is answerable directly.
    """
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_run_manifest(out_dir, cfg: dict, *, stage: str, dataset=None,
                       split_path=None, upstream: dict | None = None) -> list[Path]:
    """Write everything needed to reconstruct a run, beside its weights.

    A checkpoint alone is not reproducible: it does not record which dataset or
    which split produced it, nor which upstream checkpoints it sits on. That has
    already cost this project real time -- ``runs/finetune_20260730_003724/sr1/``
    holds two ``.pth`` files and no config whatsoever, so the released SR1's
    configuration had to be recovered by loading candidates until one matched.

    Writes ``config_resolved.yaml`` (the merged config *including* any sweep
    overrides, which is what actually trained) and ``run_manifest.json``, and
    returns both paths so they can go up to W&B with the checkpoints.
    """
    import json
    import subprocess
    import time

    import yaml

    out_dir = Path(out_dir)
    cfg_path = out_dir / "config_resolved.yaml"
    manifest_path = out_dir / "run_manifest.json"

    def _plain(value):
        """Coerce to something YAML/JSON will accept, without guessing."""
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except Exception:  # noqa: BLE001 -- provenance is best-effort, never fatal
        commit = None

    run_id = None
    try:
        import wandb

        run_id = wandb.run.id if wandb.run is not None else None
    except Exception:  # noqa: BLE001
        pass

    with cfg_path.open("w") as fh:
        yaml.safe_dump({k: _plain(v) for k, v in cfg.items()}, fh, sort_keys=True)

    manifest = {
        "stage": stage,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": commit,
        "wandb_run_id": run_id,
        "dataset": _plain(dataset),
        "split_file": _plain(split_path),
        "upstream": {k: _plain(v) for k, v in (upstream or {}).items()},
        "out_dir": str(out_dir),
    }
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    return [cfg_path, manifest_path]


def log_checkpoint_artifact(paths, *, kind: str, metadata: dict | None = None) -> None:
    """Upload finished checkpoints to W&B as a run artifact.

    Nothing in this package did this, so W&B held the metrics, config and
    validation plots of every run ever trained here and **not one set of
    weights**. That is an exposure rather than an untidiness: this is a shared
    workstation on a single 1.5 TB partition that has hit 100% full with other
    users' data, and a checkpoint living only under ``runs/`` is one cleanup
    away from leaving its own metrics attached to no model at all.

    Does nothing when there is no active run -- a direct call outside W&B,
    ``SPECSR_WANDB_MODE=disabled``, or W&B not installed at all -- and never
    raises. A failed upload must not destroy a model that has already finished
    training, so the error is reported and the local path is printed instead.
    """
    # `wandb` is an optional dependency: the package installs, imports and
    # evaluates without it, and CI does not install it. The import therefore
    # belongs inside the guard rather than above it -- an ImportError here would
    # break the "never raises" contract at the one moment it matters, after
    # training has already finished.
    try:
        import wandb
    except ImportError:
        return

    # A successful import is not proof the library is there. W&B writes its run
    # data to a `wandb/` directory in the working directory, and when the
    # library is *not* installed that directory is importable as a namespace
    # package -- so `import wandb` succeeds and returns something with no `run`
    # attribute. `getattr` keeps the "never raises" promise in that case, where
    # a bare attribute access raises AttributeError at the end of training.
    if getattr(wandb, "run", None) is None:
        return
    files = [Path(p) for p in paths]
    files = [p for p in files if p.exists()]
    if not files:
        return
    try:
        artifact = wandb.Artifact(
            f"{kind}-{wandb.run.id}", type="model", metadata=dict(metadata or {})
        )
        for path in files:
            artifact.add_file(str(path))
        wandb.log_artifact(artifact)
        names = ", ".join(p.name for p in files)
        print(f"[specsr] W&B artifact {kind}-{wandb.run.id}: {names}", flush=True)
    except Exception as exc:  # noqa: BLE001 -- see docstring
        print(
            f"[specsr] WARNING: W&B artifact upload failed ({exc}). Checkpoints are "
            f"intact at {files[0].parent}",
            flush=True,
        )


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and Torch together.

    Note this does not make CUDA fully deterministic -- cuDNN picks algorithms
    by benchmark. Runs are reproducible to within that, not bit-exact.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 -- legacy global seed, matches trained runs
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_param_groups(
    model: nn.Module, lr: float, weight_decay: float
) -> list[dict[str, Any]]:
    """Split parameters into decayed and undecayed groups.

    Weight decay is applied to matrix-like weights only. Biases, and every
    normalisation parameter, are exempt: decaying a normalisation scale pulls the
    layer towards an output of zero, which is not a regularisation of anything
    meaningful and measurably hurts.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_norm = "norm" in name.lower()
        if p.ndim >= 2 and "weight" in name and not is_norm:
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
    ]


def resolve_out_dir(out_dir: str | Path | None, default_name: str) -> Path:
    """Directory for checkpoints and artefacts, created if needed.

    Every stage takes this explicitly rather than writing to the working
    directory. The old scripts saved ``best_superres_model.pth`` to CWD, so where
    a checkpoint landed depended on where you happened to launch from, and a
    rerun silently overwrote the previous one in place. That is how a 1-epoch
    smoke run once replaced a 10-hour model.

    When no directory is given and we are running under a W&B sweep agent, the
    run id is appended. Without it every trial in a sweep resolves to the same
    ``runs/<stage>`` path: trials overwrite each other's checkpoints, two agents
    on one machine corrupt a file mid-write, and the best trial's weights are
    gone by the time the sweep finishes. W&B has no ``${run_id}`` macro for a
    sweep's ``command:`` block, so the uniqueness has to come from here.
    """
    if out_dir is not None:
        path = Path(out_dir)
    else:
        run_id = os.environ.get("WANDB_RUN_ID")
        name = f"{default_name}_{run_id}" if run_id else default_name
        path = Path("runs") / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def clip_grad_or_skip(parameters, max_norm: float) -> tuple[float, bool]:
    """Clip gradients, and report whether the optimiser step is safe to apply.

    Returns ``(grad_norm, ok)``. When ``ok`` is False the caller must skip
    ``optimizer.step()`` and zero the gradients -- applying them would destroy
    the model.

    Why this exists rather than a bare ``clip_grad_norm_``: **clipping does not
    protect against a non-finite gradient, it launders one into every weight.**
    ``clip_grad_norm_`` computes ``scale = max_norm / (total_norm + eps)``; if
    any single element is ``inf`` the norm is ``inf``, the scale is ``0``, and
    the offending element becomes ``inf * 0 = nan``. The optimiser then writes
    NaN into every parameter it touches, and Adam's moments keep it there
    forever. This is not hypothetical: it destroyed the 2026-07-30 SR2
    fine-tune. SR2's line branch emits rare gradient spikes (typical norm 0.04,
    observed 5.6e10 four steps before overflow), and 94 seconds into the run
    every weight was NaN -- which surfaced only as a baffling ``torch.cat():
    expected a non-empty list of Tensors`` when the validation loop's
    finite-check rejected every batch.

    Skipping the step is the standard remedy (it is what AMP's ``GradScaler``
    does on overflow) and is safe: a step whose gradient contains an infinity
    carries no usable direction anyway.
    """
    total = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    ok = bool(torch.isfinite(total))
    return float(total), ok


class EMA:
    """Exponentially-smoothed scalar, for scheduling and checkpoint selection.

    Validation loss is noisy enough that "best epoch" on the raw value often
    picks a lucky draw. Smoothing first makes the selection reflect a trend.
    """

    def __init__(self, alpha: float = 0.9) -> None:
        self.alpha = float(alpha)
        self.value: float | None = None

    def update(self, x: float) -> float:
        x = float(x)
        self.value = x if self.value is None else self.alpha * self.value + (1.0 - self.alpha) * x
        return self.value
