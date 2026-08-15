"""Load a trained stage from its checkpoint.

These are the functions a user of the installed package needs most: given a
``.pth`` written by one of the training loops, hand back a frozen, eval-mode
module ready for inference. They belong in the package rather than beside a
training script: anything :mod:`specsr.evaluation` reaches for outside
``src/specsr`` is absent from an installed wheel, so it works from a checkout
and raises ``ImportError`` for everybody else.

Everything here reads the *stored* config out of the checkpoint rather than
taking architecture on faith from a YAML file. A checkpoint whose architecture
is inferred from a config that has since been edited loads with silently
mismatched layers, and ``strict=False`` will happily let that pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from .sr1 import SuperRes1D
from .sr2 import SR2Attention
from .zhead import ZHead1D

__all__ = [
    "load_yaml_config",
    "get_activation",
    "load_sr1",
    "load_zhead",
    "load_sr2",
    "freeze",
]

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "gelu": nn.GELU,
    "elu": nn.ELU,
}


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Read a stage config, tolerating Weights & Biases run dumps.

    Most configs in ``configs/`` are exported W&B run configs rather than
    hand-written YAML, so every value is wrapped as ``{"value": ...}`` and the
    file carries a ``_wandb`` bookkeeping block. Unwrap the values and drop the
    underscore keys, so both shapes load identically.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return {
        k: (v["value"] if isinstance(v, dict) and "value" in v else v)
        for k, v in cfg.items()
        if not k.startswith("_")
    }


def get_activation(name: str) -> nn.Module:
    """Activation module by name, as spelled in the stage configs."""
    key = str(name).lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation {name!r}. Choose from {sorted(_ACTIVATIONS)}.")
    return _ACTIVATIONS[key]()


def freeze(module: nn.Module) -> nn.Module:
    """Put a module in eval mode and stop gradients through it.

    Both halves matter. ``eval()`` alone leaves ``requires_grad`` set, so an
    upstream stage still accumulates gradients and quietly drifts during the
    next stage's training; ``requires_grad_(False)`` alone leaves dropout and
    batch-norm in training mode, so a frozen stage returns a different spectrum
    on every call.
    """
    module.eval()
    for p in module.parameters():
        p.requires_grad = False
    return module


def load_sr1(
    sr1_config_path: str | Path,
    sr1_ckpt_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[SuperRes1D, dict[str, Any]]:
    """Load the frozen SR1 backbone, returning ``(model, config)``."""
    cfg = load_yaml_config(sr1_config_path)
    sr1 = SuperRes1D(
        in_channels=1,
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        num_res_blocks=int(cfg.get("num_res_blocks", 12)),
        dropout=float(cfg.get("dropout", 0.02)),
        activation_fn=get_activation(cfg.get("activation", "gelu")),
    ).to(device)
    sr1.load_state_dict(torch.load(sr1_ckpt_path, map_location="cpu"))
    return freeze(sr1), cfg


def load_zhead(
    zhead_ckpt_path: str | Path,
    device: torch.device | str = "cpu",
    unfreeze_last_n: int = 0,
) -> tuple[ZHead1D, float, float, bool, dict[str, Any]]:
    """Load the redshift head.

    Returns ``(model, z_mean, z_std, use_sigma_channel, config)``. The two
    moments are the normalisation the head was trained under and are stored in
    the checkpoint; predicting with any other pair yields plausible-looking
    redshifts that are wrong by a constant factor.

    ``unfreeze_last_n`` reopens the final *n* parameterised leaf modules for
    fine-tuning during SR2 training. It defaults to fully frozen.
    """
    ck = torch.load(zhead_ckpt_path, map_location="cpu", weights_only=False)
    zcfg = ck.get("config", {})
    use_sigma = bool(ck.get("use_sigma_channel", False))
    zhead = ZHead1D(
        in_channels=(2 if use_sigma else 1),
        hidden_dim=int(zcfg.get("hidden_dim", 64)),
        num_blocks=int(zcfg.get("num_blocks", 4)),
        dropout=float(zcfg.get("dropout", 0.1)),
        # v2 keys are absent from historical checkpoints; these defaults
        # reconstruct the v1 architecture exactly.
        coord_channel=bool(zcfg.get("coord_channel", False)),
        dilation_growth=int(zcfg.get("dilation_growth", 1)),
        pooling=str(zcfg.get("pooling", "mean")),
        head=str(zcfg.get("head", "gaussian")),
        n_z_bins=int(zcfg.get("n_z_bins", 1024)),
        # The grid bounds are the transform's own normalised bounds, stored
        # alongside the weights; a classification head cannot be rebuilt without
        # the axis its PDF is over.
        z_grid_min_n=(float(ck["z_min_n"]) if zcfg.get("head") == "softmax" else None),
        z_grid_max_n=(float(ck["z_max_n"]) if zcfg.get("head") == "softmax" else None),
        soft_argmax_half=int(zcfg.get("soft_argmax_half", 8)),
    ).to(device)
    zhead.load_state_dict(ck["zhead_state_dict"])
    freeze(zhead)

    if unfreeze_last_n > 0:
        leaves = [
            (n, list(m.parameters(recurse=False)))
            for n, m in zhead.named_modules()
            if n and list(m.parameters(recurse=False))
        ]
        for name, params in leaves[-min(unfreeze_last_n, len(leaves)):]:
            for p in params:
                p.requires_grad = True
            print(f"  ZHead unfrozen: {name} ({sum(p.numel() for p in params):,} params)")

    return zhead, float(ck["z_mean"]), float(ck["z_std"]), use_sigma, zcfg


def load_sr2(
    sr2_ckpt_path: str | Path,
    wave_hi_um,
    line_rest_um,
    device: torch.device | str = "cpu",
) -> tuple[SR2Attention, dict[str, Any]]:
    """Load the SR2 refiner, returning ``(model, config)``.

    The input channel count is *derived* from the flags the checkpoint was
    trained with, never assumed: :func:`specsr.models.sr2.build_sr2_input`
    appends channels conditionally, so a model built with different flags loads
    a mismatched first layer.
    """
    import numpy as np

    from .sr2 import sr2_input_channels

    ck = torch.load(sr2_ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    sr2 = SR2Attention(
        in_channels=sr2_input_channels(
            use_sr1_sigma=bool(cfg.get("use_sr1_sigma", True)),
            use_line_mask=bool(cfg.get("use_line_mask", True)),
            use_zhat_channel=bool(cfg.get("use_zhat_channel", True)),
            # Absent from checkpoints predating the sigma_z channel, which were
            # trained without it.
            use_zsigma_channel=bool(cfg.get("use_zsigma_channel", False)),
        ),
        line_rest_um=np.asarray(line_rest_um, dtype=np.float32),
        wave_hi_um=np.asarray(wave_hi_um, dtype=np.float32),
        line_dim=int(cfg.get("line_dim", 64)),
        num_attn_heads=int(cfg.get("num_attn_heads", 4)),
        num_attn_layers=int(cfg.get("num_attn_layers", 2)),
        window_half=int(cfg.get("window_half", 24)),
        cnn_dim=int(cfg.get("cnn_dim", 64)),
        num_cnn_blocks=int(cfg.get("num_cnn_blocks", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        # Older checkpoints predate the flag and were trained at full CNN
        # weight, so absent means 1.0 -- not 0.0.
        cnn_scale=float(cfg.get("cnn_scale", 1.0) or 1.0),
    ).to(device)
    missing, unexpected = sr2.load_state_dict(ck["sr2_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"SR2 checkpoint does not match the packaged architecture: "
            f"missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}"
        )
    return freeze(sr2), cfg
