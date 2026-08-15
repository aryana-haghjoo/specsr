"""Building blocks shared across the SR1, ZHead and SR2 architectures.

Attribute names and numerical behaviour here are load-bearing: published
checkpoints are keyed on them, so changing a submodule name breaks weight
loading. Keep signatures stable and add new behaviour behind defaults.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ResidualBlock1D",
    "get_activation",
    "build_param_groups",
    "smooth1d",
    "highpass",
    "largest_divisor_at_most",
    "odd_kernel",
]


def largest_divisor_at_most(channels: int, groups: int = 8) -> int:
    """Largest ``g <= groups`` that divides ``channels`` exactly.

    ``GroupNorm`` requires the channel count to be divisible by the group count.
    Hyperparameter sweeps produce channel widths that are not multiples of 8
    (e.g. 108), so the group count is chosen adaptively rather than fixed: 108
    resolves to 6 groups (108/6 = 18) while 96 keeps 8 (96/8 = 12).
    """
    for i in range(groups, 0, -1):
        if channels % i == 0:
            return i
    return 1


class ResidualBlock1D(nn.Module):
    """Pre-activation 1D residual block with a scaled residual branch.

    The residual is scaled by ``alpha`` rather than added at unit weight, which
    keeps activations well conditioned when many blocks are stacked.
    """

    def __init__(self, channels: int, groups: int = 8, alpha: float = 0.2, p_drop: float = 0.1):
        super().__init__()
        self.alpha = alpha
        g = largest_divisor_at_most(channels, groups)

        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=g, num_channels=channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=g, num_channels=channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.Dropout(p_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.block(x)


def get_activation(name: str) -> nn.Module:
    """Resolve an activation by name.

    ``elu`` is the Exponential Linear Unit: ``x`` for ``x > 0`` and
    ``alpha * (exp(x) - 1)`` otherwise. Unlike ReLU it has non-zero gradient for
    negative inputs, which avoids dead units, and its output mean sits closer to
    zero.
    """
    key = name.lower()
    activations = {
        "relu": lambda: nn.ReLU(),
        "leaky_relu": lambda: nn.LeakyReLU(0.1),
        "gelu": lambda: nn.GELU(),
        "elu": lambda: nn.ELU(alpha=1.0),
    }
    if key not in activations:
        raise ValueError(
            f"Unsupported activation: {name!r}. Choose from {sorted(activations)}."
        )
    return activations[key]()


def build_param_groups(model: nn.Module, lr: float, weight_decay: float) -> list[dict]:
    """Split parameters into decayed and non-decayed groups.

    Weight decay is applied only to multi-dimensional weight tensors. Biases and
    normalisation parameters are excluded, which is standard practice: decaying
    them tends to hurt without regularising anything meaningful.
    """
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_norm = ("norm" in n.lower()) or ("groupnorm" in n.lower())
        if (p.ndim >= 2) and ("weight" in n) and (not is_norm):
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
    ]


def odd_kernel(k: int, length: int) -> int:
    """Clamp ``k`` to an odd kernel size that fits in a sequence of ``length``.

    Reflect-padding by ``k // 2`` on each side and pooling with stride 1 returns
    the input length only when ``k`` is odd. The original code had two
    inconsistent versions of this clamp — one rounding even ``k`` *down*
    (``k -= 1``) and one rounding *up* (``k | 1``) — and the rounding-up variant
    could leave ``k`` even after being clipped to the sequence length, silently
    returning a sequence one sample longer than the input.

    Both agree for odd ``k``, which is what every shipped config uses, so this
    did not affect published results. It is fixed here so it cannot.
    """
    k = int(k)
    limit = max(1, length - 1)
    k = min(k, limit)
    if k % 2 == 0:
        k -= 1
    return max(3, k)


def smooth1d(x: torch.Tensor, k: int = 31) -> torch.Tensor:
    """Moving-average smooth along the last axis with reflect padding.

    ``k`` is forced odd and clipped to the sequence length, so the output always
    has the same length as the input.
    """
    k = odd_kernel(k, x.shape[-1])
    return F.avg_pool1d(F.pad(x, (k // 2, k // 2), mode="reflect"), k, stride=1)


def highpass(x: torch.Tensor, k: int = 51) -> torch.Tensor:
    """Remove the smooth component, leaving small-scale structure."""
    return x - smooth1d(x, k=k)
