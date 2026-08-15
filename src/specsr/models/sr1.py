"""SR1 — the super-resolution backbone (stage 1 of 3).

Maps a low-resolution prism spectrum, already interpolated onto the common
high-resolution wavelength grid, to a super-resolved estimate with a
wavelength-dependent predictive uncertainty.

SR1 is deliberately conservative: it recovers broad structure and begins to
sharpen lines, but is not asked to synthesise fine detail on its own. That is
left to :mod:`specsr.models.sr2`, which refines the SR1 output using an
explicit emission-line prior conditioned on an inferred redshift.

.. note::

   SR1 is *not* the same operation as the interpolation applied during
   preprocessing. Preprocessing resamples the prism spectrum onto the target
   wavelength grid, changing the sampling but not the information content. SR1
   is trained against the medium-resolution reference and learns to redistribute
   flux into narrower features, which changes the effective resolution. The
   interpolation step supplies SR1's input grid; it does not do SR1's job.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ResidualBlock1D

__all__ = ["SuperRes1D"]


class SuperRes1D(nn.Module):
    """1D ResNet-style super-resolution backbone with heteroscedastic output.

    Parameters
    ----------
    in_channels
        Number of input channels. 1 for flux alone.
    hidden_dim
        Width of the residual trunk.
    num_res_blocks
        Number of :class:`~specsr.models.blocks.ResidualBlock1D` blocks.
    dropout
        Dropout probability inside each residual block.
    activation_fn
        Activation applied after the input convolution.

    Returns
    -------
    mean, log_var
        Both shaped ``(B, 1, L)``. ``log_var`` is the log of the *model*
        variance; the measurement variance of the reference spectrum is added
        separately in the loss, so the two noise sources stay distinguishable.

    Notes
    -----
    The log-variance head is initialised to a constant ``-2.0`` bias with zero
    weights, so the model starts by predicting a small, uniform uncertainty
    (sigma ~ 0.37 in normalised units). Starting from a data-dependent variance
    lets the network drive the likelihood down by inflating uncertainty before it
    has learned anything, which stalls training.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 96,
        num_res_blocks: int = 12,
        dropout: float = 0.02,
        activation_fn: nn.Module | None = None,
    ):
        super().__init__()
        if activation_fn is None:
            activation_fn = nn.GELU()

        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2, bias=True),
            activation_fn,
        )
        self.resblocks = nn.Sequential(
            *[ResidualBlock1D(hidden_dim, p_drop=dropout) for _ in range(num_res_blocks)]
        )
        self.mean_head = nn.Conv1d(hidden_dim, 1, kernel_size=1, bias=True)
        self.log_var_head = nn.Conv1d(hidden_dim, 1, kernel_size=1, bias=True)

        nn.init.constant_(self.log_var_head.weight, 0.0)
        nn.init.constant_(self.log_var_head.bias, -2.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.initial(x)
        x = self.resblocks(x)
        mean = self.mean_head(x)
        log_var = self.log_var_head(x)
        return mean, log_var
