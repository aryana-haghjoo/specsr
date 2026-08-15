"""Input representations for the redshift head.

The redshift experiment trains *the same* :class:`~specsr.models.zhead.ZHead1D`
architecture on four different spectral representations, so that any difference
in redshift accuracy is attributable to the information content of the input
rather than to model capacity:

==========  =====================================  ========
``name``    pipeline                               channels
==========  =====================================  ========
``lowres``  ``flux_low`` (prism) directly          1
``hires``   ``flux_high`` (grating) directly       1
``sr1``     LR -> SR1 -> (mean, log sigma)         2
``sr2``     LR -> SR1 -> SR2 -> (mean, log sigma)  2
==========  =====================================  ========

Previously these lived in four near-duplicate copies of the training script
(``redshift_head``, ``_hires``, ``_lowres``, ``_sr2``) which had drifted apart by
hundreds of lines, making it genuinely hard to confirm that the architecture and
training procedure were held fixed across the comparison. Collapsing them to one
script plus a swappable source makes that guarantee structural: there is only one
training loop, one loss and one split.

``hires`` is the upper bound of the comparison and ``lowres`` the lower bound;
``sr1``/``sr2`` are the quantities of interest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .ztransform import RedshiftTransform

__all__ = [
    "ZHeadSource",
    "RawFluxSource",
    "SR1Source",
    "SR2Source",
    "SOURCE_NAMES",
    "build_source",
]

SOURCE_NAMES = ("lowres", "hires", "sr1", "sr2")


class ZHeadSource(Protocol):
    """Produces the ``(B, C, L)`` tensor fed to the redshift head."""

    name: str
    n_channels: int

    def __call__(self, batch: dict, device: torch.device) -> torch.Tensor: ...


@dataclass
class RawFluxSource:
    """Feed an observed spectrum straight to the head, with no model in between.

    Used for the ``lowres`` and ``hires`` ends of the comparison.
    """

    key: str
    name: str
    n_channels: int = 1

    def __call__(self, batch: dict, device: torch.device) -> torch.Tensor:
        x = batch[self.key].to(device)
        if x.ndim == 2:  # (B, L) -> (B, 1, L)
            x = x.unsqueeze(1)
        return x


@dataclass
class SR1Source:
    """Run the frozen SR1 backbone and hand its output to the head.

    When ``use_sigma`` is set, the predictive log-sigma is supplied as a second
    channel so the head can down-weight wavelengths SR1 is unsure about. Note the
    conversion: SR1 emits ``log_var``, and the head consumes ``log_sigma =
    0.5 * log_var``.
    """

    sr1: torch.nn.Module
    use_sigma: bool = True
    name: str = "sr1"

    @property
    def n_channels(self) -> int:
        return 2 if self.use_sigma else 1

    @torch.no_grad()
    def __call__(self, batch: dict, device: torch.device) -> torch.Tensor:
        x_low = batch["flux_low"].to(device)
        if x_low.ndim == 2:
            x_low = x_low.unsqueeze(1)
        mean, log_var = self.sr1(x_low)
        if not self.use_sigma:
            return mean
        return torch.cat([mean, 0.5 * log_var], dim=1)


@dataclass
class SR2Source:
    """Run the frozen SR1 -> ZHead -> SR2 chain and hand SR2's output to the head.

    SR2 is conditioned on a redshift, so this needs a *bootstrap* redshift head
    (the one trained on the SR1 representation) to supply it. That head is frozen
    and is not the head being trained here — the model under evaluation only ever
    sees the final SR2 spectrum.

    The SR2 input stack is built by
    :func:`specsr.models.sr2.build_sr2_input`, whose channel order and flags must
    match what the SR2 checkpoint was trained with. Take those flags from the
    checkpoint's stored config rather than assuming defaults.
    """

    sr1: torch.nn.Module
    zhead_bootstrap: torch.nn.Module
    sr2: torch.nn.Module
    ztransform: RedshiftTransform
    delta_cap: float = 0.0
    use_sigma: bool = True
    use_sr1_sigma: bool = True
    use_line_mask: bool = True
    use_zhat_channel: bool = True
    use_zsigma_channel: bool = False
    zsigma_line_mask: bool = False
    zsigma_mask_max_um: float = 0.05
    sigma_base_um: float = 0.005
    name: str = "sr2"

    @property
    def n_channels(self) -> int:
        return 2 if self.use_sigma else 1

    @torch.no_grad()
    def __call__(self, batch: dict, device: torch.device) -> torch.Tensor:
        from ..models.sr2 import build_sr2_input, constrain_delta

        x_low = batch["flux_low"].to(device)
        if x_low.ndim == 2:
            x_low = x_low.unsqueeze(1)

        sr1_mean, sr1_log_var = self.sr1(x_low)
        sr1_log_sigma = 0.5 * sr1_log_var

        # Bootstrap redshift, decoded through the same transform training uses.
        mu_raw, log_var = self.zhead_bootstrap(torch.cat([sr1_mean, sr1_log_sigma], dim=1))
        zhat, z_sigma = self.ztransform.predict(
            mu_raw, log_var, bounded=self.zhead_bootstrap.bounded_mean)

        x_in = build_sr2_input(
            x_low=x_low,
            sr1_mean=sr1_mean,
            sr1_log_sigma=sr1_log_sigma,
            zhat=zhat,
            wave_hi_um=self.sr2.wave_hi_um,
            line_rest_um=self.sr2.line_rest_um,
            use_sr1_sigma=self.use_sr1_sigma,
            use_line_mask=self.use_line_mask,
            use_zhat_channel=self.use_zhat_channel,
            use_zsigma_channel=self.use_zsigma_channel,
            sigma_base_um=self.sigma_base_um,
            z_sigma=z_sigma if (self.zsigma_line_mask or self.use_zsigma_channel) else None,
            sigma_max_um=self.zsigma_mask_max_um,
        )
        delta, sr2_log_var = self.sr2(x_in, zhat)
        if self.delta_cap > 0:
            delta = constrain_delta(delta, self.delta_cap)

        sr2_mean = sr1_mean + delta
        if not self.use_sigma:
            return sr2_mean
        return torch.cat([sr2_mean, 0.5 * sr2_log_var], dim=1)


def build_source(name: str, **kwargs) -> ZHeadSource:
    """Construct a source by name. See :data:`SOURCE_NAMES`."""
    if name == "lowres":
        return RawFluxSource(key="flux_low", name="lowres")
    if name == "hires":
        return RawFluxSource(key="flux_high", name="hires")
    if name == "sr1":
        return SR1Source(sr1=kwargs["sr1"], use_sigma=kwargs.get("use_sigma", True))
    if name == "sr2":
        return SR2Source(
            sr1=kwargs["sr1"],
            zhead_bootstrap=kwargs["zhead_bootstrap"],
            sr2=kwargs["sr2"],
            ztransform=kwargs["ztransform"],
            delta_cap=kwargs.get("delta_cap", 0.0),
            use_sigma=kwargs.get("use_sigma", True),
            use_sr1_sigma=kwargs.get("use_sr1_sigma", True),
            use_line_mask=kwargs.get("use_line_mask", True),
            use_zhat_channel=kwargs.get("use_zhat_channel", True),
            # These three were being dropped. The caller reads them off the SR2
            # checkpoint's own config precisely so the arm reproduces the input
            # SR2 was trained against; defaulting them here instead fed SR2 an
            # unwidened line mask, degrading the one arm the comparison exists
            # to measure -- and silently, because the channel count is unchanged
            # when `use_zsigma_channel` is False.
            use_zsigma_channel=kwargs.get("use_zsigma_channel", False),
            zsigma_line_mask=kwargs.get("zsigma_line_mask", False),
            zsigma_mask_max_um=kwargs.get("zsigma_mask_max_um", 0.05),
            sigma_base_um=kwargs.get("sigma_base_um", 0.005),
        )
    raise ValueError(f"Unknown ZHead source {name!r}. Choose from {SOURCE_NAMES}.")
