"""ZHead — redshift inference from a super-resolved spectrum (stage 2 of 3).

Consumes the SR1 output and predicts a continuous redshift with an associated
uncertainty. In the full pipeline the predicted redshift conditions the SR2 line
prior, so its role is structural, not merely diagnostic: it tells SR2 *where* in
wavelength the emission lines should be.

The same architecture is also trained separately on low-resolution, super-
resolved and high-resolution inputs. Comparing those three runs isolates how
much redshift-relevant information each representation carries, holding
architecture, loss and split fixed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ZHead1D", "heteroscedastic_nll", "redshift_pdf_loss"]


class ZHead1D(nn.Module):
    """Continuous redshift head with uncertainty-aware pooling.

    Parameters
    ----------
    in_channels
        1 to consume flux alone, 2 to additionally consume the SR1 predictive
        log-sigma. With 2 channels the network routes flux and uncertainty
        through separate paths and pools by confidence.
    hidden_dim
        Width of the flux path. The uncertainty path uses ``hidden_dim // 2``,
        since it carries less information.
    num_blocks
        Number of convolutional blocks per path.
    dropout
        Dropout probability inside each block.

    coord_channel
        Append a normalised wavelength coordinate (``linspace(-1, 1, L)``) as an
        extra input channel. **v2 only.**
    dilation_growth
        Dilation of block ``i`` is ``dilation_growth ** i``. At 1 (default) this
        is the v1 architecture; at 2 with six blocks the receptive field grows
        from 37 to ~380 pixels. **v2 only.**
    pooling
        ``"mean"`` selects the original (v1) architecture below. ``"attn"``
        selects the v2 architecture: a single trunk over all input channels
        followed by attention pooling over wavelength.

    Returns
    -------
    mu, log_var
        Both shaped ``(B,)``.

    Notes
    -----
    **v1 (``pooling="mean"``) is redshift-blind by construction and is kept only
    so historical checkpoints load.** Its trunk is translation-equivariant and
    its global mean pool is translation-invariant, so a spectrum whose lines are
    shifted -- which is what a different redshift *is* on a fixed log-wavelength
    grid -- produces near-identical pooled features. The three-arm comparison
    run on it produced the impossible ordering hires < lowres, because the only
    signal it could use was shift-invariant statistics (line strengths,
    continuum shape), which the prism carries at higher per-pixel S/N.

    v2 (``pooling="attn"``) fixes this three ways: a coordinate channel tags
    every feature with its position on the grid, dilated convolutions widen the
    receptive field so multi-line patterns are co-detected, and attention
    pooling forms a position-weighted summary instead of a position-destroying
    average. When the uncertainty channel is present it is treated as one more
    input channel of the same trunk, so all comparison arms share one structure
    and differ only in channel count.

    In v1, when the uncertainty channel is present, the pooling over wavelength
    is weighted by predicted confidence rather than being a plain mean:
    confidence is derived as inverse sigma, smoothed by a wide convolution, then
    normalised to sum to one across the spectral axis.
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        dropout: float = 0.1,
        coord_channel: bool = False,
        dilation_growth: int = 1,
        pooling: str = "mean",
        head: str = "gaussian",
        n_z_bins: int = 1024,
        z_grid_min_n: float | None = None,
        z_grid_max_n: float | None = None,
        soft_argmax_half: int = 8,
    ):
        super().__init__()

        if pooling not in ("mean", "attn"):
            raise ValueError(f"pooling must be 'mean' or 'attn', got {pooling!r}")
        if head not in ("gaussian", "softmax"):
            raise ValueError(f"head must be 'gaussian' or 'softmax', got {head!r}")
        if pooling == "mean" and (coord_channel or dilation_growth != 1):
            raise ValueError(
                "coord_channel/dilation_growth are v2 options; they require pooling='attn'"
            )

        self.in_channels = in_channels
        self.coord_channel = coord_channel
        self.pooling = pooling
        self.v2 = pooling == "attn"
        self.use_uncertainty = (not self.v2) and in_channels == 2

        if self.v2:
            trunk_in = in_channels + (1 if coord_channel else 0)
            self.flux_net = self._make_conv_blocks(
                trunk_in, hidden_dim, num_blocks, dropout, dilation_growth
            )
            # Attention pooling: a learned per-position score, softmaxed over
            # wavelength. Because the features are position-tagged by the
            # coordinate channel, the weighted sum can express *where* the
            # attended structure sits -- which a mean cannot. The coordinate
            # itself is additionally pooled under the same weights, handing the
            # heads an explicit "expected position of the attended structure"
            # scalar: the most direct possible readout of line location.
            self.attn_score = nn.Conv1d(hidden_dim, 1, kernel_size=1)
            combined_dim = hidden_dim + (1 if coord_channel else 0)
        elif self.use_uncertainty:
            self.flux_net = self._make_conv_blocks(1, hidden_dim, num_blocks, dropout)
            # Lighter network: the uncertainty channel is smoother and carries
            # less structure than the flux.
            self.sigma_net = self._make_conv_blocks(1, hidden_dim // 2, num_blocks, dropout)
            # Wide kernel so confidence varies on line-group scales, not per-pixel.
            self.confidence_conv = nn.Conv1d(1, 1, kernel_size=15, padding=7)
            combined_dim = hidden_dim + hidden_dim // 2
        else:
            self.flux_net = self._make_conv_blocks(in_channels, hidden_dim, num_blocks, dropout)
            combined_dim = hidden_dim

        self.head = head
        if head == "softmax":
            if z_grid_min_n is None or z_grid_max_n is None:
                raise ValueError(
                    "head='softmax' needs z_grid_min_n/z_grid_max_n (normalised redshift "
                    "bounds). They are stored in the checkpoint so inference cannot use a "
                    "different grid from training."
                )
            self.n_z_bins = int(n_z_bins)
            self.soft_argmax_half = int(soft_argmax_half)
            self.z_logits = nn.Linear(combined_dim, self.n_z_bins)
            # The grid lives in the module (a buffer, so it rides along in the
            # state dict) rather than in the caller. A PDF is meaningless without
            # the axis it is over, and the axis is exactly the sort of thing that
            # silently drifts between training and inference.
            self.register_buffer(
                "z_grid_n",
                torch.linspace(float(z_grid_min_n), float(z_grid_max_n), self.n_z_bins),
            )
        else:
            self.mu = nn.Linear(combined_dim, 1)
            self.log_var = nn.Linear(combined_dim, 1)
            # Start with small predicted uncertainty, for the same reason as SR1.
            nn.init.constant_(self.log_var.weight, 0.0)
            nn.init.constant_(self.log_var.bias, -2.0)

    @property
    def bounded_mean(self) -> bool:
        """Whether ``forward``'s first output still needs ``decode_mean``.

        The Gaussian head emits an unbounded scalar that the transform squashes
        onto the observed redshift range. The classification head's estimate is
        already *in* normalised redshift units -- it comes from a grid that
        cannot leave the range by construction -- so squashing it again would
        be wrong. Callers branch on this rather than on the head name.
        """
        return self.head != "softmax"

    @staticmethod
    def _make_conv_blocks(
        in_ch: int, out_ch: int, num_blocks: int, dropout: float, dilation_growth: int = 1
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        c = in_ch
        for i in range(num_blocks):
            dil = int(dilation_growth) ** i
            layers += [
                nn.Conv1d(c, out_ch, kernel_size=7, padding=3 * dil, dilation=dil, bias=True),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            c = out_ch
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(mu, log_var)``, both ``(B,)``.

        For the classification head these are derived from the predicted PDF, so
        every existing consumer keeps working unchanged; train against
        :meth:`logits` instead, which is what the objective needs.
        """
        if self.head == "softmax":
            return self.moments(self.logits(x))
        return self._gaussian_forward(x)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Unnormalised log-PDF over the redshift grid, ``(B, n_z_bins)``."""
        if self.head != "softmax":
            raise RuntimeError("logits() is only defined for head='softmax'")
        return self.z_logits(self._pool(x))

    def moments(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """PDF -> ``(mu, log_var)`` in normalised redshift units.

        The point estimate is a *local* expectation in a window around the
        modal bin, not the global mean. That distinction is the whole reason
        for a classification head: redshift PDFs are genuinely multimodal
        (an emission line misidentified as a different transition puts a second
        peak elsewhere), and the global mean of a bimodal PDF lands in the
        empty valley between the two modes -- which is precisely the
        catastrophic-outlier failure this head exists to remove. Taking a
        soft-argmax keeps the estimate on a mode while staying differentiable,
        so the redshift coupling in the SR2 loss can still backpropagate.

        ``log_var`` is computed from the *full* PDF, so it stays large when the
        head is genuinely torn between two redshifts -- which is the signal
        downstream stages need in order to distrust it.
        """
        pdf = torch.softmax(logits, dim=-1)
        grid = self.z_grid_n.to(pdf.dtype)

        mode = pdf.argmax(dim=-1)
        h = self.soft_argmax_half
        offs = torch.arange(-h, h + 1, device=pdf.device)
        idx = (mode[:, None] + offs[None, :]).clamp(0, self.n_z_bins - 1)
        w = pdf.gather(1, idx)
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        mu = (w * grid[idx]).sum(dim=-1)

        var = (pdf * (grid[None, :] - mu[:, None]) ** 2).sum(dim=-1)
        return mu, torch.log(var.clamp_min(1e-12))

    def _gaussian_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._pool(x)
        return self.mu(h).squeeze(-1), self.log_var(h).squeeze(-1)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        _, C, L = x.shape

        if self.v2:
            coord = None
            if self.coord_channel:
                coord = torch.linspace(-1.0, 1.0, L, device=x.device, dtype=x.dtype)
                coord = coord.expand(x.shape[0], 1, L)
                x = torch.cat([x, coord], dim=1)
            h = self.flux_net(x)
            w = torch.softmax(self.attn_score(h), dim=-1)
            h = (h * w).sum(dim=-1)
            if coord is not None:
                h = torch.cat([h, (coord * w).sum(dim=-1)], dim=1)
        elif self.use_uncertainty and C == 2:
            flux = x[:, 0:1, :]
            log_sigma = x[:, 1:2, :]

            # Inverse-variance style weighting: low predicted sigma -> high weight.
            sigma = torch.exp(log_sigma).clamp(min=1e-3)
            confidence = 1.0 / (sigma + 1e-6)
            confidence = torch.sigmoid(self.confidence_conv(confidence))
            confidence = confidence / (confidence.sum(dim=-1, keepdim=True) + 1e-6)

            h_flux = self.flux_net(flux)
            h_sigma = self.sigma_net(log_sigma)
            h = torch.cat([h_flux, h_sigma], dim=1)

            # Confidence-weighted pooling over wavelength.
            h = (h * confidence).sum(dim=-1)
        else:
            h = self.flux_net(x)
            h = h.mean(dim=-1)

        return h


def heteroscedastic_nll(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    y: torch.Tensor,
    var_floor: float = 1e-6,
) -> torch.Tensor:
    """Gaussian negative log-likelihood with a predicted per-sample variance.

    All arguments are shaped ``(B,)``. The variance floor prevents the loss from
    diverging when the model becomes overconfident on easy examples.
    """
    var = torch.exp(log_var).clamp_min(var_floor)
    return 0.5 * (torch.log(var) + (y - mu) ** 2 / var).mean()


def redshift_pdf_loss(
    logits: torch.Tensor,
    z_true_n: torch.Tensor,
    z_grid_n: torch.Tensor,
    target_sigma_bins: float = 1.5,
) -> torch.Tensor:
    """Cross-entropy of a predicted redshift PDF against a soft target.

    ``logits`` is ``(B, n_bins)``, ``z_true_n`` is ``(B,)`` in normalised
    redshift units, ``z_grid_n`` is the bin centres.

    The target is a narrow Gaussian on the grid rather than a one-hot bin. With
    ~10^3 bins the true redshift almost never sits exactly on a centre, and
    one-hot targets make the objective discontinuous in the true value while
    telling the model that the neighbouring bin -- which may be a small fraction
    of a resolution element away -- is exactly as wrong as the far end of the
    grid. A soft target keeps the loss smooth and encodes that near misses are
    near.

    Unlike a Gaussian likelihood this objective is indifferent to how many modes
    the posterior has, which is the point: it can represent "either z=2.1 or
    z=5.3" honestly instead of splitting the difference.
    """
    z_true_n = z_true_n.reshape(-1, 1)
    grid = z_grid_n.to(logits.dtype)[None, :]
    step = (grid[0, 1] - grid[0, 0]).abs().clamp_min(1e-12)
    sigma = float(target_sigma_bins) * step

    target = torch.exp(-0.5 * ((grid - z_true_n) / sigma) ** 2)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return -(target * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
