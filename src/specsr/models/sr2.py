"""SR2 — physics-informed residual refiner (stage 3 of 3).

Predicts a correction ``delta`` to the SR1 output, as the sum of two branches:

**Line branch.** For each of ``K`` catalogued rest-frame features, the observed
position is computed from the redshift predicted by ZHead. A window of the input
is extracted around each position and encoded into a token; multi-head
self-attention is applied *across line tokens*, so the model can exploit
physical relationships between lines (fixed doublet ratios, Balmer decrements,
lines that co-occur in a given ionisation state). Each token then emits Gaussian
parameters — amplitude, width, sub-pixel offset — gated by a learned presence
probability, and the Gaussians are rendered back onto the wavelength grid.

**CNN branch.** A plain residual CNN handles smooth continuum corrections that
the parametric line model cannot express.

The parametric line branch is what makes this "physics-informed": the model
cannot place a sharp feature at an arbitrary wavelength, only at a catalogued
line position implied by the redshift. That constraint is the main defence
against the failure mode of end-to-end super-resolution models, which tend to
either over-smooth real features or hallucinate sharp ones.

.. warning::

   Attribute names here are load-bearing. Published checkpoints key on
   ``line_rest_um`` and ``wave_hi_um`` as registered buffers; an earlier
   iteration used ``line_rest_um_buf``/``wave_hi_um_buf`` and its weights will
   not load against this class without remapping.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "SR2Attention",
    "constrain_delta",
    "gaussian_line_mask",
    "build_line_mask",
    "build_sr2_input",
    "sr2_input_channels",
]


def constrain_delta(delta: torch.Tensor, cap: float) -> torch.Tensor:
    """Soft-limit the residual magnitude to ``+/- cap``.

    A ``tanh`` squash rather than a hard clamp, so gradients keep flowing when
    the correction saturates. Capping matters because SR2 is free to add flux
    anywhere; without a bound it can manufacture arbitrarily large features.
    """
    return torch.tanh(delta / cap) * cap if cap > 0 else delta


def gaussian_line_mask(
    wave_um: torch.Tensor, centers_um: torch.Tensor, sigma_um
) -> torch.Tensor:
    """Union of Gaussians centred on ``centers_um``, shaped ``(1, 1, L)``.

    ``sigma_um`` is a scalar, or a per-line tensor broadcastable to
    ``centers_um`` -- the latter is what lets a mask widen line-by-line with the
    redshift uncertainty, since a given redshift error displaces a red line
    further in wavelength than a blue one.
    """
    d2 = (wave_um[None, :] - centers_um[:, None]) ** 2
    if torch.is_tensor(sigma_um):
        var = (sigma_um.reshape(-1, 1) ** 2) + 1e-12
    else:
        var = sigma_um**2 + 1e-12
    return torch.exp(-0.5 * d2 / var).sum(0).clamp(0, 1)[None, None, :]


def build_line_mask(
    wave_hi_um: torch.Tensor,
    zhat: torch.Tensor,
    line_rest_um,
    sigma_base_um: float = 0.005,
    z_sigma: torch.Tensor | None = None,
    sigma_max_um: float = 0.05,
) -> torch.Tensor:
    """Per-sample soft mask marking where emission lines are expected.

    Used both as an SR2 input channel and by the loss to weight line regions
    differently from continuum.

    ``z_sigma`` widens the mask by the redshift uncertainty. A line of rest
    wavelength ``lam0`` sits at ``lam0 * (1 + z)``, so an uncertainty
    ``sigma_z`` displaces it by ``lam0 * sigma_z``; that is combined with the
    base width in quadrature. Without it the mask is a fixed 0.005 um -- a
    redshift precision of ``dz ~ 0.01`` -- and is simply *wrong* whenever the
    head is less certain than that, pointing SR2's attention confidently at a
    stretch of continuum. Widening makes an unsure prediction produce a vague
    mask instead of a confident false one.

    ``sigma_max_um`` caps the widening. The same mask weights the in-line term
    of the SR2 loss, so an uncapped mask on a hopeless redshift would approach
    all-ones and quietly turn that term into a plain reconstruction loss.
    """
    device = zhat.device
    wave = wave_hi_um.to(device)
    line_rest = torch.as_tensor(line_rest_um, device=device)
    zhat = zhat.reshape(-1)

    if z_sigma is None:
        widths: list = [sigma_base_um] * zhat.shape[0]
    else:
        zs = z_sigma.reshape(-1).to(device).clamp_min(0.0)
        widths = [
            torch.sqrt(sigma_base_um**2 + (line_rest * s_i) ** 2).clamp_max(sigma_max_um)
            for s_i in zs
        ]

    masks = [
        gaussian_line_mask(wave, line_rest * (1.0 + z_i.clamp_min(0.0)), sigma_um=w)
        for z_i, w in zip(zhat, widths, strict=True)
    ]
    return torch.cat(masks, dim=0)


def sr2_input_channels(
    use_sr1_sigma: bool = True,
    use_line_mask: bool = True,
    use_zhat_channel: bool = True,
    use_zsigma_channel: bool = False,
) -> int:
    """Number of channels :func:`build_sr2_input` will produce."""
    return (2 + int(use_sr1_sigma) + int(use_line_mask) + int(use_zhat_channel)
            + int(use_zsigma_channel))


def build_sr2_input(
    x_low: torch.Tensor,
    sr1_mean: torch.Tensor,
    sr1_log_sigma: torch.Tensor,
    zhat: torch.Tensor,
    wave_hi_um: torch.Tensor,
    line_rest_um,
    *,
    use_sr1_sigma: bool = True,
    use_line_mask: bool = True,
    use_zhat_channel: bool = True,
    use_zsigma_channel: bool = False,
    sigma_base_um: float = 0.005,
    z_sigma: torch.Tensor | None = None,
    sigma_max_um: float = 0.05,
) -> torch.Tensor:
    """Assemble the channel stack SR2 consumes.

    Channel order is fixed and load-bearing — a checkpoint trained on one order
    produces silent nonsense if fed another:

    0. ``x_low``    — the interpolated low-resolution input
    1. ``sr1_mean`` — the SR1 reconstruction
    2. ``sigma``    — ``exp(sr1_log_sigma)``, if ``use_sr1_sigma``
    3. ``line_mask``— soft mask of expected line positions, if ``use_line_mask``
    4. ``zhat``     — the redshift broadcast along wavelength, if ``use_zhat_channel``
    5. ``z_sigma``  — its uncertainty, likewise broadcast, if ``use_zsigma_channel``

    ``z_sigma`` is appended last so that enabling it does not renumber the
    existing channels. It still changes the input width, so a checkpoint
    trained without it cannot be warm-started with it on. Widening the line
    mask via ``z_sigma`` needs no such retraining -- the channel count is
    unchanged -- which is why the two are separate flags.

    .. important::

       Channel 2 is the **linear** predictive sigma, not the log-sigma. SR1
       emits ``log_var``; callers pass ``log_sigma = 0.5 * log_var`` and this
       function exponentiates it. Passing the log by mistake is dimensionally
       plausible and trains to a worse but non-obviously-broken model.

    Giving SR2 both the line mask and the redshift explicitly, rather than
    letting it infer them, is part of what makes the stage physics-informed: the
    conditioning is supplied rather than learned.

    This function exists so the stack is defined exactly once. It was previously
    open-coded in the SR2 training script, the inference script and two redshift
    head variants, where the copies could drift apart silently.
    """
    chans = [x_low, sr1_mean]
    if use_sr1_sigma:
        chans.append(torch.exp(sr1_log_sigma).clamp_min(1e-6))
    if use_line_mask:
        chans.append(build_line_mask(
            wave_hi_um, zhat, line_rest_um, sigma_base_um=sigma_base_um,
            z_sigma=z_sigma, sigma_max_um=sigma_max_um))
    if use_zhat_channel:
        chans.append(zhat[:, None, None].expand(-1, 1, sr1_mean.shape[-1]))
    if use_zsigma_channel:
        if z_sigma is None:
            raise ValueError("use_zsigma_channel=True requires z_sigma")
        chans.append(z_sigma.reshape(-1)[:, None, None].expand(-1, 1, sr1_mean.shape[-1]))
    return torch.cat(chans, dim=1)


class SR2Attention(nn.Module):
    """Two-branch residual refiner: line-token attention plus a residual CNN.

    Parameters
    ----------
    in_channels
        Channels of the input stack handed to SR2 (typically the SR1 mean,
        its log-sigma, and the interpolated low-resolution input).
    line_rest_um
        ``(K,)`` rest-frame line wavelengths in microns. One attention token per
        entry.
    wave_hi_um
        ``(L,)`` observed-frame wavelength grid in microns.
    line_dim
        Token embedding width for the line branch.
    num_attn_heads, num_attn_layers
        Transformer encoder shape for cross-line attention.
    window_half
        Half-width, in pixels, of the window extracted around each line.
    cnn_dim, num_cnn_blocks
        Width and depth of the continuum CNN branch.
    dropout
        Dropout probability in both branches.

    Returns
    -------
    In training mode, ``(delta, logvar, presence, presence_logit)``; in eval
    mode, ``(delta, logvar)``. ``presence`` is exposed during training so the
    loss can supervise it against the lines the reference spectrum actually
    shows — without that signal the model turns every line on weakly rather than
    committing to the ones that are really there.

    Notes
    -----
    Initialisation is deliberately asymmetric between the branches, and it is
    worth being precise about because it is not a clean identity:

    - The **CNN branch** output head is fully zero-initialised (weight *and*
      bias), so it contributes exactly zero at step zero.
    - The **line branch** is *not*. ``amp_head`` has zero bias but random
      weights (``std=0.01``), and the presence gate starts at ``sigmoid(-2) ~
      0.12``, so the branch injects small random Gaussians at every catalogued
      line position from the first forward pass. Measured on a randomly
      initialised model, ``max|delta| ~ 0.03`` in normalised flux units.

    So SR2 begins as a *near*-identity on top of SR1, not an exact one. The
    perturbation is small relative to the normalised flux scale, but it is the
    amplitude head — the component that sets emission-line flux — that carries
    it.

    .. note::

       Flux conservation is the known weak point of this architecture: it places
       lines at the right wavelengths but does not necessarily preserve their
       integrated flux. Anything touching ``amp_head``, the presence gate, or
       the residual cap should be evaluated against an SR-versus-HR line flux
       comparison (``scripts/flux_conservation.py``), not against positional
       accuracy or S/N alone.

       One mechanism, found on an earlier checkpoint, is worth stating plainly,
       because it is a property of this ``amp * presence`` factorisation rather
       than a tuning accident. Under a blanket sparsity penalty the presence
       gate settled at ~0.002 *for real and absent lines alike* -- a
       discrimination ratio of 0.95, i.e. none -- so it stopped acting as a
       selector and became a constant multiplier on every amplitude. The line
       branch then supplied 0.17% of the flux the reference required, and every
       gain SR2 showed on lines was coming from the CNN branch instead. The
       amplitudes themselves were fine: ``amp_head`` was emitting ~7 in
       normalised units where the reference wanted ~8.9. Only the gate was
       wrong. Because the two are multiplied, any pressure on presence is
       pressure on flux, so presence must be supervised rather than merely
       penalised.
    """

    def __init__(
        self,
        in_channels: int,
        line_rest_um,
        wave_hi_um,
        line_dim: int = 128,
        num_attn_heads: int = 4,
        num_attn_layers: int = 4,
        window_half: int = 25,
        cnn_dim: int = 96,
        num_cnn_blocks: int = 6,
        dropout: float = 0.02,
        cnn_scale: float = 1.0,
    ):
        super().__init__()
        self.cnn_scale = float(cnn_scale)
        self.K = len(line_rest_um)
        self.L = len(wave_hi_um)
        self.window_half = window_half
        self.W = 2 * window_half + 1

        self.register_buffer(
            "line_rest_um", torch.tensor(np.asarray(line_rest_um), dtype=torch.float32)
        )
        self.register_buffer(
            "wave_hi_um", torch.tensor(np.asarray(wave_hi_um), dtype=torch.float32)
        )

        # --- Line branch ---
        self.line_embed = nn.Embedding(self.K, line_dim)
        self.line_encoder = nn.Sequential(
            nn.Conv1d(in_channels, line_dim, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(line_dim, line_dim, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.line_proj = nn.Linear(line_dim * 4, line_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=line_dim,
            nhead=num_attn_heads,
            dim_feedforward=line_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.line_attn = nn.TransformerEncoder(enc_layer, num_layers=num_attn_layers)

        # Gaussian parameter heads
        self.amp_head = nn.Linear(line_dim, 1)
        self.logw_head = nn.Linear(line_dim, 1)
        self.offset_head = nn.Linear(line_dim, 1)
        self.presence_head = nn.Linear(line_dim, 1)

        # Start near zero amplitude, low presence, moderate width.
        nn.init.normal_(self.amp_head.weight, std=0.01)
        nn.init.constant_(self.amp_head.bias, 0.0)
        nn.init.constant_(self.presence_head.bias, -2.0)
        nn.init.constant_(self.logw_head.bias, 1.0)

        # --- CNN branch ---
        g = min(8, cnn_dim)
        self.cnn_in = nn.Sequential(nn.Conv1d(in_channels, cnn_dim, 5, padding=2), nn.GELU())
        self.cnn_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(g, cnn_dim),
                    nn.GELU(),
                    nn.Conv1d(cnn_dim, cnn_dim, 3, padding=1, bias=False),
                    nn.GroupNorm(g, cnn_dim),
                    nn.GELU(),
                    nn.Conv1d(cnn_dim, cnn_dim, 3, padding=1, bias=False),
                    nn.Dropout(dropout),
                )
                for _ in range(num_cnn_blocks)
            ]
        )
        self.cnn_out = nn.Conv1d(cnn_dim, 1, 1)
        nn.init.constant_(self.cnn_out.weight, 0.0)
        nn.init.constant_(self.cnn_out.bias, 0.0)

        # --- Predictive uncertainty ---
        self.logvar_head = nn.Sequential(
            nn.Conv1d(in_channels, cnn_dim, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(cnn_dim, 1, 1),
        )
        nn.init.constant_(self.logvar_head[-1].weight, 0.0)
        nn.init.constant_(self.logvar_head[-1].bias, -2.0)

    def _line_positions(self, zhat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map redshift to fractional pixel positions.

        Returns ``(B, K)`` positions and a ``(B, K)`` boolean marking which lines
        actually fall inside the observed grid. Lines outside the grid are kept
        as tokens (so attention still sees the full catalogue) but are zeroed out
        of the reconstruction.
        """
        lam_obs = self.line_rest_um[None, :] * (1.0 + zhat[:, None])
        wave = self.wave_hi_um
        lam_c = lam_obs.clamp(wave[0], wave[-1])
        idx = torch.searchsorted(wave, lam_c).clamp(1, self.L - 1)
        frac = (lam_c - wave[idx - 1]) / (wave[idx] - wave[idx - 1] + 1e-12)
        pos = (idx - 1).float() + frac
        in_range = (lam_obs >= wave[0]) & (lam_obs <= wave[-1])
        return pos, in_range

    def line_positions(self, zhat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Public view of :meth:`_line_positions`, for losses that need to line
        their windows up with where this model placed its Gaussians."""
        return self._line_positions(zhat)

    def _extract_windows(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Gather a window around each line position: ``(B,C,L) -> (B,K,C,W)``."""
        B, C, L = x.shape
        h = self.window_half
        pos_int = positions.round().long().clamp(h, L - h - 1)
        offsets = torch.arange(-h, h + 1, device=x.device)
        idx = (pos_int.unsqueeze(-1) + offsets[None, None, :]).clamp(0, L - 1)
        idx_flat = idx.reshape(B, -1).unsqueeze(1).expand(-1, C, -1)
        return torch.gather(x, 2, idx_flat).reshape(B, C, self.K, self.W).permute(0, 2, 1, 3)

    def _reconstruct_gaussians(
        self,
        amp: torch.Tensor,
        width: torch.Tensor,
        offset: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Render the per-line Gaussians onto the wavelength grid, ``(B,1,L)``.

        Each Gaussian is evaluated only within its own window and accumulated
        with ``scatter_add_``, so overlapping lines (a resolved doublet, say)
        sum correctly rather than one overwriting the other.
        """
        B, device, h = amp.shape[0], amp.device, self.window_half
        centers = positions + offset
        center_int = centers.round().long().clamp(h, self.L - h - 1)
        off_grid = torch.arange(-h, h + 1, device=device, dtype=torch.float32)
        pix = center_int.unsqueeze(-1).float() + off_grid[None, None, :]
        diff = pix - centers.unsqueeze(-1)
        sigma = width.unsqueeze(-1).clamp_min(0.5)
        profiles = torch.exp(-0.5 * (diff / sigma) ** 2)
        weighted = (amp.unsqueeze(-1) * profiles).reshape(B, -1)
        idx = (
            (center_int.unsqueeze(-1) + off_grid[None, None, :].long())
            .clamp(0, self.L - 1)
            .reshape(B, -1)
        )
        delta = torch.zeros(B, self.L, device=device)
        delta.scatter_add_(1, idx, weighted)
        return delta.unsqueeze(1)

    def forward(self, x: torch.Tensor, zhat: torch.Tensor):
        B = x.shape[0]

        # --- Line branch: one token per catalogued feature ---
        positions, in_range = self._line_positions(zhat)
        windows = self._extract_windows(x, positions)
        enc = self.line_encoder(windows.reshape(B * self.K, -1, self.W))
        feat = self.line_proj(enc.reshape(B * self.K, -1)).reshape(B, self.K, -1)
        # Identity embedding so attention knows which line each token is.
        feat = feat + self.line_embed.weight[None, :, :]
        feat = self.line_attn(feat)

        amp = self.amp_head(feat).squeeze(-1)
        width = torch.exp(self.logw_head(feat).squeeze(-1).clamp(-2, 3))
        offset = self.offset_head(feat).squeeze(-1).clamp(-10, 10)
        presence_logit = self.presence_head(feat).squeeze(-1)
        presence = torch.sigmoid(presence_logit)
        # Gate by presence, and hard-zero lines outside the observed range.
        amp = amp * presence * in_range.float()

        line_delta = self._reconstruct_gaussians(amp, width, offset, positions)

        # --- CNN branch: smooth continuum corrections ---
        h = self.cnn_in(x)
        for blk in self.cnn_blocks:
            h = h + 0.5 * blk(h)
        cnn_delta = self.cnn_out(h)

        delta = line_delta + self.cnn_scale * cnn_delta
        logvar = self.logvar_head(x)

        if self.training:
            # The logit is returned alongside the probability because the
            # presence loss is a *class-imbalanced* BCE -- roughly 3% of
            # catalogued lines are really present -- and `pos_weight` is only
            # available on the with-logits form. Recovering the logit by
            # inverting the sigmoid loses precision exactly where it matters,
            # at the saturated tails.
            return delta, logvar, presence, presence_logit
        return delta, logvar
