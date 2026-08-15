"""Loss functions for the SR1 and SR2 stages.

Both stages optimise a heteroscedastic Gaussian likelihood against the
medium-resolution reference, plus terms that specifically target *line*
structure. The likelihood alone is not enough: it is dominated by the many
continuum pixels, and a model that produces a smooth, well-calibrated
reconstruction with no sharp lines scores well on it. The extra terms are what
make the model commit to narrow features.

A recurring pattern here is *gating*: extra pressure is applied only where the
reference actually shows line structure, and is scaled down for spectra that
have essentially none, so featureless objects are not penalised for staying
smooth.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.blocks import highpass, odd_kernel, smooth1d

__all__ = [
    "masked_mean",
    "robust_mad",
    "finite_diff",
    "finite_diff2",
    "keep_only_wide",
    "make_line_mask_from_smoothed",
    "gather_line_windows",
    "line_window_flux",
    "line_presence_target",
    "sr1_deblend_loss",
    "sr2_loss",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def masked_mean(x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Mean over valid samples only.

    ``valid`` marks wavelengths the reference actually measured. Samples where
    the detector recorded nothing must contribute *nothing* to the objective:
    the original pipeline filled them with the per-spectrum median, which
    trained the model to emit a flat continuum there and penalised it for
    placing a real line at a wavelength nobody observed. Excluding is the only
    honest option, since any fill value is a fabricated target.
    """
    if valid is None:
        return x.mean()
    v = valid.to(x.dtype)
    if v.shape != x.shape:
        v = v.expand_as(x)
    total = v.sum()
    if total <= 0:
        return x.sum() * 0.0  # keeps the graph alive with zero gradient
    return (x * v).sum() / total


def robust_mad(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Median absolute deviation, scaled to be a consistent estimator of sigma.

    Used instead of the standard deviation because emission lines are exactly
    the kind of large outlier that would inflate it — the scale must describe the
    noise, not the signal we are trying to detect.
    """
    med = x.median(dim=dim, keepdim=True).values
    mad = (x - med).abs().median(dim=dim, keepdim=True).values
    return 1.4826 * mad + eps


def finite_diff(x: torch.Tensor) -> torch.Tensor:
    """First difference along the last axis, length preserved."""
    dx = x[..., 1:] - x[..., :-1]
    return F.pad(dx, (0, 1), mode="replicate")


def finite_diff2(x: torch.Tensor) -> torch.Tensor:
    """Second difference along the last axis, length preserved."""
    return finite_diff(finite_diff(x))


def keep_only_wide(mask: torch.Tensor, min_width: int = 7) -> torch.Tensor:
    """Drop mask features narrower than ``min_width`` samples.

    Single-sample spikes are cosmic rays or noise, not resolved emission lines.
    Requiring a minimum width keeps the line mask from chasing them.
    """
    min_width = max(int(min_width), 1)
    w = torch.ones(1, 1, min_width, device=mask.device)
    counts = F.conv1d(mask, w, padding=min_width // 2)
    return (counts >= float(min_width)).float()


def make_line_mask_from_smoothed(
    x_high_raw: torch.Tensor,
    smooth_k: int = 121,
    thresh_mad: float = 7.5,
    dilate: int = 11,
    min_width: int = 7,
) -> torch.Tensor:
    """Locate line regions in the reference spectrum.

    The continuum is estimated by heavy smoothing and subtracted; what remains is
    thresholded at ``thresh_mad`` robust sigma, filtered to remove narrow spikes,
    and dilated so the mask covers the line wings rather than only its core.

    Deriving the mask from the *reference* rather than from a line list means it
    responds to lines that are actually present at the observed strength, and
    does not mark lines that fall in the wavelength range but are undetected.

    .. warning::

       **These defaults are sized for the retired 2,500-point linear grid.** All
       four are measured in samples, so they mean different things on the
       6,671-point log grid, and ``min_width`` in particular is the difference
       between flagging 46% and 74% of the lines with HR SNR >= 20. Every caller
       on the log grid must pass its own values -- :func:`sr1_deblend_loss` takes
       them from config, and :func:`sr2_loss` passes ``presence_mask_*``, which
       it did not until 2026-08-02. The defaults are kept only so the old-grid
       behaviour stays reproducible.
    """
    x_smooth = smooth1d(x_high_raw, k=smooth_k)
    hp = x_high_raw - x_smooth
    scale = robust_mad(hp, dim=-1)
    mask = ((hp.abs() / scale) > float(thresh_mad)).float()
    mask = keep_only_wide(mask, min_width=min_width)

    if dilate and int(dilate) > 1:
        k = odd_kernel(dilate, mask.shape[-1])
        pad = k // 2
        mask = F.max_pool1d(F.pad(mask, (pad, pad), mode="replicate"), kernel_size=k, stride=1)
    return mask


# --------------------------------------------------------------------------
# per-line windows
# --------------------------------------------------------------------------
#
# The HR grid is logarithmic with a constant dln(lambda) = 2.5e-4 (R = 4000,
# 74.96 km/s per sample), so a velocity window is a *fixed* number of samples
# everywhere in the spectrum. That is what makes these windows plain integer
# slices rather than a per-line wavelength computation. The defaults below are
# the same velocities scripts/flux_conservation.py measures with -- core
# +/-500 km/s, sidebands 800-1500 km/s -- so the training signal and the
# reported measurement are the same quantity.


def gather_line_windows(
    x: torch.Tensor, positions: torch.Tensor, half: int
) -> torch.Tensor:
    """Gather ``+/-half`` samples around each line: ``(B,1,L) -> (B,K,2*half+1)``."""
    B, _, L = x.shape
    K = positions.shape[1]
    pos_int = positions.round().long().clamp(0, L - 1)
    off = torch.arange(-half, half + 1, device=x.device)
    idx = (pos_int.unsqueeze(-1) + off[None, None, :]).clamp(0, L - 1)
    return torch.gather(x.squeeze(1).unsqueeze(1).expand(B, K, L), 2, idx)


def line_window_flux(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    core_half: int = 7,
    sb_lo: int = 11,
    sb_hi: int = 20,
    all_positions: torch.Tensor | None = None,
    clean_half: int = 7,
    min_clean: int = 5,
) -> torch.Tensor:
    """Continuum-subtracted integrated flux per line, ``(B,K)``.

    The continuum is the median of two sidebands flanking the core, so the
    per-spectrum *additive* normalisation offset cancels and only the common
    multiplicative scale survives. That is what makes an SR-versus-HR ratio
    meaningful in normalised units, without de-normalising first.

    Summation is in sample units, with no ``dlambda`` factor. On this grid
    ``dlambda`` varies across the window, but it is the *same* at every sample
    for both spectra being compared, so it cancels from the relative residual
    the flux loss actually uses. Including it would only rescale each line.

    **Blended lines.** A plain sideband median assumes the sidebands hold only
    continuum, which is false for a large minority of the catalogue: 11 of the
    98 lines have a neighbour inside *both* sidebands (Halpha, hemmed in by
    [N II] 6548 at -675 km/s and [N II] 6583 at +940, is the clearest case), and
    many more have one contaminated side. Because the same estimator is applied
    to prediction and reference, this does not displace the optimum -- a perfect
    reconstruction still scores zero -- but it makes the term *degenerate*: the
    model can match the measured flux by getting a line and its neighbour wrong
    in compensating directions. That is worst exactly for the blends, which are
    what deblending has to get right, so the answer is a better continuum rather
    than a shorter line list.

    Pass ``all_positions`` (every catalogued line, ``(B,K)``) to exclude sideband
    samples sitting within ``clean_half`` of a *different* line. Where fewer than
    ``min_clean`` samples survive, the plain median over the full sideband is
    used instead, so the estimate degrades to the old behaviour rather than to
    nothing.
    """
    win = gather_line_windows(x, positions, sb_hi)
    off = torch.arange(-sb_hi, sb_hi + 1, device=x.device)
    core = (off.abs() <= core_half).nonzero(as_tuple=True)[0]
    side = ((off.abs() >= sb_lo) & (off.abs() <= sb_hi)).nonzero(as_tuple=True)[0]
    side_vals = win[..., side]                                   # (B,K,S)
    cont = side_vals.median(dim=-1, keepdim=True).values

    if all_positions is not None:
        # Offset of every other line from each target line, in samples.
        delta = all_positions[:, None, :] - positions[:, :, None]      # (B,K,Kall)
        # "Other" is decided by separation, not by index: `all_positions` is
        # normally the same tensor as `positions`, but need not be, and an
        # identity matrix sized by the target axis silently disables the whole
        # mask when the two differ in length.
        is_other = delta.abs() > 0.5
        s_off = off[side].to(delta.dtype)                              # (S,)
        near = (s_off[None, None, :, None] - delta[:, :, None, :]).abs() <= clean_half
        contaminated = (near & is_other[:, :, None, :]).any(dim=-1)    # (B,K,S)

        clean = ~contaminated
        n_clean = clean.sum(dim=-1, keepdim=True)
        masked = side_vals.masked_fill(contaminated, float("nan"))
        cont_clean = masked.nanmedian(dim=-1, keepdim=True).values
        cont = torch.where((n_clean >= min_clean) & torch.isfinite(cont_clean),
                           cont_clean, cont)

    return (win[..., core] - cont).sum(dim=-1)


def _bright_half_median(ratio: torch.Tensor, f_hr: torch.Tensor,
                        mask: torch.Tensor) -> float:
    """Median SR/HR flux ratio over the brighter half of the selected lines.

    A stand-in for the ``HR SNR >= 20`` population the paper reports on, usable
    where per-line errors are not: the dataset leaves HR errors in physical
    units while normalising the flux, so a real SNR cannot be formed here.
    Brightness is the available proxy and tracks it closely.
    """
    if not mask.any():
        return float("nan")
    sel = f_hr.abs()[mask]
    cut = sel.median()
    bright = mask & (f_hr.abs() >= cut)
    return float(ratio[bright].median().item()) if bright.any() else float("nan")


def line_presence_target(
    x_high: torch.Tensor,
    positions: torch.Tensor,
    *,
    in_range: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
    core_half: int = 7,
    **mask_kw,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Which catalogued lines the reference spectrum actually shows.

    Returns ``(target, weight)``, both ``(B, K)``. ``target`` is 1 where the
    HR-derived line mask covers the line's core window, and ``weight`` excludes
    lines that cannot be judged at all: those outside the observed grid, and
    those whose core the detector never measured. Excluding is important --
    scoring an unobservable line as a *negative* would train the presence head
    to switch off lines that are merely off the edge of the detector.

    Deriving the label from the reference rather than from the line list is the
    same choice :func:`make_line_mask_from_smoothed` makes, for the same reason:
    it responds to lines present at the observed strength, not to every
    transition that happens to fall in range.
    """
    hr_mask = make_line_mask_from_smoothed(x_high, **mask_kw)
    if valid is not None:
        hr_mask = hr_mask * valid.to(hr_mask.dtype).expand_as(hr_mask)
    target = gather_line_windows(hr_mask, positions, core_half).amax(dim=-1)

    weight = torch.ones_like(target)
    if in_range is not None:
        weight = weight * in_range.to(weight.dtype)
    if valid is not None:
        v = valid.to(weight.dtype).expand(-1, 1, -1)
        # More than half the core has to be real data for the label to mean
        # anything; a line sitting on the edge of a gap is not evidence either
        # way.
        v_frac = gather_line_windows(v, positions, core_half).mean(dim=-1)
        weight = weight * (v_frac > 0.5).to(weight.dtype)
    return target, weight


# --------------------------------------------------------------------------
# SR1
# --------------------------------------------------------------------------


def sr1_deblend_loss(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    x_high: torch.Tensor,
    x_high_err: torch.Tensor,
    valid: torch.Tensor | None = None,
    logvar_reg: float = 3.38737824685736e-6,
    mask_smooth_k: int = 121,
    mask_thresh_mad: float = 7.5,
    mask_dilate: int = 11,
    mask_min_width: int = 7,
    lam_d1: float = 0.11001162460004914,
    lam_d2: float = 0.010153811105728492,
    gate_min_frac: float = 0.015,
    gate_temp: float = 0.05,
    score_w_recon: float = 0.2,
    score_w_line: float = 2.0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict]:
    """Heteroscedastic likelihood plus gated sharpness matching, for SR1.

    Three parts:

    1. **Likelihood.** Gaussian NLL where the total variance is the *sum* of the
       predicted model variance and the reference spectrum's own measurement
       variance. Keeping these separate matters: the model should not be
       rewarded for reporting confidence the data cannot support, nor penalised
       for failing to match noise.

    2. **Gated sharpness.** First and second derivatives of the prediction are
       matched to the reference, but only inside the line mask and normalised by
       the robust scale of the reference's own derivatives. Matching derivatives
       is what forces narrow features to be reproduced at the right width — plain
       NLL is happy with a broadened line of the correct integrated flux.

    3. **Gate.** The sharpness term is scaled by a smooth function of how much of
       the spectrum the line mask covers. Spectra with essentially no detected
       lines contribute almost nothing, so featureless objects do not push the
       model to sharpen noise.

    Returns the total loss and a dict of diagnostics for logging.

    ``valid`` marks wavelengths the reference actually measured. Two things are
    required of the caller, and they are different:

    * Invalid samples must carry a **neutral numerical value** (zero, in the
      per-spectrum normalised units this loss works in) rather than ``nan`` or an
      arbitrary number. The line mask is derived from ``x_high`` by smoothing
      over a wide kernel, so a wild value at an unmeasured wavelength would be
      dragged into neighbouring *measured* samples and be mistaken for a line.
    * ``valid`` then removes those samples from the objective, so no gradient
      rewards matching the neutral value.

    Both are needed. Filling alone is what the original pipeline did — it used
    the per-spectrum median and trained on it, teaching the model to emit flat
    continuum wherever the detector recorded nothing. Masking alone leaves the
    fill value free to contaminate any neighbourhood statistic.
    """
    model_var = torch.exp(log_var)
    total_var = (model_var + x_high_err**2).clamp_min(1e-8)
    nll = 0.5 * (torch.log(total_var + eps) + (mean - x_high) ** 2 / (total_var + eps))
    base_loss = masked_mean(nll, valid)

    reg = float(logvar_reg) * (log_var**2).mean()

    line_mask = make_line_mask_from_smoothed(
        x_high_raw=x_high,
        smooth_k=mask_smooth_k,
        thresh_mad=mask_thresh_mad,
        dilate=mask_dilate,
        min_width=mask_min_width,
    )

    if valid is not None:
        # A line cannot be "detected" where nothing was measured.
        line_mask = line_mask * valid.to(line_mask.dtype).expand_as(line_mask)

    frac = line_mask.mean(dim=-1, keepdim=True)
    # Detached: the gate decides how much to weight this spectrum, it is not
    # itself something to optimise.
    gate = torch.sigmoid((frac - float(gate_min_frac)) / float(gate_temp)).detach()

    d1_pred, d1_tgt = finite_diff(mean), finite_diff(x_high)
    d2_pred, d2_tgt = finite_diff2(mean), finite_diff2(x_high)

    s1 = robust_mad(d1_tgt, dim=-1)
    s2 = robust_mad(d2_tgt, dim=-1)

    denom = line_mask.sum(dim=-1) + 1e-6
    sharp1 = (((d1_pred - d1_tgt).abs() / s1) * line_mask).sum(dim=-1) / denom
    sharp2 = (((d2_pred - d2_tgt).abs() / s2) * line_mask).sum(dim=-1) / denom
    sharp_loss = (gate.squeeze(-1) * (float(lam_d1) * sharp1 + float(lam_d2) * sharp2)).mean()

    total = float(score_w_recon) * base_loss + float(score_w_line) * sharp_loss + reg

    with torch.no_grad():
        resid = mean - x_high
        comps = {
            "loss_base_nll": base_loss.item(),
            "loss_logvar_reg": reg.item(),
            "loss_sharp": sharp_loss.item(),
            "loss_total": total.item(),
            "mask_frac_mean": frac.mean().item(),
            "mask_frac_p50": frac.median().item(),
            "mask_frac_p90": frac.quantile(0.9).item(),
            "gate_mean": gate.mean().item(),
            "resid_rms": masked_mean(resid.pow(2), valid).sqrt().item(),
            "total_var_min": total_var.min().item(),
            "total_var_p50": total_var.median().item(),
            "total_var_p90": total_var.quantile(0.9).item(),
        }
    return total, comps


# --------------------------------------------------------------------------
# SR2
# --------------------------------------------------------------------------


def sr2_loss(
    *,
    sr2_mean: torch.Tensor,
    sr2_logvar: torch.Tensor,
    x_high: torch.Tensor,
    x_high_err: torch.Tensor,
    line_mask: torch.Tensor,
    valid: torch.Tensor | None = None,
    presence: torch.Tensor,
    zhead=None,
    z_true: torch.Tensor | None = None,
    ztransform=None,
    use_sigma: bool = True,
    lam_z: float = 0.0,
    lam_hp_in: float = 0.0,
    lam_hp_out: float = 0.0,
    hp_k: int = 51,
    lam_sparse: float = 0.0,
    var_floor: float = 1e-8,
    hp_in_noise_weighted: bool = False,
    presence_logit: torch.Tensor | None = None,
    line_positions: torch.Tensor | None = None,
    line_in_range: torch.Tensor | None = None,
    lam_presence: float = 0.0,
    lam_flux: float = 0.0,
    presence_pos_weight: float = 0.0,
    presence_pos_weight_max: float = 50.0,
    flux_core_half: int = 7,
    flux_sb_lo: int = 11,
    flux_sb_hi: int = 20,
    flux_eps: float = 1.0,
    flux_beta: float = 0.5,
    flux_weight_power: float = 1.0,
    flux_clean_half: int = 7,
    presence_mask_smooth_k: int = 259,
    presence_mask_thresh_mad: float = 7.5,
    presence_mask_dilate: int = 11,
    presence_mask_min_width: int = 5,
) -> tuple[torch.Tensor, dict]:
    """Likelihood, high-pass matching, presence sparsity and a redshift term.

    **High-pass matching** compares the small-scale content of prediction and
    reference after removing the smooth component. It is applied separately
    inside and outside the line mask (``lam_hp_in`` / ``lam_hp_out``) because the
    two regions want opposite behaviour: sharp structure should be reproduced on
    lines, and suppressed off them. A single weight cannot express that.

    The two halves are weighted differently, and this is not an oversight.
    Outside the lines the term is inverse-variance weighted, which is the
    ordinary thing to do. Inside the lines it is **not**.

    The reason is a comparison *within* the mask, not between lines and
    continuum: ``hp_in`` is a weighted mean over masked pixels alone, so a
    uniform rescaling of those weights cancels and the continuum's weight is
    irrelevant to it. What survives is the spread inside the mask. Photon noise
    scales with flux, so the bright line cores are the noisiest pixels present
    and ``1/sigma^2`` drives their weight to near zero: on the validation split
    the brightest 10% of in-line pixels take 2.1% of the in-line weight, and the
    faintest 10% are weighted 34x more heavily than the brightest. The term that
    exists to fit lines was therefore decided by the faintest pixels in the mask
    and nearly blind to the flux-carrying cores, which starved the line branch
    and collapsed the presence gate.

    Set ``hp_in_noise_weighted=True`` to restore the old behaviour; it is kept
    only so the ablation can be run.

    **Presence supervision** (``lam_presence``) trains the gate as a classifier
    against the lines the reference spectrum actually shows, labelled by
    :func:`line_presence_target`. This replaces the older blanket sparsity
    penalty (``lam_sparse``, now off by default), which pushed every line logit
    down at a constant rate regardless of whether the line was real and drove
    the gate to a constant ~0.002 — the same value for a bright line as for one
    that is absent. Because the model forms ``amp * presence``, that constant
    multiplied *all* line flux, and the line branch ended up supplying 0.17% of
    the flux the reference required. Sparsity is still what keeps spurious lines
    off, but it now comes from the negative half of a two-sided objective rather
    than from a uniform downward force.

    **The presence mask settings are grid-dependent and must be passed**, which
    is why ``presence_mask_*`` are explicit parameters rather than left to
    :func:`make_line_mask_from_smoothed`'s defaults. Those defaults are sized for
    the retired 2,500-point linear grid, and this function silently used them
    until 2026-08-02 -- so the mask defining both the presence labels *and* the
    flux term's line list (``w_flux = w_line * tgt``) was built with the wrong
    kernel sizes on the log grid.

    Measured on the 572-galaxy val split against integrated HR line SNR, the
    cost was carried almost entirely by ``min_width``:

    ==================  ==========  ========  ==========
    setting             recall@20   fpr@<3    precision
    ==================  ==========  ========  ==========
    ``min_width=7``     0.459       0.012     0.631
    ``min_width=5``     **0.735**   0.031     0.520
    ``min_width=3``     0.865       0.067     0.374
    ==================  ==========  ========  ==========

    At 7 the mask **missed 55% of the lines with HR SNR >= 20** -- they were
    labelled absent for the presence BCE and dropped from the flux term. The
    cause is not the threshold: only 7.4% of those lines peak below 7.5 sigma
    per pixel (median 32.7). It is that :func:`keep_only_wide` demands
    ``min_width`` *consecutive* supra-threshold samples, and an R=1000 line on
    this R=4000 grid has sigma ~1.7 samples, so it clears 7.5 sigma over only
    ~5-6. ``smooth_k`` and ``dilate`` were measured to be near-neutral for this
    consumer over 121-430 and 11-29 respectively.

    Do **not** "restore" the nominal log-grid rescaling recorded in
    by rescaling arithmetic (``smooth_k=430, dilate=29, min_width=19``): measured,
    it flags **1.1%** of bright lines, because 19 samples is 1425 km/s and no
    real line is that wide.

    **Line flux matching** (``lam_flux``) compares continuum-subtracted
    integrated flux in a velocity window around each real line, prediction
    against reference. Nothing else in this objective measures integrated line
    flux: the likelihood is inverse-variance weighted and so is quietest at the
    bright cores, and the high-pass term is a shape comparison that a correct
    profile at the wrong amplitude can still satisfy reasonably well.

    **Redshift coupling** optionally back-propagates a redshift error through
    SR2, so the refinement is pushed towards spectra from which redshift is
    recoverable. The error is scaled by ``1 / (1 + z)``, the standard convention,
    so high-redshift objects are not weighted disproportionately.

    ``ztransform`` is a :class:`~specsr.training.ztransform.RedshiftTransform`;
    passing it rather than loose statistics guarantees the decoding matches what
    the redshift head was trained with.

    ``valid`` marks wavelengths the reference actually measured. Two things are
    required of the caller, and they are different:

    * Invalid samples must carry a **neutral numerical value** (zero, in the
      per-spectrum normalised units this loss works in) rather than ``nan`` or an
      arbitrary number. The line mask is derived from ``x_high`` by smoothing
      over a wide kernel, so a wild value at an unmeasured wavelength would be
      dragged into neighbouring *measured* samples and be mistaken for a line.
    * ``valid`` then removes those samples from the objective, so no gradient
      rewards matching the neutral value.

    Both are needed. Filling alone is what the original pipeline did — it used
    the per-spectrum median and trained on it, teaching the model to emit flat
    continuum wherever the detector recorded nothing. Masking alone leaves the
    fill value free to contaminate any neighbourhood statistic.
    """
    model_var = torch.exp(sr2_logvar)
    total_var = (model_var + x_high_err**2).clamp_min(var_floor)
    nll = 0.5 * (torch.log(total_var + 1e-12) + (sr2_mean - x_high) ** 2 / (total_var + 1e-12))
    loss = masked_mean(nll, valid)

    # --- high-pass matching, inverse-variance weighted ---
    hp_sr2 = highpass(sr2_mean, k=hp_k)
    hp_hr = highpass(x_high, k=hp_k)
    # Normalise by the reference's own high-pass scale so the term is
    # dimensionless and comparable across spectra of very different brightness.
    hp_scale = hp_hr.detach().abs().median().clamp_min(1e-3)
    hp_diff = (hp_sr2 - hp_hr) / hp_scale

    inv_var = 1.0 / (x_high_err**2).clamp_min(var_floor)
    inv_var = inv_var / inv_var.mean().clamp_min(1e-12)
    inv_var_hp = smooth1d(inv_var, k=hp_k)
    inv_var_hp = inv_var_hp / inv_var_hp.mean().clamp_min(1e-12)

    hp_loss = F.smooth_l1_loss(hp_diff, torch.zeros_like(hp_diff), reduction="none", beta=0.1)

    # Keep the validity mask separate from the inverse-variance weights. They
    # used to be multiplied together here, which meant the in-line term could
    # only drop unmeasured samples by *also* accepting noise weighting -- and
    # turning that weighting off would silently readmit pixels the detector
    # never recorded.
    if valid is not None:
        vmask = valid.to(inv_var_hp.dtype).expand_as(inv_var_hp)
    else:
        vmask = torch.ones_like(inv_var_hp)

    hp_in_val = hp_out_val = 0.0
    if lam_hp_in > 0:
        # Deliberately NOT inverse-variance weighted by default; see the
        # docstring. Short version: this is a weighted mean over line pixels
        # only, so the continuum's weight cancels and what matters is the spread
        # *within* the mask -- where 1/sigma^2 gives the brightest 10% of pixels
        # 2.1% of the weight, leaving the term blind to the very cores it exists
        # to fit. The line mask already selects the pixels of interest.
        w_in = line_mask * vmask * (inv_var_hp if hp_in_noise_weighted else 1.0)
        hp_in = (w_in * hp_loss).sum() / w_in.sum().clamp_min(1e-8)
        loss = loss + lam_hp_in * hp_in
        hp_in_val = float(hp_in.item())
    if lam_hp_out > 0:
        # Off the lines, noise weighting is doing what it should: this term
        # suppresses spurious structure in the continuum, and pixels the
        # instrument measured poorly should not drive that.
        w_out = (1.0 - line_mask) * vmask * inv_var_hp
        hp_out = (w_out * hp_loss).sum() / w_out.sum().clamp_min(1e-8)
        loss = loss + lam_hp_out * hp_out
        hp_out_val = float(hp_out.item())

    # --- push unused lines towards off ---
    #
    # `lam_sparse` is the original blanket penalty on mean presence, and it is
    # off by default now because it is what killed the line branch. Its gradient
    # is a constant on every one of the 98 line logits whether or not the line
    # is real, while the only upward pressure -- the likelihood -- is
    # inverse-variance weighted and therefore weakest at the bright line cores
    # it needs to defend. Presence settled at ~0.002 for real and absent lines
    # alike (a discrimination ratio of 0.95, i.e. none), and since the model
    # forms `amp * presence`, that constant became a blanket 0.002 multiplier on
    # every line amplitude. Kept only so the ablation can be run.
    if lam_sparse > 0:
        loss = loss + lam_sparse * presence.mean()

    # --- supervise presence against the lines the reference actually shows ---
    pres_bce_val = 0.0
    pres_sep_val = float("nan")
    pos_w_val = 0.0
    if (lam_presence > 0 or lam_flux > 0) and line_positions is not None:
        tgt, w_line = line_presence_target(
            x_high, line_positions, in_range=line_in_range, valid=valid,
            core_half=flux_core_half,
            smooth_k=presence_mask_smooth_k, thresh_mad=presence_mask_thresh_mad,
            dilate=presence_mask_dilate, min_width=presence_mask_min_width,
        )

        if lam_presence > 0 and presence_logit is not None:
            n_pos = (tgt * w_line).sum()
            n_neg = ((1.0 - tgt) * w_line).sum()
            # Real lines are ~3% of catalogued positions. Unweighted BCE on that
            # imbalance has the same fixed point as the penalty it replaces --
            # predict "absent" everywhere -- so the positives are up-weighted to
            # parity. Capped because a batch with one positive would otherwise
            # hand that single line the whole gradient.
            if presence_pos_weight > 0:
                pos_w = torch.as_tensor(presence_pos_weight, device=loss.device)
            else:
                pos_w = (n_neg / n_pos.clamp_min(1.0)).clamp(1.0, presence_pos_weight_max)
            bce = F.binary_cross_entropy_with_logits(
                presence_logit, tgt, reduction="none", pos_weight=pos_w)
            pres_bce = (bce * w_line).sum() / w_line.sum().clamp_min(1.0)
            loss = loss + lam_presence * pres_bce
            pres_bce_val = float(pres_bce.item())
            pos_w_val = float(pos_w.item())

        # --- match integrated line flux on the lines that are really there ---
        #
        # Integrated line flux, optimised directly rather than hoped for. It is
        # deliberately NOT inverse-variance weighted: photon noise makes the
        # line cores the noisiest samples in the spectrum, so a 1/sigma^2 weight
        # would once again make the term that exists to fit lines nearly blind
        # to the flux-carrying cores. Same argument as `hp_in` above.
        if lam_flux > 0:
            fkw = dict(core_half=flux_core_half, sb_lo=flux_sb_lo, sb_hi=flux_sb_hi,
                       all_positions=line_positions, clean_half=flux_clean_half)
            f_hr = line_window_flux(x_high, line_positions, **fkw).detach()
            f_sr = line_window_flux(sr2_mean, line_positions, **fkw)

            # Weighting by HR brightness, because an unweighted mean over the
            # mask's positives is dominated by faint lines -- they are most of
            # the population -- while the flux number the paper reports is
            # computed on bright ones (HR SNR >= 20). With power 1 the weighted
            # mean of the relative residual is, up to `flux_eps`, total absolute
            # flux error over total flux; power 0 recovers the unweighted form.
            # HR errors cannot be used to select on SNR here: the dataset leaves
            # them in physical units while normalising the flux, so an SNR
            # computed inside this loss would be meaningless.
            w_flux = w_line * tgt
            if flux_weight_power > 0:
                bright = f_hr.abs().pow(flux_weight_power)
                w_flux = w_flux * bright / bright.max().clamp_min(1e-12)

            # `flux_eps` floors the denominator so a line whose HR flux happens
            # to integrate to ~0 cannot dominate the relative residual.
            rel = (f_sr - f_hr) / (f_hr.abs() + flux_eps)
            fl = F.smooth_l1_loss(rel, torch.zeros_like(rel),
                                  reduction="none", beta=flux_beta)
            flux_loss = (fl * w_flux).sum() / w_flux.sum().clamp_min(1e-8)
            loss = loss + lam_flux * flux_loss
            flux_loss_val = float(flux_loss.item())
            with torch.no_grad():
                ratio = f_sr / torch.where(f_hr.abs() > flux_eps, f_hr,
                                           torch.full_like(f_hr, float("nan")))
                m = ((w_line * tgt) > 0) & torch.isfinite(ratio)
                flux_ratio_val = float(ratio[m].median().item()) if m.any() else float("nan")
                # Tracked separately from the median over all positives, because
                # that median moves with the faint lines and the reported number
                # does not.
                flux_ratio_bright_val = _bright_half_median(ratio, f_hr, m)
        else:
            flux_loss_val = 0.0
            flux_ratio_val = float("nan")
            flux_ratio_bright_val = float("nan")

        # The metric that would have caught the collapse: mean presence on real
        # lines over mean presence on absent ones. `presence_mean` alone cannot
        # -- it stays comfortably above any floor while carrying no information.
        with torch.no_grad():
            p_pos = (presence * tgt * w_line).sum() / (tgt * w_line).sum().clamp_min(1.0)
            p_neg = (presence * (1 - tgt) * w_line).sum() / \
                ((1 - tgt) * w_line).sum().clamp_min(1.0)
            pres_sep_val = float((p_pos / p_neg.clamp_min(1e-9)).item())
    else:
        flux_loss_val = 0.0
        flux_ratio_val = float("nan")
        flux_ratio_bright_val = float("nan")

    # --- redshift coupling through SR2 ---
    z_loss_val = torch.tensor(0.0, device=loss.device)
    if lam_z > 0 and zhead is not None and z_true is not None and ztransform is not None:
        sr2_log_sigma = 0.5 * sr2_logvar
        z_in = torch.cat([sr2_mean, sr2_log_sigma], dim=1) if use_sigma else sr2_mean
        mu_raw, logvar_z = zhead(z_in)
        # `bounded_mean` is False only for the classification head, whose
        # estimate is already in normalised units. Default True so any
        # duck-typed head keeps the original Gaussian decoding.
        z_pred, _ = ztransform.predict(
            mu_raw.clamp(-30, 30), logvar_z,
            bounded=getattr(zhead, "bounded_mean", True),
        )
        z_pred = z_pred.reshape(-1)
        # Deliberately not named `valid`: that is the caller's per-wavelength
        # mask, shaped (B, 1, L). This one is per-*sample*, shaped (B,). Reusing
        # the name shadowed the parameter and fed the wrong mask to the metrics
        # below, which crashed on the shape mismatch -- and would have silently
        # mis-reported the NLL had the shapes happened to agree.
        z_finite = torch.isfinite(z_pred) & torch.isfinite(z_true)
        if z_finite.any():
            dz = (z_pred[z_finite] - z_true[z_finite]) / (1.0 + z_true[z_finite].abs())
            z_loss_val = F.smooth_l1_loss(dz, torch.zeros_like(dz), beta=0.1)
            loss = loss + lam_z * z_loss_val

    return loss, {
        "nll": float(masked_mean(nll, valid).item()),
        "hp_in": hp_in_val,
        "hp_out": hp_out_val,
        "z_loss": float(z_loss_val.item()),
        "presence_mean": float(presence.mean().item()),
        "presence_bce": pres_bce_val,
        "presence_sep": pres_sep_val,
        "presence_pos_weight": pos_w_val,
        "flux_loss": flux_loss_val,
        "flux_ratio_median": flux_ratio_val,
        "flux_ratio_bright": flux_ratio_bright_val,
        "loss_total": float(loss.item()),
    }
