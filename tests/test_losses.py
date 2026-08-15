"""Tests for the SR1 and SR2 loss functions.

These check the properties the losses exist to enforce — that sharpness matching
actually rewards narrow features, that the gate spares featureless spectra, that
presence sparsity pushes lines off — rather than just that the arithmetic runs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from specsr.models.blocks import odd_kernel, smooth1d
from specsr.training.losses import (
    finite_diff,
    finite_diff2,
    keep_only_wide,
    line_presence_target,
    line_window_flux,
    make_line_mask_from_smoothed,
    robust_mad,
    sr1_deblend_loss,
    sr2_loss,
)

B, L = 3, 512


def _spectrum_with_lines(n_lines=3, amp=8.0, width=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, 1, L, generator=g) * 0.1
    centers = np.linspace(80, L - 80, n_lines).astype(int)
    for i in range(B):
        for c in centers:
            x[i, 0, c - width : c + width + 1] += amp
    return x, centers


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def test_odd_kernel_always_odd_and_preserves_length():
    """A single sample of length drift here silently corrupts every downstream
    residual, so check exhaustively."""
    for length in (4, 5, 8, 9, 16, 33, 64, 257):
        for k in (3, 4, 5, 8, 31, 64, 121, 200):
            kk = odd_kernel(k, length)
            assert kk % 2 == 1, (k, length, kk)
            assert kk >= 3
            x = torch.randn(1, 1, length)
            assert smooth1d(x, k).shape[-1] == length, (k, length)


def test_robust_mad_ignores_line_outliers():
    """The scale must describe the noise, not the emission lines."""
    noise = torch.randn(1, 1, L) * 0.1
    with_lines = noise.clone()
    with_lines[0, 0, 100:107] += 50.0
    mad_clean = robust_mad(noise).item()
    mad_lines = robust_mad(with_lines).item()
    std_lines = with_lines.std().item()
    assert mad_lines == pytest.approx(mad_clean, rel=0.2)
    assert std_lines > 10 * mad_lines, "std is wrecked by the lines; MAD is not"


def test_finite_diff_preserves_length():
    x = torch.randn(B, 1, L)
    assert finite_diff(x).shape == x.shape
    assert finite_diff2(x).shape == x.shape


def test_keep_only_wide_drops_single_sample_spikes():
    mask = torch.zeros(1, 1, L)
    mask[0, 0, 50] = 1.0                 # 1-sample spike, should go
    mask[0, 0, 200:215] = 1.0            # 15-sample feature, should stay
    out = keep_only_wide(mask, min_width=7)
    assert out[0, 0, 50].item() == 0.0
    assert out[0, 0, 207].item() == 1.0


def test_line_mask_finds_injected_lines_and_ignores_flat_continuum():
    x, centers = _spectrum_with_lines()
    mask = make_line_mask_from_smoothed(x)
    assert mask.shape == x.shape
    assert ((mask == 0) | (mask == 1)).all()
    for c in centers:
        assert mask[0, 0, c].item() == 1.0, f"missed line at {c}"
    # A featureless spectrum should light up almost nothing.
    flat = torch.randn(1, 1, L) * 0.1
    assert make_line_mask_from_smoothed(flat).mean().item() < 0.1


# --------------------------------------------------------------------------
# SR1 loss
# --------------------------------------------------------------------------


def test_sr1_loss_runs_and_reports_diagnostics():
    x_high, _ = _spectrum_with_lines()
    mean = x_high.clone()
    log_var = torch.full_like(mean, -2.0)
    err = torch.full_like(mean, 0.1)
    loss, comps = sr1_deblend_loss(mean, log_var, x_high, err)
    assert torch.isfinite(loss)
    for key in ("loss_base_nll", "loss_sharp", "gate_mean", "mask_frac_mean", "resid_rms"):
        assert key in comps and np.isfinite(comps[key])


def test_sr1_loss_prefers_sharp_reconstruction_over_smoothed():
    """The whole point of the sharpness term: a broadened line must score worse
    than a correctly narrow one, even though both can match integrated flux."""
    x_high, _ = _spectrum_with_lines()
    err = torch.full_like(x_high, 0.1)
    log_var = torch.full_like(x_high, -2.0)

    sharp = x_high.clone()
    blurred = smooth1d(x_high, k=15)

    loss_sharp, _ = sr1_deblend_loss(sharp, log_var, x_high, err)
    loss_blur, _ = sr1_deblend_loss(blurred, log_var, x_high, err)
    assert loss_sharp < loss_blur


def test_sr1_gate_suppresses_line_term_on_featureless_spectra():
    """Objects with no detected lines should not be pushed to sharpen noise."""
    flat = torch.randn(1, 1, L) * 0.1
    err = torch.full_like(flat, 0.1)
    log_var = torch.full_like(flat, -2.0)
    _, comps_flat = sr1_deblend_loss(flat, log_var, flat, err)

    lined, _ = _spectrum_with_lines()
    _, comps_lines = sr1_deblend_loss(lined, torch.full_like(lined, -2.0),
                                      lined, torch.full_like(lined, 0.1))
    assert comps_flat["gate_mean"] < comps_lines["gate_mean"]


@pytest.mark.parametrize("err,log_var_val", [(0.01, -2.0), (1.0, -2.0), (0.3, 0.5)])
def test_sr1_total_variance_is_model_plus_measurement(err, log_var_val):
    """Total variance must be the *sum* of predicted model variance and the
    reference's own measurement variance.

    Keeping them separate is what stops the model being rewarded for reporting
    confidence the data cannot support, or penalised for failing to fit noise.
    Note the NLL is not monotone in the measurement error — log(total_var)
    competes with resid^2/total_var and the likelihood is optimal when
    total_var ~ resid^2 — so this checks the decomposition directly rather than
    an ordering of losses.
    """
    x_high, _ = _spectrum_with_lines()
    mean = x_high + 0.5
    log_var = torch.full_like(mean, log_var_val)
    err_t = torch.full_like(mean, err)

    _, comps = sr1_deblend_loss(mean, log_var, x_high, err_t)
    expected = float(np.exp(log_var_val) + err**2)
    assert comps["total_var_p50"] == pytest.approx(expected, rel=1e-5)


def test_sr1_loss_responds_to_measurement_error():
    """Sanity: the reference uncertainty must actually enter the loss."""
    x_high, _ = _spectrum_with_lines()
    mean = x_high + 0.5
    log_var = torch.full_like(mean, -2.0)
    a, _ = sr1_deblend_loss(mean, log_var, x_high, torch.full_like(mean, 0.01))
    b, _ = sr1_deblend_loss(mean, log_var, x_high, torch.full_like(mean, 1.0))
    assert not torch.allclose(a, b)


def test_sr1_loss_is_differentiable():
    x_high, _ = _spectrum_with_lines()
    mean = x_high.clone().requires_grad_(True)
    loss, _ = sr1_deblend_loss(mean, torch.full_like(x_high, -2.0), x_high,
                               torch.full_like(x_high, 0.1))
    loss.backward()
    assert mean.grad is not None and torch.isfinite(mean.grad).all()


# --------------------------------------------------------------------------
# SR2 loss
# --------------------------------------------------------------------------


def _sr2_kwargs(**over):
    K = 98
    kw = dict(
        sr2_mean=torch.randn(B, 1, L),
        sr2_logvar=torch.full((B, 1, L), -2.0),
        x_high=torch.randn(B, 1, L),
        x_high_err=torch.full((B, 1, L), 0.1),
        line_mask=(torch.rand(B, 1, L) > 0.7).float(),
        presence=torch.rand(B, K),
    )
    kw.update(over)
    return kw


def test_sr2_loss_runs_and_reports_components():
    loss, comps = sr2_loss(lam_hp_in=0.5, lam_hp_out=0.3, lam_sparse=0.01, **_sr2_kwargs())
    assert torch.isfinite(loss)
    for key in ("nll", "hp_in", "hp_out", "presence_mean", "loss_total"):
        assert key in comps


def test_sr2_presence_sparsity_penalises_switching_lines_on():
    """Without this term the line branch turns every line on weakly."""
    base = _sr2_kwargs(presence=torch.full((B, 98), 0.05))
    dense = dict(base, presence=torch.full((B, 98), 0.95))
    lo, _ = sr2_loss(lam_sparse=1.0, **base)
    hi, _ = sr2_loss(lam_sparse=1.0, **dense)
    assert hi > lo
    # With the weight off, presence must not affect the loss at all.
    lo0, _ = sr2_loss(lam_sparse=0.0, **base)
    hi0, _ = sr2_loss(lam_sparse=0.0, **dense)
    assert torch.allclose(lo0, hi0)


# --------------------------------------------------------------------------
# per-line windows, presence supervision and flux matching
# --------------------------------------------------------------------------


# Long enough to hold well-separated lines with 20-sample sidebands around
# each, which the 512-sample fixture above cannot.
LL = 2048


def _clean_lines(centres, amp=20.0, sigma=3.0, continuum=0.0):
    """A flat spectrum with Gaussian emission lines at the given samples."""
    x = torch.full((1, 1, LL), float(continuum))
    grid = torch.arange(LL, dtype=torch.float32)
    for c in centres:
        x[0, 0] += amp * torch.exp(-0.5 * ((grid - c) / sigma) ** 2)
    return x


def _line_kwargs(x_high, positions, **over):
    kw = dict(
        sr2_mean=x_high.clone(), sr2_logvar=torch.full((1, 1, LL), -2.0),
        x_high=x_high, x_high_err=torch.full((1, 1, LL), 0.1),
        line_mask=torch.zeros(1, 1, LL), line_positions=positions,
        lam_hp_in=0.0, lam_hp_out=0.0, lam_presence=0.0, lam_flux=0.0,
    )
    kw.update(over)
    return kw


def test_line_window_flux_is_invariant_to_an_additive_continuum():
    """The sideband subtraction is what makes SR/HR ratios meaningful in
    normalised units, where each spectrum carries its own additive offset."""
    pos = torch.tensor([[300.0, 900.0]])
    flat = _clean_lines([300, 900], continuum=0.0)
    offset = _clean_lines([300, 900], continuum=7.5)
    assert torch.allclose(line_window_flux(flat, pos),
                          line_window_flux(offset, pos), atol=1e-3)


def test_line_window_flux_scales_with_line_amplitude():
    pos = torch.tensor([[500.0]])
    weak = line_window_flux(_clean_lines([500], amp=10.0), pos)
    strong = line_window_flux(_clean_lines([500], amp=30.0), pos)
    assert torch.allclose(strong / weak, torch.tensor(3.0), rtol=1e-3)


def test_sideband_median_tolerates_one_narrow_neighbour():
    """A median over both sidebands is robust to a *single* intruder, which is
    why isolated-ish lines measure fine. Pinned so the failure below is
    understood as specific to two-sided contamination, not generic."""
    isolated = _clean_lines([500], amp=20.0, sigma=2.0)
    blended = isolated + _clean_lines([516], amp=20.0, sigma=2.0)
    pos = torch.tensor([[500.0]])
    f_iso = line_window_flux(isolated, pos)
    f_one = line_window_flux(blended, pos)
    assert f_one == pytest.approx(f_iso, rel=0.05)


def test_two_sided_contamination_defeats_the_sideband_median():
    """The Halpha geometry: neighbours on *both* sides, covering most of the
    sideband, so more than half the samples are line rather than continuum and
    the median moves with them."""
    isolated = _clean_lines([500], amp=20.0, sigma=2.0)
    crowded = (isolated
               + _clean_lines([486], amp=30.0, sigma=2.0)
               + _clean_lines([514], amp=30.0, sigma=2.0)
               + _clean_lines([482], amp=30.0, sigma=2.0)
               + _clean_lines([518], amp=30.0, sigma=2.0))
    pos = torch.tensor([[500.0]])
    f_iso = line_window_flux(isolated, pos)
    f_bad = line_window_flux(crowded, pos)
    assert f_bad < 0.9 * f_iso, "a blanketed sideband must bias the continuum high"


def test_line_free_continuum_recovers_a_crowded_line():
    """The fix keeps blended lines in the objective rather than dropping them:
    deblending is the capability being trained, so the answer is a better
    continuum, not a shorter line list. With the neighbours known, the estimate
    moves out to samples that are actually continuum."""
    isolated = _clean_lines([500], amp=20.0, sigma=2.0)
    crowded = (isolated
               + _clean_lines([486], amp=30.0, sigma=2.0)
               + _clean_lines([514], amp=30.0, sigma=2.0)
               + _clean_lines([482], amp=30.0, sigma=2.0)
               + _clean_lines([518], amp=30.0, sigma=2.0))
    pos = torch.tensor([[500.0]])
    allpos = torch.tensor([[500.0, 486.0, 514.0, 482.0, 518.0]])

    f_iso = line_window_flux(isolated, pos, sb_hi=30)
    f_fixed = line_window_flux(crowded, pos, sb_hi=30,
                               all_positions=allpos, clean_half=7)
    assert f_fixed == pytest.approx(f_iso, rel=0.05)


def test_line_free_continuum_falls_back_when_nothing_is_clean():
    """11 of the 98 catalogued lines have a neighbour inside both sidebands --
    Halpha among them. With no clean sample available the estimate must degrade
    to the plain median, not to NaN."""
    x = _clean_lines([500], amp=20.0)
    pos = torch.tensor([[500.0]])
    # Neighbours on both sides, blanketing the whole sideband.
    allpos = torch.tensor([[500.0, 486.0, 514.0, 474.0, 526.0]])
    f = line_window_flux(x, pos, all_positions=allpos, clean_half=12)
    assert torch.isfinite(f).all()
    assert f == pytest.approx(line_window_flux(x, pos), rel=1e-5)


def test_presence_target_marks_real_lines_and_only_those():
    x = _clean_lines([400, 1600])
    pos = torch.tensor([[400.0, 1000.0, 1600.0]])
    tgt, w = line_presence_target(x, pos)
    assert tgt[0, 0] == 1.0 and tgt[0, 2] == 1.0, "injected lines must be positive"
    assert tgt[0, 1] == 0.0, "a position with no line must be negative"
    assert (w == 1.0).all()


def _log_grid_line(centre=1000.0, amp=30.0, sigma=1.7, seed=0):
    """One emission line at the width the *log* grid actually produces.

    R=1000 gratings on the R=4000 grid give sigma ~1.7 samples, against the
    sigma=3.0 the other fixtures here use. That difference is the whole bug:
    a wider synthetic line clears the mask's width filter and hides it.
    """
    torch.manual_seed(seed)
    x = torch.randn(1, 1, LL)
    grid = torch.arange(LL, dtype=torch.float32)
    x[0, 0] += amp * torch.exp(-0.5 * ((grid - centre) / sigma) ** 2)
    return x


def test_presence_target_min_width_rejects_real_log_grid_lines():
    """The old-grid ``min_width=7`` misses a line of the width this grid makes.

    ``keep_only_wide`` demands ``min_width`` *consecutive* supra-threshold
    samples. At sigma ~1.7 a line clears 7.5 sigma over only ~5-6 of them, so 7
    rejects it outright. Measured on 1,203 real lines at HR SNR >= 20 this was
    the difference between flagging 46% and 74% of them -- and an unflagged line
    is labelled *absent* for the presence gate and dropped from the flux term.
    """
    x = _log_grid_line()
    pos = torch.tensor([[1000.0]])
    common = dict(core_half=7, smooth_k=259, thresh_mad=7.5, dilate=11)

    old, _ = line_presence_target(x, pos, min_width=7, **common)
    new, _ = line_presence_target(x, pos, min_width=5, **common)
    assert old[0, 0] == 0.0, "min_width=7 is expected to miss this line"
    assert new[0, 0] == 1.0, "min_width=5 must find it"

    # Not the threshold's fault: the line is far above 7.5 sigma per pixel.
    hp = x - smooth1d(x, k=259)
    mad = robust_mad(hp, dim=-1)
    assert float((hp.abs() / mad).max()) > 15.0


def test_sr2_loss_forwards_the_presence_mask_settings():
    """The regression: ``sr2_loss`` used to let these default to the retired
    linear grid's values, so the mask that labels presence *and* selects the
    flux term's lines was built with the wrong kernel sizes."""
    x = _log_grid_line()
    pos = torch.tensor([[1000.0]])
    # A prediction that under-emits the line, so there is a flux error to see.
    # `w_flux = w_line * tgt`, so a line the mask fails to flag carries zero
    # weight and the flux term drops it entirely -- which is the consequence
    # that matters, and unlike the presence BCE it is not degenerate for a
    # single line (BCE at logit 0 is ln 2 whichever way the label falls).
    under = x.clone()
    grid = torch.arange(LL, dtype=torch.float32)
    under[0, 0] -= 0.6 * 30.0 * torch.exp(-0.5 * ((grid - 1000.0) / 1.7) ** 2)

    kw = dict(sr2_mean=under, presence=torch.full((1, 1), 0.5), lam_flux=1.0,
              line_in_range=torch.ones(1, 1, dtype=torch.bool))

    _, found = sr2_loss(**_line_kwargs(x, pos, **kw), presence_mask_min_width=5)
    _, missed = sr2_loss(**_line_kwargs(x, pos, **kw), presence_mask_min_width=7)

    assert found["flux_loss"] > 0.0, "a flagged line must enter the flux term"
    assert missed["flux_loss"] == 0.0, "an unflagged line must not"
    assert found["flux_loss"] != missed["flux_loss"], (
        "min_width did not reach line_presence_target -- the parameter is not "
        "plumbed through sr2_loss"
    )


def test_presence_target_refuses_to_judge_lines_it_cannot_see():
    """Out-of-range and unmeasured lines must be excluded, not scored as
    negatives -- otherwise the head learns to switch off lines that merely fall
    off the edge of the detector."""
    x = _clean_lines([400])
    pos = torch.tensor([[400.0, 1500.0]])
    _, w = line_presence_target(x, pos, in_range=torch.tensor([[True, False]]))
    assert w[0, 0] == 1.0 and w[0, 1] == 0.0

    valid = torch.ones(1, 1, LL)
    valid[0, 0, 1450:1550] = 0.0
    _, w2 = line_presence_target(x, pos, valid=valid)
    assert w2[0, 0] == 1.0 and w2[0, 1] == 0.0


def test_presence_bce_pushes_real_lines_on_and_absent_lines_off():
    """The regression that matters: the old blanket penalty drove presence to a
    constant, identical on real and absent lines. The supervised term must move
    the two in opposite directions."""
    x = _clean_lines([400, 1600])
    pos = torch.tensor([[400.0, 1000.0, 1600.0]])
    logit = torch.zeros(1, 3, requires_grad=True)

    loss, comps = sr2_loss(**_line_kwargs(
        x, pos, presence=torch.sigmoid(logit), presence_logit=logit,
        lam_presence=1.0))
    loss.backward()
    # Negative gradient on a logit means the step raises it.
    assert logit.grad[0, 0] < 0 and logit.grad[0, 2] < 0, "real lines must be pushed on"
    assert logit.grad[0, 1] > 0, "the absent line must be pushed off"
    assert comps["presence_bce"] > 0


def test_presence_positives_are_upweighted_to_parity():
    """Real lines are ~3% of catalogued positions; unweighted BCE has the same
    fixed point as the penalty it replaces."""
    x = _clean_lines([400])
    pos = torch.tensor([[400.0] + [700.0 + 45 * i for i in range(30)]])
    logit = torch.zeros(1, 31, requires_grad=True)
    _, comps = sr2_loss(**_line_kwargs(
        x, pos, presence=torch.sigmoid(logit), presence_logit=logit,
        lam_presence=1.0))
    assert comps["presence_pos_weight"] == pytest.approx(30.0, rel=0.1)


def test_flux_term_penalises_an_under_predicted_line():
    """SR2's actual failure: the line is in the right place at the wrong
    amplitude. The likelihood barely notices; this term must."""
    hr = _clean_lines([500], amp=20.0)
    pos = torch.tensor([[500.0]])
    kw = _line_kwargs(hr, pos, presence=torch.full((1, 1), 0.5), lam_flux=1.0)
    exact, c_exact = sr2_loss(**kw)
    starved, c_starved = sr2_loss(**dict(kw, sr2_mean=_clean_lines([500], amp=2.0)))
    assert c_starved["flux_loss"] > c_exact["flux_loss"]
    assert c_exact["flux_ratio_median"] == pytest.approx(1.0, abs=1e-3)
    assert c_starved["flux_ratio_median"] == pytest.approx(0.1, abs=0.02)


def test_flux_term_gradient_raises_a_starved_line():
    hr = _clean_lines([500], amp=20.0)
    sr = _clean_lines([500], amp=2.0).requires_grad_(True)
    loss, _ = sr2_loss(**_line_kwargs(
        hr, torch.tensor([[500.0]]), sr2_mean=sr,
        presence=torch.full((1, 1), 0.5), lam_flux=1.0))
    loss.backward()
    # Adding flux at the core must lower the loss.
    assert sr.grad[0, 0, 500] < 0


def test_flux_term_brightness_weighting_follows_the_bright_lines():
    """The reported flux number is computed on bright lines, but mask positives
    are mostly faint ones. With power 1 the term must care more about getting a
    bright line wrong than a faint one."""
    hr = _clean_lines([400], amp=60.0) + _clean_lines([1200], amp=6.0)
    pos = torch.tensor([[400.0, 1200.0]])
    bright_starved = _clean_lines([400], amp=6.0) + _clean_lines([1200], amp=6.0)
    faint_starved = _clean_lines([400], amp=60.0) + _clean_lines([1200], amp=0.6)

    def flux_loss(sr, power):
        _, c = sr2_loss(**_line_kwargs(
            hr, pos, sr2_mean=sr, presence=torch.full((1, 2), 0.5),
            lam_flux=1.0, flux_weight_power=power))
        return c["flux_loss"]

    # Weighted: starving the bright line is the worse error.
    assert flux_loss(bright_starved, 1.0) > flux_loss(faint_starved, 1.0)
    # Unweighted: the two count equally, so the ordering does not hold.
    assert flux_loss(bright_starved, 0.0) == pytest.approx(
        flux_loss(faint_starved, 0.0), rel=0.2)


def test_new_line_terms_are_inert_without_line_positions():
    """Both weights on but no positions must not silently contribute."""
    kw = _sr2_kwargs()
    off, c_off = sr2_loss(lam_presence=0.0, lam_flux=0.0, **kw)
    on, c_on = sr2_loss(lam_presence=2.0, lam_flux=2.0, **kw)
    assert torch.allclose(off, on)
    assert c_on["presence_bce"] == 0.0 and c_on["flux_loss"] == 0.0


def test_sr2_highpass_terms_are_separately_controllable():
    """Inside and outside the line mask want opposite behaviour, so the two
    weights must have independent effect."""
    kw = _sr2_kwargs()
    none, _ = sr2_loss(lam_hp_in=0.0, lam_hp_out=0.0, **kw)
    only_in, _ = sr2_loss(lam_hp_in=1.0, lam_hp_out=0.0, **kw)
    only_out, _ = sr2_loss(lam_hp_in=0.0, lam_hp_out=1.0, **kw)
    assert only_in > none
    assert only_out > none
    assert not torch.allclose(only_in, only_out)


def test_sr2_redshift_term_is_inert_without_its_dependencies():
    """lam_z > 0 but no zhead/ztransform must not silently contribute."""
    kw = _sr2_kwargs()
    a, ca = sr2_loss(lam_z=0.0, **kw)
    b, cb = sr2_loss(lam_z=5.0, **kw)
    assert torch.allclose(a, b)
    assert ca["z_loss"] == 0.0 and cb["z_loss"] == 0.0


def test_sr2_loss_is_differentiable():
    kw = _sr2_kwargs()
    kw["sr2_mean"] = kw["sr2_mean"].requires_grad_(True)
    loss, _ = sr2_loss(lam_hp_in=0.5, lam_hp_out=0.3, lam_sparse=0.01, **kw)
    loss.backward()
    assert torch.isfinite(kw["sr2_mean"].grad).all()


# --------------------------------------------------------------------------
# validity masking
#
# The reference has ~1% of samples that no detector measured, concentrated at
# grating edges. The original pipeline filled them with the per-spectrum median,
# which trains the model to emit a flat continuum there. These tests pin that
# such samples now contribute nothing instead.
# --------------------------------------------------------------------------


def test_masked_mean_ignores_invalid_samples():
    from specsr.training.losses import masked_mean

    x = torch.tensor([[1.0, 2.0, 1000.0, 4.0]])
    valid = torch.tensor([[True, True, False, True]])
    assert masked_mean(x, valid).item() == pytest.approx((1 + 2 + 4) / 3)
    assert masked_mean(x, None).item() == pytest.approx(x.mean().item())


def test_masked_mean_survives_an_all_invalid_spectrum():
    """Must not produce NaN, and must keep the graph alive."""
    from specsr.training.losses import masked_mean

    x = torch.randn(2, 1, 16, requires_grad=True)
    out = masked_mean(x, torch.zeros(2, 1, 16, dtype=torch.bool))
    assert torch.isfinite(out)
    out.backward()
    assert torch.isfinite(x.grad).all()


def test_sr1_likelihood_is_exactly_invariant_to_unmeasured_samples():
    """The likelihood term must not see unmeasured wavelengths at all."""
    x_high, _ = _spectrum_with_lines()
    mean, log_var = x_high.clone(), torch.full_like(x_high, -2.0)
    err = torch.full_like(x_high, 0.1)
    valid = torch.ones_like(x_high, dtype=torch.bool)
    valid[..., 200:260] = False

    _, ref = sr1_deblend_loss(mean, log_var, x_high, err, valid=valid)
    corrupted = x_high.clone()
    corrupted[..., 200:260] = 1e3
    _, got = sr1_deblend_loss(mean, log_var, corrupted, err, valid=valid)

    assert got["loss_base_nll"] == pytest.approx(ref["loss_base_nll"], rel=1e-9)
    assert got["resid_rms"] == pytest.approx(ref["resid_rms"], rel=1e-6)


def test_sr1_total_loss_invariant_under_the_documented_neutral_fill():
    """The contract is: neutral value in invalid samples AND a mask.

    The line mask is derived by smoothing x_high over a wide kernel, so an
    arbitrary value at an unmeasured wavelength leaks into neighbouring measured
    samples and is mistaken for a line. With the neutral fill the caller is
    required to supply, the whole loss is stable.
    """
    x_high, _ = _spectrum_with_lines()
    mean, log_var = x_high.clone(), torch.full_like(x_high, -2.0)
    err = torch.full_like(x_high, 0.1)
    valid = torch.ones_like(x_high, dtype=torch.bool)
    valid[..., 200:260] = False

    a = x_high.clone()
    a[..., 200:260] = 0.0        # neutral fill
    b = x_high.clone()
    b[..., 200:260] = 0.0
    b[..., 210:220] = 0.0                             # still neutral, different span
    la, _ = sr1_deblend_loss(mean, log_var, a, err, valid=valid)
    lb, _ = sr1_deblend_loss(mean, log_var, b, err, valid=valid)
    assert la.item() == pytest.approx(lb.item(), rel=1e-6)


def test_sr1_unfilled_garbage_does_contaminate_the_line_mask():
    """Regression guard on *why* the neutral fill is required.

    If this ever stops being true the fill requirement can be relaxed; until
    then, silently dropping it would corrupt the line mask.
    """
    x_high, _ = _spectrum_with_lines()
    mean, log_var = x_high.clone(), torch.full_like(x_high, -2.0)
    err = torch.full_like(x_high, 0.1)
    valid = torch.ones_like(x_high, dtype=torch.bool)
    valid[..., 200:260] = False

    _, clean = sr1_deblend_loss(mean, log_var, x_high, err, valid=valid)
    wild = x_high.clone()
    wild[..., 200:260] = 1e3
    _, dirty = sr1_deblend_loss(mean, log_var, wild, err, valid=valid)
    assert dirty["mask_frac_mean"] > clean["mask_frac_mean"]


def test_sr1_line_mask_cannot_fire_on_unmeasured_samples():
    """A line cannot be 'detected' at a wavelength nobody observed."""
    x_high, centers = _spectrum_with_lines()
    valid = torch.ones_like(x_high, dtype=torch.bool)
    c = int(centers[0])
    valid[..., c - 10 : c + 10] = False

    _, comps = sr1_deblend_loss(
        x_high, torch.full_like(x_high, -2.0), x_high,
        torch.full_like(x_high, 0.1), valid=valid,
    )
    _, comps_all = sr1_deblend_loss(
        x_high, torch.full_like(x_high, -2.0), x_high, torch.full_like(x_high, 0.1)
    )
    assert comps["mask_frac_mean"] < comps_all["mask_frac_mean"]


def test_sr2_likelihood_is_exactly_invariant_to_unmeasured_samples():
    kw = _sr2_kwargs()
    valid = torch.ones_like(kw["x_high"], dtype=torch.bool)
    valid[..., 100:180] = False

    _, ref = sr2_loss(lam_hp_in=0.0, lam_hp_out=0.0, valid=valid, **kw)
    kw2 = dict(kw)
    kw2["x_high"] = kw["x_high"].clone()
    kw2["x_high"][..., 100:180] = 1e3
    _, got = sr2_loss(lam_hp_in=0.0, lam_hp_out=0.0, valid=valid, **kw2)

    assert got["nll"] == pytest.approx(ref["nll"], rel=1e-9)


def test_sr2_highpass_weights_drop_unmeasured_samples():
    """Invalid samples must fall out of both numerator and denominator of the
    inverse-variance weighted high-pass terms."""
    kw = _sr2_kwargs()
    valid = torch.ones_like(kw["x_high"], dtype=torch.bool)
    valid[..., 100:180] = False
    _, masked = sr2_loss(lam_hp_in=1.0, lam_hp_out=1.0, valid=valid, **kw)
    _, unmasked = sr2_loss(lam_hp_in=1.0, lam_hp_out=1.0, **kw)
    assert masked["hp_in"] != unmasked["hp_in"]
    assert np.isfinite(masked["hp_in"]) and np.isfinite(masked["hp_out"])


def test_masking_is_opt_in_and_changes_nothing_when_absent():
    """Existing call sites must behave exactly as before."""
    x_high, _ = _spectrum_with_lines()
    a, _ = sr1_deblend_loss(x_high, torch.full_like(x_high, -2.0), x_high,
                            torch.full_like(x_high, 0.1))
    b, _ = sr1_deblend_loss(x_high, torch.full_like(x_high, -2.0), x_high,
                            torch.full_like(x_high, 0.1), valid=None)
    assert a.item() == pytest.approx(b.item())


# --------------------------------------------------------------------------
# Gradient guard
# --------------------------------------------------------------------------


def test_clip_grad_or_skip_rejects_nonfinite_gradients():
    """A non-finite gradient must be reported as unsafe, not laundered into NaN.

    `clip_grad_norm_` alone scales by `max_norm / inf == 0`, and `inf * 0` is
    NaN -- so clipping an overflowed gradient silently poisons every weight the
    optimiser touches. This is what destroyed the 2026-07-30 SR2 fine-tune.
    """
    import torch

    from specsr.training.common import clip_grad_or_skip

    lin = torch.nn.Linear(3, 1)

    lin.weight.grad = torch.tensor([[1.0, 2.0, 3.0]])
    norm, ok = clip_grad_or_skip(lin.parameters(), 0.5)
    assert ok and norm > 0
    assert torch.isfinite(lin.weight.grad).all()

    for bad in (float("inf"), float("nan")):
        lin.weight.grad = torch.tensor([[bad, 1.0, 2.0]])
        _, ok = clip_grad_or_skip(lin.parameters(), 0.5)
        assert not ok, f"{bad} gradient was not reported as unsafe"


def test_bare_clip_grad_norm_would_have_produced_nan():
    """Pin the PyTorch behaviour the guard exists to defend against."""
    import torch

    lin = torch.nn.Linear(2, 1)
    lin.weight.grad = torch.tensor([[float("inf"), 1.0]])
    torch.nn.utils.clip_grad_norm_(lin.parameters(), 0.5)
    assert torch.isnan(lin.weight.grad).any(), (
        "clip_grad_norm_ no longer turns inf into NaN; the guard's rationale "
        "should be re-checked against this PyTorch version"
    )
