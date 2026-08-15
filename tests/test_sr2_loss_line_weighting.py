"""The in-line high-pass term must not be inverse-variance weighted.

This is a regression test for the presence-gate collapse. ``sr2_loss`` weights
the *out-of-line* high-pass term by ``1/sigma^2``, which is correct, and for a
while it weighted the *in-line* term the same way, which is not.

The mechanism is a comparison *within* the line mask, not between lines and
continuum. ``hp_in`` is a weighted mean over masked pixels only, so a uniform
rescaling of those weights cancels exactly and the continuum's weight never
enters. What does not cancel is the spread inside the mask: photon noise scales
with flux, so the brightest line pixels are the noisiest, and ``1/sigma^2``
drives their weight towards zero. Measured on the 286-spectrum validation split,
under that weighting

* the brightest 10% of in-line pixels receive **2.1%** of the total in-line weight,
* the brightest 50% receive **6.7%**,
* the faintest 10% are weighted **34x** more heavily than the brightest 10%.

So the term that exists to fit emission lines was determined almost entirely by
the faintest pixels in the mask and was nearly blind to the bright cores that
carry the flux. The line branch had little incentive to reconstruct them, and
the presence gate decayed into an absorbing state.

The failure is silent -- training runs to completion and reports a plausible
MSE -- so it needs a test rather than a comment.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from specsr.training.losses import sr2_loss  # noqa: E402

BRIGHT, FAINT = 2000.0, 2.0  # peak flux over a continuum of 1.0

# Wide enough to survive the hp_k=51 smoothing of the weights, which is the
# regime the real data is in: `make_line_mask_from_smoothed` dilates by 11
# samples, so masked line regions are a couple of dozen samples across even
# though the lines themselves are barely resolved. A narrow synthetic line has
# its weight contrast averaged away by that smoothing and the effect vanishes --
# an earlier version of this test used width 1.5 and passed with the bug in
# place, which is how the too-narrow-line trap was found.
LINE_WIDTH = 12.0


def _inputs(*, miss_bright: float = 0.0, miss_faint: float = 0.0):
    """A spectrum with one bright and one faint emission line.

    Photon noise: ``sigma = sqrt(floor + flux)``, so the bright line is the
    noisiest thing in the spectrum and inverse-variance weighting gives its
    pixels almost no weight. Measured on the real validation split, under that
    weighting the brightest 10% of in-line pixels receive **2.1%** of the total
    in-line weight, and the faintest 10% receive 34x as much as the brightest --
    the term is decided almost entirely by the faintest pixels in the mask.

    ``miss_bright``/``miss_faint`` set the fraction of each line the prediction
    fails to reproduce, so a caller can ask what missing each one costs.
    """
    B, L = 2, 512
    x = torch.arange(L, dtype=torch.float32)

    def line(centre, peak, width=LINE_WIDTH):
        return peak * torch.exp(-0.5 * ((x - centre) / width) ** 2)

    bright = line(150.0, BRIGHT)[None, None, :].expand(B, 1, L)
    faint = line(350.0, FAINT)[None, None, :].expand(B, 1, L)

    x_high = 1.0 + bright + faint
    err = torch.sqrt(0.01 + x_high)
    line_mask = ((bright + faint) > 0.05).float()

    sr2_mean = x_high - miss_bright * bright - miss_faint * faint

    return dict(
        sr2_mean=sr2_mean.clone().requires_grad_(True),
        sr2_logvar=torch.zeros(B, 1, L),
        x_high=x_high,
        x_high_err=err,
        line_mask=line_mask,
        presence=torch.full((B, 8), 0.5),
    )


def test_missing_a_bright_line_costs_more_than_missing_a_faint_one():
    """The term must not be blind to the brightest lines in the spectrum.

    This is the property the collapse violated. The bright line carries 1000x
    the peak flux of the faint one, so failing to reproduce it should dominate
    the term. Inverse-variance weighting pulls it back towards the faint line:
    photon noise makes the bright line the noisiest region, so ``1/sigma^2``
    drives its weight down and the faint line gets a say out of all proportion
    to its flux. The line branch is then under less pressure to reconstruct
    exactly the features it exists for.
    """
    def cost_ratio(**kw):
        _, bright = sr2_loss(**_inputs(miss_bright=1.0), lam_hp_in=1.0, **kw)
        _, faint = sr2_loss(**_inputs(miss_faint=1.0), lam_hp_in=1.0, **kw)
        return bright["hp_in"] / faint["hp_in"]

    default = cost_ratio()
    weighted = cost_ratio(hp_in_noise_weighted=True)

    # Compare against the weighted configuration rather than a fixed threshold:
    # the absolute ratio depends on the synthetic line profile, but the
    # *suppression* caused by the weighting is the thing under test.
    assert default > 2.0 * weighted, (
        f"missing the bright line costs {default:.0f}x missing the faint one by "
        f"default, against {weighted:.0f}x with inverse-variance weighting "
        "explicitly on. These are too close: the default is noise weighting the "
        "in-line term, so it is discounting exactly the pixels it exists to fit "
        "-- this is what starved the line branch and collapsed the presence gate."
    )


def test_noise_weighting_is_available_but_off_by_default():
    """The old behaviour stays reachable for the ablation, but must not be the default."""
    _, default = sr2_loss(**_inputs(miss_bright=1.0), lam_hp_in=1.0)
    _, weighted = sr2_loss(**_inputs(miss_bright=1.0), lam_hp_in=1.0,
                           hp_in_noise_weighted=True)

    assert default["hp_in"] != pytest.approx(weighted["hp_in"], rel=1e-9), (
        "hp_in_noise_weighted=True changed nothing; the flag is not wired up"
    )


def test_valid_still_gates_the_in_line_term_without_noise_weighting():
    """``valid`` must gate the in-line term on its own.

    The validity mask used to be folded into the inverse-variance weights, so
    with ``hp_in_noise_weighted=False`` the in-line weight degenerated to
    ``line_mask * 1.0`` and ``valid`` stopped having any effect at all —
    quietly readmitting wavelengths the detector never recorded.

    Note this checks only that ``valid`` is *consulted*. It cannot check that
    unmeasured samples have no influence whatsoever: ``highpass`` smooths over
    ``hp_k`` samples, so whatever sits at a masked wavelength bleeds into its
    measured neighbours regardless. That is why the caller is required to fill
    invalid samples with a neutral value as well as passing ``valid`` — masking
    alone is not sufficient, and neither is filling.
    """
    base = _inputs(miss_bright=1.0, miss_faint=1.0)

    valid = torch.ones_like(base["line_mask"])
    valid[:, :, 300:] = 0.0  # blind the faint line entirely

    _, all_valid = sr2_loss(**base, lam_hp_in=1.0)
    _, half_blind = sr2_loss(**base, valid=valid, lam_hp_in=1.0)

    assert half_blind["hp_in"] != pytest.approx(all_valid["hp_in"], rel=1e-9), (
        "blinding half the line did not change the in-line high-pass term, so "
        "`valid` is being ignored -- the validity mask is riding on the "
        "inverse-variance weights again"
    )


def test_redshift_term_does_not_shadow_the_validity_mask():
    """The z term must not clobber the caller's per-wavelength ``valid`` mask.

    ``sr2_loss`` builds a per-*sample* finite-value mask while computing the
    redshift term. That local was once called ``valid``, shadowing the
    per-*wavelength* parameter of the same name, which is shaped ``(B, 1, L)``
    rather than ``(B,)``. The reported NLL was then computed against the wrong
    mask -- and with ``lam_z > 0`` the shapes disagree and it raises outright,
    which is how it was found: no caller exercised this path until the SR2
    training loop moved into the package.
    """
    inp = _inputs(miss_bright=1.0)
    B, L = inp["x_high"].shape[0], inp["x_high"].shape[-1]

    class _Head(torch.nn.Module):
        def forward(self, x):
            b = x.shape[0]
            return torch.zeros(b, 1), torch.zeros(b, 1)

    from specsr.training.ztransform import RedshiftTransform

    loss, m = sr2_loss(
        **inp,
        valid=torch.ones(B, 1, L),
        lam_hp_in=1.0,
        lam_z=1.0,
        zhead=_Head(),
        z_true=torch.full((B,), 3.5),
        ztransform=RedshiftTransform(mean=3.3, std=1.9, z_min_n=-1.8, z_max_n=6.2),
    )
    assert torch.isfinite(loss)
    assert np.isfinite(m["nll"]), "NLL was computed against the wrong mask"
