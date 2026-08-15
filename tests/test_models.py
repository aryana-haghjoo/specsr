"""Architecture tests for the three pipeline stages.

These run without any weights or data. Tests that need the published
checkpoints are marked ``needs_weights`` and skip unless a local checkpoint
directory is available.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from specsr.models import (
    LINE_LIST_REST_AA,
    SR2Attention,
    SuperRes1D,
    ZHead1D,
    get_activation,
    line_wavelengths_um,
)
from specsr.models.blocks import largest_divisor_at_most

L = 512  # short grid keeps these tests fast


# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "channels,groups,expected",
    [(96, 8, 8), (108, 8, 6), (64, 8, 8), (7, 8, 7), (13, 8, 1)],
)
def test_group_count_divides_channels(channels, groups, expected):
    """GroupNorm needs channels % groups == 0; the helper must guarantee it."""
    g = largest_divisor_at_most(channels, groups)
    assert g == expected
    assert channels % g == 0


def test_get_activation_known_and_unknown():
    assert isinstance(get_activation("elu"), torch.nn.ELU)
    assert isinstance(get_activation("GELU"), torch.nn.GELU)  # case-insensitive
    with pytest.raises(ValueError, match="Unsupported activation"):
        get_activation("swish")


# --------------------------------------------------------------------------
# SR1
# --------------------------------------------------------------------------


def test_sr1_shapes_and_uncertainty_init():
    m = SuperRes1D(in_channels=1, hidden_dim=32, num_res_blocks=2)
    x = torch.randn(3, 1, L)
    mean, log_var = m(x)
    assert mean.shape == (3, 1, L)
    assert log_var.shape == (3, 1, L)
    # Starts at a small, uniform predicted uncertainty rather than data-dependent.
    assert torch.allclose(log_var, torch.full_like(log_var, -2.0), atol=1e-5)


def test_sr1_odd_hidden_dim_does_not_crash():
    """Sweeps produce widths that are not multiples of 8."""
    m = SuperRes1D(hidden_dim=108, num_res_blocks=2)
    assert m(torch.randn(2, 1, L))[0].shape == (2, 1, L)


# --------------------------------------------------------------------------
# ZHead
# --------------------------------------------------------------------------


@pytest.mark.parametrize("in_channels", [1, 2])
def test_zhead_shapes(in_channels):
    z = ZHead1D(in_channels=in_channels, hidden_dim=16, num_blocks=2)
    mu, log_var = z(torch.randn(4, in_channels, L))
    assert mu.shape == (4,)
    assert log_var.shape == (4,)


def test_zhead_confidence_pooling_downweights_uncertain_pixels():
    """With the uncertainty channel, noisy regions should influence the
    prediction less than well-measured ones."""
    z = ZHead1D(in_channels=2, hidden_dim=16, num_blocks=2).eval()
    flux = torch.randn(1, 1, L)

    low_sigma = torch.full((1, 1, L), -3.0)
    high_sigma = torch.full((1, 1, L), -3.0)
    # Make the second half of the spectrum very uncertain.
    high_sigma[..., L // 2 :] = 3.0

    with torch.no_grad():
        mu_ref, _ = z(torch.cat([flux, low_sigma], dim=1))
        mu_masked, _ = z(torch.cat([flux, high_sigma], dim=1))
    # The two must differ: the uncertainty channel has to actually be used.
    assert not torch.allclose(mu_ref, mu_masked)


@pytest.mark.parametrize("in_channels", [1, 2])
def test_zhead_v2_shapes(in_channels):
    z = ZHead1D(in_channels=in_channels, hidden_dim=16, num_blocks=3,
                coord_channel=True, dilation_growth=2, pooling="attn")
    mu, log_var = z(torch.randn(4, in_channels, L))
    assert mu.shape == (4,)
    assert log_var.shape == (4,)


def test_zhead_v2_can_learn_line_position_and_v1_cannot():
    """v2 must be able to learn "where is the line" -- v1 provably cannot.

    Redshift on a fixed log-wavelength grid *is* line position, and v1's conv
    trunk plus global mean pool is translation-invariant, which is why the
    original three-arm comparison produced the impossible ordering
    hires < lowres. Train both heads briefly on a synthetic position-regression
    task: v2 must largely solve it, v1 must remain near chance.
    """

    def fit(model, steps=250):
        torch.manual_seed(1)
        n, bs = 256, 64
        grid = torch.arange(n, dtype=torch.float32)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(steps):
            c = torch.rand(bs) * (n - 60) + 30          # line centres
            y = (c / n) * 2 - 1                          # target in [-1, 1]
            spec = torch.exp(-0.5 * ((grid - c[:, None]) / 3.0) ** 2)
            mu, _ = model(spec.unsqueeze(1))
            loss = ((mu - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return float(loss.item())

    v2_loss = fit(ZHead1D(in_channels=1, hidden_dim=16, num_blocks=3,
                          coord_channel=True, dilation_growth=2, pooling="attn"))
    v1_loss = fit(ZHead1D(in_channels=1, hidden_dim=16, num_blocks=3))
    # Var of the uniform target is 1/3 ~ 0.33: v1 cannot beat predicting the
    # mean, v2 must get well below it.
    assert v2_loss < 0.05, f"v2 failed to learn line position (loss={v2_loss:.3f})"
    assert v1_loss > 0.15, f"v1 unexpectedly learned position (loss={v1_loss:.3f})"


def test_zhead_v2_rejects_v2_options_with_mean_pooling():
    with pytest.raises(ValueError):
        ZHead1D(in_channels=1, coord_channel=True, pooling="mean")


def test_zhead_v2_roundtrips_through_loader(tmp_path):
    """A v2 checkpoint saved in the trainer's format must reload identically,
    and a config without v2 keys must still build the v1 architecture."""
    from specsr.models.loaders import load_zhead

    cfg = {"hidden_dim": 16, "num_blocks": 3, "dropout": 0.0,
           "coord_channel": True, "dilation_growth": 2, "pooling": "attn"}
    z = ZHead1D(in_channels=2, hidden_dim=16, num_blocks=3, dropout=0.0,
                coord_channel=True, dilation_growth=2, pooling="attn").eval()
    path = tmp_path / "best_zhead_v2.pth"
    torch.save({"zhead_state_dict": z.state_dict(), "config": cfg,
                "use_sigma_channel": True, "z_mean": 3.0, "z_std": 1.5,
                "z_min_n": -2.0, "z_max_n": 4.0}, path)

    loaded, z_mean, z_std, use_sigma, _ = load_zhead(path)
    assert use_sigma and z_mean == 3.0 and z_std == 1.5
    xin = torch.randn(2, 2, L)
    with torch.no_grad():
        assert torch.allclose(z(xin)[0], loaded(xin)[0])


def _pdf_head(**kw):
    from specsr.models.zhead import ZHead1D as Z
    args = dict(in_channels=1, hidden_dim=16, num_blocks=3, coord_channel=True,
                dilation_growth=2, pooling="attn", head="softmax", n_z_bins=256,
                z_grid_min_n=-2.0, z_grid_max_n=4.0)
    args.update(kw)
    return Z(**args)


def test_zhead_pdf_head_shapes_and_range():
    z = _pdf_head()
    logits = z.logits(torch.randn(4, 1, L))
    assert logits.shape == (4, 256)
    mu, log_var = z(torch.randn(4, 1, L))
    assert mu.shape == (4,) and log_var.shape == (4,)
    # The estimate is read off the grid, so it cannot leave the observed range.
    assert bool((mu >= -2.0).all() and (mu <= 4.0).all())
    assert not z.bounded_mean, "a grid-based estimate must not be sigmoid-squashed again"


def test_zhead_pdf_point_estimate_picks_a_mode_not_the_valley():
    """The reason for a classification head, stated as a test.

    Given a bimodal posterior -- one line identification putting z near one
    value, a competing one near another -- a Gaussian head can only answer with
    a single mean, which falls between the two modes where the true redshift
    almost certainly is not. That is a catastrophic outlier by construction.
    The PDF head must instead answer with one of the modes.
    """
    z = _pdf_head(n_z_bins=401, z_grid_min_n=-2.0, z_grid_max_n=2.0)
    grid = z.z_grid_n
    lo, hi = -1.0, 1.0
    logits = (-0.5 * ((grid - lo) / 0.02) ** 2).exp() * 8.0 \
        + (-0.5 * ((grid - hi) / 0.02) ** 2).exp() * 7.0
    mu, log_var = z.moments(logits[None, :])
    mu = float(mu)

    assert min(abs(mu - lo), abs(mu - hi)) < 0.05, (
        f"point estimate {mu:.3f} is not on either mode ({lo}, {hi})"
    )
    assert abs(mu - 0.0) > 0.5, f"point estimate {mu:.3f} landed in the valley between modes"
    # The spread is taken from the whole PDF, so bimodality still reads as
    # "uncertain" downstream even though the estimate sits on one mode.
    assert float(log_var.exp().sqrt()) > 0.5


def test_zhead_pdf_loss_prefers_the_true_bin():
    from specsr.models.zhead import redshift_pdf_loss

    z = _pdf_head(n_z_bins=201, z_grid_min_n=-1.0, z_grid_max_n=1.0)
    grid = z.z_grid_n
    truth = torch.tensor([0.3])
    right = (-0.5 * ((grid - 0.3) / 0.02) ** 2).exp() * 10.0
    wrong = (-0.5 * ((grid + 0.6) / 0.02) ** 2).exp() * 10.0
    assert float(redshift_pdf_loss(right[None, :], truth, grid)) < \
        float(redshift_pdf_loss(wrong[None, :], truth, grid))


def test_zhead_pdf_head_roundtrips_through_loader(tmp_path):
    from specsr.models.loaders import load_zhead

    z = _pdf_head(in_channels=2).eval()
    cfg = {"hidden_dim": 16, "num_blocks": 3, "dropout": 0.1, "coord_channel": True,
           "dilation_growth": 2, "pooling": "attn", "head": "softmax", "n_z_bins": 256,
           "soft_argmax_half": 8}
    path = tmp_path / "best_zhead_pdf.pth"
    torch.save({"zhead_state_dict": z.state_dict(), "config": cfg,
                "use_sigma_channel": True, "z_mean": 3.0, "z_std": 1.5,
                "z_min_n": -2.0, "z_max_n": 4.0}, path)
    loaded, _, _, _, _ = load_zhead(path)
    assert not loaded.bounded_mean
    assert torch.allclose(loaded.z_grid_n, z.z_grid_n)
    xin = torch.randn(2, 2, L)
    with torch.no_grad():
        assert torch.allclose(z(xin)[0], loaded(xin)[0])


def test_zhead_pdf_head_requires_its_grid():
    from specsr.models.zhead import ZHead1D as Z
    with pytest.raises(ValueError, match="z_grid"):
        Z(in_channels=1, pooling="attn", coord_channel=True, head="softmax")



def test_line_mask_widens_with_redshift_uncertainty():
    """An unsure redshift must produce a vague mask, not a confident wrong one.

    The mask is a fixed 0.005 um wide, which asserts a redshift precision of
    dz ~ 0.01. Measured on validation, the head's median [O III] position error
    was ~20 mask widths -- so SR2's line conditioning pointed confidently at
    continuum. Scaling the width by sigma_z makes the mask at least contain the
    true position when the head is unsure.
    """
    from specsr.models.sr2 import build_line_mask

    wave = torch.linspace(1.0, 5.3, 2000)
    rest = torch.tensor([0.5007, 0.6563])
    z = torch.tensor([3.0, 3.0])

    base = build_line_mask(wave, z, rest, sigma_base_um=0.005)
    zeroed = build_line_mask(wave, z, rest, sigma_base_um=0.005,
                             z_sigma=torch.tensor([0.0, 0.0]))
    # sigma_z = 0 must reproduce the historical mask exactly.
    assert torch.allclose(base, zeroed)

    widened = build_line_mask(wave, z, rest, sigma_base_um=0.005,
                              z_sigma=torch.tensor([0.0, 0.05]))
    assert float(widened[0].sum()) == pytest.approx(float(base[0].sum()), rel=1e-5)
    assert float(widened[1].sum()) > 3 * float(base[1].sum())

    # The cap bounds it: without one, a hopeless redshift would drive the mask
    # towards all-ones and silently neuter the loss's in-line term.
    capped = build_line_mask(wave, z, rest, sigma_base_um=0.005,
                             z_sigma=torch.tensor([0.0, 50.0]), sigma_max_um=0.02)
    assert float(capped[1].mean()) < 0.9


def test_zsigma_channel_is_appended_last_and_counted():
    """Enabling sigma_z must not renumber the existing channels."""
    from specsr.models.sr2 import build_sr2_input, sr2_input_channels

    assert sr2_input_channels() == 5
    assert sr2_input_channels(use_zsigma_channel=True) == 6

    n = 256
    kw = dict(x_low=torch.randn(2, 1, n), sr1_mean=torch.randn(2, 1, n),
              sr1_log_sigma=torch.randn(2, 1, n), zhat=torch.tensor([2.0, 3.0]),
              wave_hi_um=torch.linspace(1.0, 5.3, n),
              line_rest_um=torch.tensor([0.5007]))
    zs = torch.tensor([0.1, 0.2])
    without = build_sr2_input(**kw)
    with_zs = build_sr2_input(**kw, use_zsigma_channel=True, z_sigma=zs)
    assert without.shape[1] == 5 and with_zs.shape[1] == 6
    assert torch.allclose(without[:, :3], with_zs[:, :3])
    assert torch.allclose(with_zs[:, 5, :], zs[:, None].expand(-1, n))

    with pytest.raises(ValueError, match="requires z_sigma"):
        build_sr2_input(**kw, use_zsigma_channel=True)



# --------------------------------------------------------------------------
# SR2
# --------------------------------------------------------------------------


@pytest.fixture
def sr2():
    wave = np.linspace(1.0, 5.0, L).astype("float32")
    lines = line_wavelengths_um()
    return SR2Attention(
        in_channels=3,
        line_rest_um=lines,
        wave_hi_um=wave,
        line_dim=32,
        num_attn_heads=2,
        num_attn_layers=1,
        window_half=5,
        cnn_dim=16,
        num_cnn_blocks=2,
    )


def test_sr2_shapes_train_vs_eval(sr2):
    x = torch.randn(2, 3, L)
    zhat = torch.tensor([3.0, 5.0])

    sr2.train()
    out = sr2(x, zhat)
    assert len(out) == 4, "training mode must expose presence and its logit for the BCE"
    delta, logvar, presence, presence_logit = out
    assert delta.shape == (2, 1, L)
    assert logvar.shape == (2, 1, L)
    assert presence.shape == (2, sr2.K)
    assert presence_logit.shape == (2, sr2.K)
    assert ((presence >= 0) & (presence <= 1)).all()
    # The BCE is computed from the logit and the diagnostics from the
    # probability; they have to describe the same gate.
    assert torch.allclose(presence, torch.sigmoid(presence_logit))

    sr2.eval()
    assert len(sr2(x, zhat)) == 2


def test_sr2_cnn_branch_starts_at_exactly_zero(sr2):
    """The CNN branch head is zero-initialised in weight and bias."""
    sr2.eval()
    with torch.no_grad():
        h = sr2.cnn_in(torch.randn(2, 3, L))
        for blk in sr2.cnn_blocks:
            h = h + 0.5 * blk(h)
        cnn_delta = sr2.cnn_out(h)
    assert torch.equal(cnn_delta, torch.zeros_like(cnn_delta))


def test_sr2_starts_as_near_identity_not_exact(sr2):
    """SR2 is a *near*-identity at init, not an exact one.

    Only the CNN branch is fully zeroed. The line branch has random amp_head
    weights (std=0.01) and a presence gate at sigmoid(-2) ~ 0.12, so it injects
    small Gaussians at catalogued line positions from the first forward pass.
    This pins that behaviour: it must stay small, but it is not zero, and the
    amplitude head is what carries it — the same component implicated in the
    known flux-conservation weakness.
    """
    torch.manual_seed(0)
    sr2.eval()
    delta, _ = sr2(torch.randn(2, 3, L), torch.tensor([2.0, 4.0]))
    peak = delta.abs().max().item()
    assert peak > 0.0, "line branch is not zero-initialised; see the class docstring"
    assert peak < 0.2, f"initial perturbation unexpectedly large ({peak:.3g})"


def test_sr2_zeroes_lines_outside_observed_range(sr2):
    """Lines redshifted off the grid must not contribute flux."""
    # At z=0 nearly every rest-frame line sits blueward of 1 micron.
    pos, in_range = sr2._line_positions(torch.tensor([0.0]))
    assert in_range.sum() < sr2.K, "expected some lines to fall outside the grid"
    assert pos.shape == (1, sr2.K)


def test_sr2_line_positions_track_redshift(sr2):
    """A higher redshift must push line positions redward."""
    p_low, _ = sr2._line_positions(torch.tensor([2.0]))
    p_high, _ = sr2._line_positions(torch.tensor([4.0]))
    # Consider only lines in range at both redshifts.
    _, r_low = sr2._line_positions(torch.tensor([2.0]))
    _, r_high = sr2._line_positions(torch.tensor([4.0]))
    both = (r_low & r_high).squeeze(0)
    if both.any():
        assert (p_high.squeeze(0)[both] >= p_low.squeeze(0)[both]).all()


def test_line_catalogue_is_deduplicated_and_sorted_usable():
    waves = [w for _, w in LINE_LIST_REST_AA]
    assert len(waves) == len(set(waves)), "duplicate wavelengths waste attention capacity"
    assert all(w > 0 for w in waves)
    um = line_wavelengths_um()
    assert um.dtype == np.float32
    assert np.allclose(um, np.array(waves, dtype="float32") * 1e-4)


# --------------------------------------------------------------------------
# Checkpoint compatibility — the extraction must not have changed any
# state_dict key, or published weights stop loading.
# --------------------------------------------------------------------------

# A frozen archive, not a live run directory: a chain in progress rewrites its
# checkpoints in place, so pointing tests at one makes them pass or fail
# depending on how far tonight's training has got.
_CKPT_DIR = os.environ.get("SPECSR_TEST_CHECKPOINT_DIR", "checkpoints/checkpoints_run5_20260728")

needs_weights = pytest.mark.skipif(
    not (Path(_CKPT_DIR) / "best_sr2.pth").exists(),
    reason="published checkpoints not available locally",
)


@needs_weights
def test_published_sr2_loads_strict():
    ck = torch.load(
        Path(_CKPT_DIR) / "best_sr2.pth", map_location="cpu", weights_only=False
    )
    sd, cfg = ck["sr2_state_dict"], ck["config"]
    model = SR2Attention(
        in_channels=sd["line_encoder.0.weight"].shape[1],
        line_rest_um=sd["line_rest_um"].numpy(),
        wave_hi_um=sd["wave_hi_um"].numpy(),
        line_dim=sd["line_embed.weight"].shape[1],
        num_attn_heads=cfg.get("num_attn_heads", 4),
        num_attn_layers=len({k.split(".")[2] for k in sd if k.startswith("line_attn.layers.")}),
        window_half=cfg.get("window_half", 25),
        cnn_dim=sd["cnn_in.0.weight"].shape[0],
        num_cnn_blocks=len({k.split(".")[1] for k in sd if k.startswith("cnn_blocks.")}),
    )
    model.load_state_dict(sd, strict=True)


@needs_weights
def test_published_sr1_loads_strict():
    ck = torch.load(
        Path(_CKPT_DIR) / "best_superres_model.pth",
        map_location="cpu",
        weights_only=False,
    )
    sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    if not any(torch.is_tensor(v) for v in sd.values()):
        sd = next(v for v in ck.values() if isinstance(v, dict))
    model = SuperRes1D(
        in_channels=sd["initial.0.weight"].shape[1],
        hidden_dim=sd["initial.0.weight"].shape[0],
        num_res_blocks=len({k.split(".")[1] for k in sd if k.startswith("resblocks.")}),
    )
    model.load_state_dict(sd, strict=True)


# --------------------------------------------------------------------------
# W&B validation figures
# --------------------------------------------------------------------------


def test_wandb_plot_helpers_never_raise():
    """A plotting bug must not end a training run.

    Both helpers swallow their own failures and return an empty dict, so a
    stage keeps training and simply logs no picture that epoch.
    """
    from specsr import wandb_plots

    wave = np.linspace(1.0, 5.3, 128)
    good = wandb_plots.spectrum_panel(wave, np.zeros(128), np.zeros(128), np.zeros(128),
                                      z_true=3.0, title="t")
    assert set(good) <= {wandb_plots.SPECTRUM_KEY}

    # Mismatched lengths would raise inside matplotlib; the helper must absorb it.
    bad = wandb_plots.spectrum_panel(wave, np.zeros(4), np.zeros(128), np.zeros(128))
    assert bad == {}
    assert wandb_plots.redshift_panel(np.zeros(5), np.zeros(3)) == {}


def test_wandb_plot_keys_sort_to_the_top():
    """Panels must appear above the scalar charts on the run page."""
    from specsr import wandb_plots

    for key in (wandb_plots.SPECTRUM_KEY, wandb_plots.REDSHIFT_KEY):
        assert key.startswith("val/0_")
        assert key < "val/a"
