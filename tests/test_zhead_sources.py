"""Tests for the redshift-head input sources and the redshift transform.

The scientific claim these support is that the LR / SR1 / SR2 / HR redshift
comparison holds architecture, loss and split fixed and varies *only* the input
representation. That is now structural — one training loop, four sources — so
these tests check the sources agree on shape and interface, and that the
redshift transform round-trips.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from specsr.models import SR2Attention, SuperRes1D, ZHead1D, line_wavelengths_um
from specsr.training.zhead_sources import SOURCE_NAMES, RawFluxSource, build_source
from specsr.training.ztransform import RedshiftTransform

L = 256
DEV = torch.device("cpu")


# --------------------------------------------------------------------------
# RedshiftTransform
# --------------------------------------------------------------------------


@pytest.fixture
def ztrans():
    rng = np.random.default_rng(0)
    return RedshiftTransform.from_training_redshifts(rng.uniform(1.0, 9.0, 1000))


def test_normalize_denormalize_roundtrip(ztrans):
    z = torch.tensor([1.5, 4.0, 8.7])
    assert torch.allclose(ztrans.denormalize(ztrans.normalize(z)), z, atol=1e-5)


def test_decoded_mean_always_inside_observed_range(ztrans):
    """The sigmoid bound is the point: no raw output may escape the range."""
    extreme = torch.tensor([-1e4, -50.0, 0.0, 50.0, 1e4])
    mu_n = ztrans.decode_mean(extreme)
    assert (mu_n >= ztrans.z_min_n - 1e-5).all()
    assert (mu_n <= ztrans.z_max_n + 1e-5).all()


def test_predict_clamps_logvar_against_overflow(ztrans):
    """Unclamped, exp(log_var) overflows and takes the loss with it."""
    z, sigma = ztrans.predict(torch.zeros(3), torch.tensor([500.0, 0.0, -500.0]))
    assert torch.isfinite(z).all()
    assert torch.isfinite(sigma).all()
    assert (sigma > 0).all()


def test_transform_fitted_only_on_supplied_redshifts():
    """Statistics must come from the training split alone, or the transform
    itself leaks information about held-out redshifts."""
    train = np.array([1.0, 2.0, 3.0])
    t = RedshiftTransform.from_training_redshifts(train)
    assert t.mean == pytest.approx(2.0)
    assert t.denormalize(torch.tensor(t.z_max_n)).item() == pytest.approx(3.0, abs=1e-5)


def test_transform_dict_roundtrip(ztrans):
    assert RedshiftTransform.from_dict(ztrans.as_dict()) == ztrans


def test_zero_variance_redshifts_do_not_divide_by_zero():
    t = RedshiftTransform.from_training_redshifts(np.full(10, 3.0))
    assert t.std == 1.0
    assert torch.isfinite(t.normalize(torch.tensor([3.0]))).all()


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


def test_raw_sources_shape_and_channels():
    batch = {"flux_low": torch.randn(4, L), "flux_high": torch.randn(4, L)}
    for name, key in (("lowres", "flux_low"), ("hires", "flux_high")):
        src = build_source(name)
        out = src(batch, DEV)
        assert out.shape == (4, 1, L)
        assert src.n_channels == 1
        assert src.name == name
        assert isinstance(src, RawFluxSource) and src.key == key


def test_raw_source_accepts_already_channelled_input():
    src = build_source("lowres")
    assert src({"flux_low": torch.randn(2, 1, L)}, DEV).shape == (2, 1, L)


@pytest.mark.parametrize("use_sigma,expected_c", [(True, 2), (False, 1)])
def test_sr1_source_channels(use_sigma, expected_c):
    sr1 = SuperRes1D(in_channels=1, hidden_dim=16, num_res_blocks=2).eval()
    src = build_source("sr1", sr1=sr1, use_sigma=use_sigma)
    out = src({"flux_low": torch.randn(3, L)}, DEV)
    assert out.shape == (3, expected_c, L)
    assert src.n_channels == expected_c


def test_sr1_source_second_channel_is_log_sigma_not_log_var():
    """SR1 emits log_var; the head consumes log_sigma = 0.5 * log_var.
    Getting this wrong is silent and halves the effective uncertainty scale."""
    sr1 = SuperRes1D(in_channels=1, hidden_dim=16, num_res_blocks=2).eval()
    src = build_source("sr1", sr1=sr1, use_sigma=True)
    x = torch.randn(2, L)
    out = src({"flux_low": x}, DEV)
    with torch.no_grad():
        _, log_var = sr1(x.unsqueeze(1))
    assert torch.allclose(out[:, 1:2], 0.5 * log_var, atol=1e-6)


@pytest.mark.parametrize(
    "flags",
    [
        {},
        {"use_zhat_channel": False},
        {"use_sr1_sigma": False, "use_line_mask": False, "use_zhat_channel": False},
    ],
)
def test_sr2_source_runs_full_chain(flags):
    """SR2's in_channels must be derived from the same contract the source uses.

    Hard-coding it is how the stack and the model silently disagree.
    """
    from specsr.models.sr2 import sr2_input_channels

    sr1 = SuperRes1D(in_channels=1, hidden_dim=16, num_res_blocks=2).eval()
    boot = ZHead1D(in_channels=2, hidden_dim=16, num_blocks=2).eval()
    sr2 = SR2Attention(
        in_channels=sr2_input_channels(**flags),
        line_rest_um=line_wavelengths_um(),
        wave_hi_um=np.linspace(1.0, 5.0, L).astype("float32"),
        line_dim=32, num_attn_heads=2, num_attn_layers=1,
        window_half=5, cnn_dim=16, num_cnn_blocks=2,
    ).eval()
    t = RedshiftTransform.from_training_redshifts(np.random.default_rng(1).uniform(1, 9, 200))

    src = build_source("sr2", sr1=sr1, zhead_bootstrap=boot, sr2=sr2, ztransform=t, **flags)
    out = src({"flux_low": torch.randn(2, L)}, DEV)
    assert out.shape == (2, 2, L)
    assert torch.isfinite(out).all()


def test_sr2_source_forwards_every_flag_it_is_given():
    """`train_zhead` reads the SR2 flags off the checkpoint's own config so the
    arm reproduces the input SR2 was trained against. `build_source` silently
    dropped three of them, which fed SR2 an unwidened line mask and degraded the
    single arm the whole comparison exists to measure -- invisibly, because
    `use_zsigma_channel=False` leaves the channel count unchanged."""
    from dataclasses import fields

    from specsr.models.sr2 import sr2_input_channels

    sr1 = SuperRes1D(in_channels=1, hidden_dim=16, num_res_blocks=2).eval()
    boot = ZHead1D(in_channels=2, hidden_dim=16, num_blocks=2).eval()
    sr2 = SR2Attention(
        in_channels=sr2_input_channels(),
        line_rest_um=line_wavelengths_um(),
        wave_hi_um=np.linspace(1.0, 5.0, L).astype("float32"),
        line_dim=32, num_attn_heads=2, num_attn_layers=1,
        window_half=5, cnn_dim=16, num_cnn_blocks=2,
    ).eval()
    t = RedshiftTransform.from_training_redshifts(np.random.default_rng(1).uniform(1, 9, 200))

    passed = dict(zsigma_line_mask=True, zsigma_mask_max_um=0.077,
                  sigma_base_um=0.006, delta_cap=30.0)
    src = build_source("sr2", sr1=sr1, zhead_bootstrap=boot, sr2=sr2,
                       ztransform=t, **passed)
    for k, v in passed.items():
        assert getattr(src, k) == v, f"build_source dropped {k!r}"

    # Nothing on the dataclass may be silently defaulted: every configurable
    # field has to be reachable through build_source.
    reachable = {f.name for f in fields(src)} - {
        "sr1", "zhead_bootstrap", "sr2", "ztransform", "name"}
    for k in reachable:
        probe = build_source("sr2", sr1=sr1, zhead_bootstrap=boot, sr2=sr2,
                             ztransform=t, **{k: passed.get(k, False)})
        assert getattr(probe, k) == passed.get(k, False), f"{k!r} is not forwarded"


def test_unknown_source_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown ZHead source"):
        build_source("medres")


def test_all_declared_sources_are_constructible():
    """SOURCE_NAMES must not drift away from build_source."""
    assert set(SOURCE_NAMES) == {"lowres", "hires", "sr1", "sr2"}
    for name in ("lowres", "hires"):
        assert build_source(name) is not None


# --------------------------------------------------------------------------
# SR2 input stack — channel order and semantics are load-bearing
# --------------------------------------------------------------------------


def _sr2_stack_pieces():
    from specsr.models.sr2 import build_sr2_input

    B = 2
    x_low = torch.randn(B, 1, L)
    sr1_mean = torch.randn(B, 1, L)
    sr1_log_sigma = torch.randn(B, 1, L) * 0.1
    zhat = torch.tensor([3.0, 5.0])
    wave = torch.linspace(1.0, 5.0, L)
    lines = torch.as_tensor(line_wavelengths_um())
    stack = build_sr2_input(
        x_low=x_low, sr1_mean=sr1_mean, sr1_log_sigma=sr1_log_sigma,
        zhat=zhat, wave_hi_um=wave, line_rest_um=lines,
    )
    return stack, x_low, sr1_mean, sr1_log_sigma, zhat


def test_sr2_input_channel_order_is_fixed():
    """A checkpoint trained on one channel order produces silent nonsense if
    fed another, so the order is pinned here."""
    stack, x_low, sr1_mean, sr1_log_sigma, zhat = _sr2_stack_pieces()
    assert stack.shape[1] == 5
    assert torch.allclose(stack[:, 0:1], x_low)
    assert torch.allclose(stack[:, 1:2], sr1_mean)


def test_sr2_sigma_channel_is_linear_not_log():
    """Channel 2 is exp(log_sigma). Passing the log instead is dimensionally
    plausible and trains to a worse but non-obviously-broken model."""
    stack, _, _, sr1_log_sigma, _ = _sr2_stack_pieces()
    assert torch.allclose(stack[:, 2:3], torch.exp(sr1_log_sigma).clamp_min(1e-6), atol=1e-6)
    assert (stack[:, 2:3] > 0).all()


def test_sr2_zhat_channel_is_constant_along_wavelength():
    stack, _, _, _, zhat = _sr2_stack_pieces()
    zchan = stack[:, 4, :]
    assert torch.allclose(zchan, zhat[:, None].expand_as(zchan))


def test_sr2_line_mask_channel_is_bounded():
    stack, *_ = _sr2_stack_pieces()
    mask = stack[:, 3, :]
    assert (mask >= 0).all() and (mask <= 1).all()
    assert mask.max() > 0, "expected some lines inside the grid at these redshifts"


@pytest.mark.parametrize(
    "flags,expected",
    [
        ({}, 5),
        ({"use_zhat_channel": False}, 4),
        ({"use_line_mask": False, "use_zhat_channel": False}, 3),
        ({"use_sr1_sigma": False, "use_line_mask": False, "use_zhat_channel": False}, 2),
    ],
)
def test_sr2_input_channels_matches_built_stack(flags, expected):
    """sr2_input_channels() must agree with what build_sr2_input() produces, or
    a model will be constructed with the wrong in_channels."""
    from specsr.models.sr2 import build_sr2_input, sr2_input_channels

    assert sr2_input_channels(**flags) == expected
    stack = build_sr2_input(
        x_low=torch.randn(2, 1, L), sr1_mean=torch.randn(2, 1, L),
        sr1_log_sigma=torch.zeros(2, 1, L), zhat=torch.tensor([3.0, 4.0]),
        wave_hi_um=torch.linspace(1.0, 5.0, L),
        line_rest_um=torch.as_tensor(line_wavelengths_um()), **flags,
    )
    assert stack.shape[1] == expected
