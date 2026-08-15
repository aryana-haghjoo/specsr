"""Tests for the paired-spectra dataset and config loading."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from specsr.config import load_config, unwrap_wandb_config
from specsr.data.datasets import PairedSpectra, normalize_spectrum

L = 64
N = 12


@pytest.fixture
def product(tmp_path):
    """A minimal built product, shaped like the real thing."""
    rng = np.random.default_rng(0)
    # Distinct per-row offsets and scales, so normalisation is observable.
    scales = np.linspace(1.0, 50.0, N)[:, None]
    offsets = np.linspace(-10.0, 10.0, N)[:, None]
    hi = rng.normal(size=(N, L)) * scales + offsets
    path = tmp_path / "product.npz"
    np.savez(
        path,
        flux_low=rng.normal(size=(N, L)) * scales + offsets,
        flux_high=hi,
        flux_high_err=np.abs(rng.normal(size=(N, L))) * 0.1 * scales,
        z=rng.uniform(1.0, 9.0, N),
        wavelength_high=np.geomspace(1.0, 5.3, L),
        ra=rng.uniform(0, 1, N),
        dec=rng.uniform(0, 1, N),
        field=np.array(["goods-s"] * N),
        parent_id=np.arange(N) // 3,
    )
    return path


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------


def test_normalize_spectrum_standardises():
    x = np.random.default_rng(0).normal(5.0, 3.0, 1000)
    out, mean, std = normalize_spectrum(x)
    assert mean == pytest.approx(5.0, abs=0.3)
    assert std == pytest.approx(3.0, abs=0.3)
    assert out.mean() == pytest.approx(0.0, abs=1e-9)
    assert out.std() == pytest.approx(1.0, abs=1e-9)


def test_normalize_spectrum_survives_flat_input():
    """A flat spectrum must not divide by zero."""
    out, mean, std = normalize_spectrum(np.full(50, 7.0))
    assert np.isfinite(out).all()
    assert std > 0


def test_normalize_spectrum_is_nan_aware():
    x = np.concatenate([np.full(10, np.nan), np.random.default_rng(0).normal(0, 1, 100)])
    _, mean, std = normalize_spectrum(x)
    assert np.isfinite(mean) and np.isfinite(std)


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------


def test_dataset_returns_named_fields(product):
    ds = PairedSpectra(product)
    item = ds[0]
    assert set(item) == {
        "flux_low", "flux_high", "flux_high_err", "z",
        "flux_high_mean", "flux_high_std", "index",
    }
    assert item["flux_low"].shape == (L,)
    assert item["flux_high"].shape == (L,)
    assert len(ds) == N
    assert ds.n_samples == L


def test_dataset_normalisation_is_per_row(product):
    """Per-row standardisation carries no information between rows, so unlike a
    dataset-wide scaling it cannot leak anything about the held-out split."""
    ds = PairedSpectra(product, normalize_flux=True)
    for i in range(N):
        hi = ds[i]["flux_high"].numpy()
        assert hi.mean() == pytest.approx(0.0, abs=1e-4)
        assert hi.std() == pytest.approx(1.0, abs=1e-3)


def test_dataset_errors_scaled_into_normalised_units(product):
    """The reference uncertainty must be divided by the same scale as the flux,
    or the likelihood mixes physical and normalised units."""
    raw = np.load(product)
    ds = PairedSpectra(product, normalize_flux=True)
    for i in (0, N // 2, N - 1):
        expected = raw["flux_high_err"][i] / ds.flux_high_std[i].item()
        assert np.allclose(ds[i]["flux_high_err"].numpy(), expected, rtol=1e-4)


def test_denormalize_roundtrips(product):
    ds = PairedSpectra(product, normalize_flux=True)
    raw = np.load(product)["flux_high"]
    for i in (0, N - 1):
        back = ds.denormalize(ds[i]["flux_high"], i).numpy()
        assert np.allclose(back, raw[i], rtol=1e-3, atol=1e-3)


def test_denormalize_handles_batches(product):
    ds = PairedSpectra(product)
    idx = torch.arange(4)
    batch = torch.stack([ds[i]["flux_high"] for i in range(4)])
    out = ds.denormalize(batch, idx)
    assert out.shape == batch.shape


def test_unnormalised_mode_still_reports_statistics(product):
    ds = PairedSpectra(product, normalize_flux=False)
    raw = np.load(product)["flux_high"]
    assert np.allclose(ds[0]["flux_high"].numpy(), raw[0], rtol=1e-4)
    assert ds.flux_high_std[0].item() == pytest.approx(raw[0].std(), rel=1e-3)


def test_dataset_reads_provenance_when_present(product):
    ds = PairedSpectra(product)
    assert ds.parent_id is not None and len(ds.parent_id) == N
    assert ds.field is not None and ds.field[0] == "goods-s"
    assert ds.wavelength is not None and ds.wavelength.shape == (L,)


def test_missing_arrays_raise_a_useful_error(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, flux_low=np.zeros((3, L)), z=np.zeros(3))
    with pytest.raises(KeyError, match="missing"):
        PairedSpectra(path)


def test_dataset_stores_float32_by_default(product):
    """float64 products at the log grid would be several GB per array."""
    ds = PairedSpectra(product)
    assert ds.flux_low.dtype == torch.float32
    assert ds.flux_high.dtype == torch.float32


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def test_unwrap_wandb_style_config():
    """W&B dumps wrap every entry as {value: ...}; loading one naively gives
    dicts where floats are expected."""
    raw = {
        "lr": {"value": 1e-4},
        "hidden_dim": {"value": 120},
        "_wandb": {"value": {"cli_version": "0.21.1"}},
        "plain": 3,
    }
    cfg = unwrap_wandb_config(raw)
    assert cfg == {"lr": 1e-4, "hidden_dim": 120, "plain": 3}
    assert "_wandb" not in cfg
    assert float(cfg["lr"]) == 1e-4


def test_unwrap_leaves_genuine_nested_dicts_alone():
    """Only the exact {'value': ...} shape is unwrapped."""
    raw = {"sweep": {"value": 1, "other": 2}}
    assert unwrap_wandb_config(raw) == {"sweep": {"value": 1, "other": 2}}


def test_load_config_handles_both_forms(tmp_path):
    plain = tmp_path / "plain.yaml"
    plain.write_text("lr: 0.001\nhidden_dim: 64\n")
    assert load_config(plain) == {"lr": 0.001, "hidden_dim": 64}

    wandb_style = tmp_path / "wandb.yaml"
    wandb_style.write_text("lr:\n  value: 0.001\nhidden_dim:\n  value: 64\n")
    assert load_config(wandb_style) == {"lr": 0.001, "hidden_dim": 64}


def test_load_config_overrides_ignore_none(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("lr: 0.001\nepochs: 10\n")
    cfg = load_config(p, lr=0.5, epochs=None)
    assert cfg["lr"] == 0.5
    assert cfg["epochs"] == 10, "None overrides must not clobber the file"


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")


def test_real_shipped_config_parses_to_scalars():
    """The shipped SR1 config is a W&B dump; every hyperparameter must come back
    as a usable scalar."""
    from pathlib import Path

    p = Path("configs/sr1.yaml")
    if not p.exists():
        pytest.skip("shipped config not available")
    cfg = load_config(p)
    assert float(cfg["lr"]) > 0
    assert int(cfg["hidden_dim"]) == 120
    assert int(cfg["num_res_blocks"]) == 16
    assert cfg["activation"] == "gelu"
    assert "_wandb" not in cfg


# --------------------------------------------------------------------------
# shipped configs must name only keys the trainers read
# --------------------------------------------------------------------------
# A config overrides the stage's DEFAULTS, and `cfg.update(config)` accepts any
# key at all. So a name the trainer never reads is silent: W&B logs it, a reader
# believes it, and nothing uses it. An earlier SR2 sweep config accumulated 28 of
# them -- teacher forcing, an anti-hallucination ramp, a whole line-shape
# schedule -- all left over from the retired train/sr2_best script.
#
# The mirror-image failure is worse and this catches it too: a *live* key set to
# a stale value. `configs/sr2.yaml` was a February 2026 W&B export carrying
# `window_half: 25` and `hp_k: 31`, sized for the retired linear grid, so every
# from-scratch SR2 run trained a different network from the released one.

_STAGE_CONFIGS = [
    "configs/sr1.yaml",
    "configs/sr2.yaml",
    "configs/zhead.yaml",
    "configs/zhead_sr2.yaml",
    "configs/finetune/sr1.yaml",
    "configs/finetune/sr2.yaml",
    "configs/finetune/sr2_fluxfix_smoke.yaml",
]

# Consumed via `cfg.get(...)` rather than declared in DEFAULTS, or passed
# through to wandb.init. Not dead, just not defaulted.
_UNDECLARED_BUT_READ = {"dataset_npz", "wandb_name", "source"}


def _defaults_for(path):
    """The DEFAULTS dict of the stage a config drives, keyed off its filename."""
    from pathlib import Path

    stage = Path(path).stem.split("_")[0]
    import importlib

    return importlib.import_module(f"specsr.training.{stage}").DEFAULTS


@pytest.mark.parametrize("rel", _STAGE_CONFIGS)
def test_shipped_config_names_only_live_keys(rel):
    from pathlib import Path

    p = Path(rel)
    if not p.exists():
        pytest.skip(f"{rel} not available")
    cfg = load_config(p)
    dead = sorted(set(cfg) - set(_defaults_for(rel)) - _UNDECLARED_BUT_READ)
    assert not dead, (
        f"{rel} sets {dead}, which {_defaults_for.__name__} says the stage never "
        "reads. Either the trainer lost the feature and the key should go, or "
        "the key is misspelled and the value it was meant to set is silently "
        "still at its default."
    )


# Keys that define the network itself. Two SR2 configs exist because there are
# two routes to the same model -- from scratch and warm-started -- and they must
# describe the same network, or `--finetune` and a plain run produce
# architectures that cannot be compared or warm-started from each other.
_SR2_ARCHITECTURE_KEYS = [
    "cnn_dim", "num_cnn_blocks", "line_dim", "num_attn_heads", "num_attn_layers",
    "window_half", "dropout", "use_sr1_sigma", "use_line_mask", "use_zhat_channel",
    "use_zsigma_channel",
]


def test_sr2_configs_describe_the_same_network():
    from pathlib import Path

    paths = [Path(p) for p in
             ("configs/sr2.yaml", "configs/finetune/sr2.yaml",
              "configs/finetune/sr2_fluxfix_smoke.yaml")]
    if not all(p.exists() for p in paths):
        pytest.skip("SR2 configs not available")

    from specsr.training.sr2 import DEFAULTS

    loaded = {p: load_config(p) for p in paths}
    for key in _SR2_ARCHITECTURE_KEYS:
        seen = {str(cfg.get(key, DEFAULTS[key])) for cfg in loaded.values()}
        assert len(seen) == 1, (
            f"SR2 configs disagree on {key!r}: "
            + ", ".join(f"{p.name}={loaded[p].get(key, DEFAULTS[key])!r}" for p in paths)
            + ". They describe one network reached two ways."
        )


# --------------------------------------------------------------------------
# error normalisation at astrophysical flux scales
# --------------------------------------------------------------------------
# These use ~1e-21 fluxes on purpose. The fixture above uses scales of order
# unity, which is exactly why the bug these pin down survived: at that scale the
# guard threshold happens to behave, and the error comes out normalised anyway.


@pytest.fixture
def faint_product(tmp_path):
    """A product in real JADES flux units, where std is ~1e-21."""
    rng = np.random.default_rng(1)
    scale = 1e-21
    hi = rng.normal(size=(N, L)) * scale
    err = np.abs(rng.normal(size=(N, L))) * 0.3 * scale
    # A masked pixel, flagged the way the real product flags them.
    err[:, 0] = 1.0
    path = tmp_path / "faint.npz"
    np.savez(
        path,
        flux_low=rng.normal(size=(N, L)) * scale,
        flux_high=hi,
        flux_high_err=err,
        z=rng.uniform(1.0, 9.0, N),
        wavelength_high=np.geomspace(1.0, 5.3, L),
        ra=rng.uniform(0, 1, N),
        dec=rng.uniform(0, 1, N),
        field=np.array(["goods-s"] * N),
        parent_id=np.arange(N) // 3,
    )
    return path


def test_error_is_normalised_with_the_flux(faint_product):
    """err/flux must survive normalisation, at any flux scale.

    Before the fix the guard read ``stds[i] > 1e-9``, which is false for
    ~1e-21 data, so the divisor stayed 1.0: the flux was standardised while the
    error kept physical units. Nothing raised -- every downstream ``1/err**2``
    simply clamped to var_floor and inverse-variance weighting went uniform.
    """
    ds = PairedSpectra(faint_product, normalize_flux=True)
    item = ds[1]
    err = item["flux_high_err"].numpy()
    std = float(item["flux_high_std"])

    real = err[1:]  # skip the masked pixel
    # Normalised flux has unit variance, so a 30% fractional error must land at
    # ~0.3 -- not at the raw 1e-21 it would keep if the divisor were 1.0.
    assert 0.05 < np.median(real) < 2.0, (
        f"median normalised error {np.median(real):.3e} is not on the scale of "
        "the normalised flux; the error was probably left unnormalised"
    )
    assert std < 1e-9, "fixture is meant to exercise the small-std regime"


def test_masked_sentinel_survives_squaring_in_float32(faint_product):
    """A masked pixel must not overflow once squared.

    The sentinel is 1.0 in physical units; normalised by a ~1e-21 std that is
    ~1e21, and float32 tops out near 3.4e38, so ``err**2`` becomes inf and takes
    the whole NLL with it.
    """
    ds = PairedSpectra(faint_product, normalize_flux=True)
    err = ds[1]["flux_high_err"]
    assert torch.isfinite(err).all()
    assert torch.isfinite(err.float() ** 2).all(), "err**2 overflowed float32"
    # The masked pixel must still be heavily down-weighted relative to real ones.
    assert err[0] > 100 * err[1:].median()
