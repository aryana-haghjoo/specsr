"""The packaged model classes must reproduce the shipped checkpoints.

This is the test that keeps `src/specsr/` honest. The package used to be a
parallel implementation that nothing imported, so nothing exercised it and it
drifted — `cnn_scale` was added to the trainer's SR2 and was missing from the
packaged one within a day. Anyone loading a checkpoint through the package would
have got a different network from the one that trained.

Structure is not behaviour, so these assert on *numbers*, not on key names alone.
Each is a value measured on the 286-spectrum validation split with the baseline
checkpoints. See ARCHITECTURE.md.

Skips cleanly when the data or checkpoints are absent, so CI without the 4 GB
dataset stays green — but it must not be deleted, and a failure here means the
package no longer matches what produced the paper's numbers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "checkpoints/checkpoints_baseline_20260726"
DATASET = REPO / "data" / "paired_DR4_logR.npz"

# Measured on the validation split with the baseline checkpoints. Changing these
# means the package computes something different from what trained.
# Val MSE of the shipped SR1, on the 80/20 split's 572-galaxy evaluation set.
#
# This value is tied to the split, not only to the weights. It was 0.845963 on
# the three-way split's 286 galaxies, and the same checkpoint reproduces that
# number to 2.4e-07 on that same set today -- so the change below is the sample
# growing, not the package drifting. Verify that way before ever editing it: the
# whole point of this test is to catch a package/trainer divergence, and quietly
# re-baselining it would throw that away.
SR1_VAL_MSE = 0.850640
SR1_VAL_MSE_286 = 0.845963  # historical, on the retired 80/10/10 split
SR2_PRESENCE_MEAN = 0.00972

needs_artifacts = pytest.mark.skipif(
    not (DATASET.exists() and BASELINE.exists()),
    reason="dataset or baseline checkpoints not available",
)


def _val_loader(batch_size=32):

    from specsr.data.datasets import FixedGridSpectraDataset
    from specsr.evaluation import load_split

    ds = FixedGridSpectraDataset(str(DATASET), normalize_flux=True)
    idx = load_split("val", DATASET)
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idx), batch_size=batch_size, shuffle=False)


@needs_artifacts
def test_package_sr1_reproduces_baseline_mse():
    """`specsr.models.sr1.SuperRes1D` must give the baseline SR1 number exactly."""
    import yaml

    from specsr.models.sr1 import SuperRes1D

    cfg = yaml.safe_load((BASELINE / "config_logR.yaml").read_text())
    state = torch.load(BASELINE / "best_superres_model.pth", map_location="cpu")

    model = SuperRes1D(
        in_channels=1, hidden_dim=int(cfg["hidden_dim"]),
        num_res_blocks=int(cfg["num_res_blocks"]), dropout=float(cfg["dropout"]),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (
        f"package SuperRes1D no longer matches the shipped checkpoint: "
        f"missing={missing[:5]} unexpected={unexpected[:5]}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in _val_loader():
            x_low = torch.nan_to_num(batch[0].to(device).unsqueeze(1))
            x_high = torch.nan_to_num(batch[1].to(device).unsqueeze(1))
            pred, _ = model(x_low)
            total += float(((pred - x_high) ** 2).mean()) * x_high.shape[0]
            n += x_high.shape[0]

    assert np.isclose(total / n, SR1_VAL_MSE, atol=1e-6), (
        f"package SuperRes1D gives val MSE {total / n:.6f}, expected {SR1_VAL_MSE}. "
        "The package no longer computes what produced the paper's numbers."
    )


@needs_artifacts
def test_package_sr2_loads_shipped_checkpoint():
    """`specsr.models.sr2.SR2Attention` must accept the shipped SR2 weights."""
    from specsr.models.lines import LINE_LIST_REST_AA
    from specsr.models.sr2 import SR2Attention

    ck = torch.load(BASELINE / "best_sr2.pth", map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    with np.load(DATASET, allow_pickle=True) as d:
        wave = np.asarray(d["wavelength_high"], dtype=np.float32)
    line_rest = np.asarray([w for _, w in LINE_LIST_REST_AA], dtype=np.float32) * 1e-4

    in_ch = 2 + int(bool(cfg["use_sr1_sigma"])) + int(bool(cfg["use_line_mask"])) \
        + int(bool(cfg["use_zhat_channel"]))
    model = SR2Attention(
        in_channels=in_ch, line_rest_um=line_rest, wave_hi_um=wave,
        line_dim=int(cfg["line_dim"]), num_attn_heads=int(cfg["num_attn_heads"]),
        num_attn_layers=int(cfg["num_attn_layers"]), window_half=int(cfg["window_half"]),
        cnn_dim=int(cfg["cnn_dim"]), num_cnn_blocks=int(cfg["num_cnn_blocks"]),
        dropout=float(cfg["dropout"]),
    )
    missing, unexpected = model.load_state_dict(ck["sr2_state_dict"], strict=False)
    assert not missing and not unexpected, (
        f"package SR2Attention no longer matches the shipped checkpoint: "
        f"missing={missing[:5]} unexpected={unexpected[:5]}"
    )


@needs_artifacts
def test_package_zhead_loads_shipped_checkpoint():
    """`specsr.models.zhead.ZHead1D` must accept the shipped ZHead weights."""
    from specsr.models.zhead import ZHead1D

    ck = torch.load(BASELINE / "best_zhead.pth", map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    model = ZHead1D(
        in_channels=2 if ck["use_sigma_channel"] else 1,
        hidden_dim=int(cfg["hidden_dim"]), num_blocks=int(cfg["num_blocks"]),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    missing, unexpected = model.load_state_dict(ck["zhead_state_dict"], strict=False)
    assert not missing and not unexpected, (
        f"package ZHead1D no longer matches the shipped checkpoint: "
        f"missing={missing[:5]} unexpected={unexpected[:5]}"
    )


def test_sr2_exposes_cnn_scale():
    """The CNN-branch scale must exist in the package, not only in the trainer.

    This is the specific divergence that motivated the consolidation: the flag
    was added under `train/` and the packaged copy silently lacked it.
    """
    import inspect

    from specsr.models.sr2 import SR2Attention

    assert "cnn_scale" in inspect.signature(SR2Attention.__init__).parameters, (
        "specsr.models.sr2.SR2Attention is missing cnn_scale; the package has "
        "drifted from the trainer again (see ARCHITECTURE.md)"
    )
