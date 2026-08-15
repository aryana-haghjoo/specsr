"""Shared evaluation plumbing: split selection, model loading, inference.

Every figure and every analysis should go through this, so that "which split,
which checkpoints" is answered the same way everywhere and appears in the output
rather than living in a notebook cell.

Everything comes from ``specsr`` itself -- the line mask, the delta cap and the
checkpoint loaders included. Anything reached outside ``src/specsr`` is absent
from an installed wheel, which would make this module, and therefore every
published figure, runnable only from a git checkout.

Which partition
---------------
The project runs a **two-way 80/20 group split** (2,286 / 572 parent galaxies);
``"val"`` *is* the held-out set and is the default for everything diagnostic.
There is no sealed test partition, so ``load_split("test")`` returns nothing
unless the split was built with ``allow_empty_test=False``; it still requires
``allow_test=True`` and prints a banner, because the value of a sealed set is
destroyed by looking at it repeatedly. ``ARCHITECTURE.md`` explains why the
three-way split was retired and what that costs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..data.datasets import FixedGridSpectraDataset
from ..data.splits import get_or_make_split_3way
from ..models.lines import LINE_LIST_REST_AA
from ..models.loaders import load_sr1, load_sr2, load_zhead
from ..models.sr2 import build_line_mask, constrain_delta
from ..training.ztransform import RedshiftTransform

# src/specsr/evaluation/__init__.py -> repo root is three levels up.
REPO = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = REPO / "data" / "paired_DR4_logR.npz"
BASELINE = REPO / "checkpoints/checkpoints_baseline_20260726"


#: The split every evaluation uses. Kept equal to `get_training_split`'s
#: defaults so that "the held-out set" means one thing across the project --
#: training validation, the figures, and anything quoted in the paper.
TRAIN_FRAC = 0.8
VAL_FRAC = 0.2
SEED = 42
ALLOW_EMPTY_TEST = True

def line_rest_um() -> np.ndarray:
    """Rest wavelengths in microns, in the token order the models were built with."""
    return np.asarray([w for _, w in LINE_LIST_REST_AA], dtype=np.float32) * 1e-4


def load_split(split: str, dataset=DEFAULT_DATASET, *, allow_test: bool = False) -> np.ndarray:
    """Row indices for ``split`` in {"train", "val", "test"}.

    ``get_training_split`` returns ``(train, val, split_path)`` and withholds the
    test indices on purpose, so it must not be unpacked as three index arrays —
    doing so silently yields a *path string* where indices were expected.
    """
    # These fractions must match `get_training_split`'s defaults exactly. They
    # were hardcoded at 0.8/0.1 after the switch to 80/20, which would have had
    # every figure and every reported number computed on 286 galaxies while
    # training validated on 572 -- two different held-out sets, silently, with
    # nothing in any log to say so.
    train_idx, val_idx, test_idx, _ = get_or_make_split_3way(
        str(dataset), TRAIN_FRAC, VAL_FRAC, SEED, allow_empty_test=ALLOW_EMPTY_TEST
    )
    if split == "train":
        return np.asarray(train_idx)
    if split == "val":
        return np.asarray(val_idx)
    if split == "test":
        if len(test_idx) == 0:
            raise RuntimeError(
                "There is no test partition in the current two-way "
                f"{int(TRAIN_FRAC * 100)}/{int(VAL_FRAC * 100)} configuration: the "
                "held-out galaxies are all in 'val'. Set VAL_FRAC=0.1 and "
                "ALLOW_EMPTY_TEST=False (and retrain) for a sealed test set."
            )
        if not allow_test:
            raise RuntimeError(
                "Refusing to load the test split without allow_test=True.\n"
                "The test set is for final numbers on a frozen model. Iterating "
                "against it turns it into a second validation set, which is how "
                "the augmentation leak stopped meaning anything. Use 'val' for "
                "anything you plan to look at more than once."
            )
        print("!! TEST SPLIT — final numbers only. Do not iterate against this.")
        return np.asarray(test_idx)
    raise ValueError(f"unknown split {split!r}")


@dataclass
class Pipeline:
    """LR -> SR1 -> ZHead -> SR2, with the checkpoints it was built from."""

    sr1: torch.nn.Module
    zhead: torch.nn.Module
    sr2: torch.nn.Module | None
    cfg: dict
    z_mean: float
    z_std: float
    use_sigma: bool
    z_min_n: float
    z_max_n: float
    wave: np.ndarray
    device: torch.device
    provenance: dict


def load_pipeline(
    *,
    sr2_ckpt=None,
    sr1_ckpt=BASELINE / "best_superres_model.pth",
    sr1_config=BASELINE / "config_logR.yaml",
    zhead_ckpt=BASELINE / "best_zhead.pth",
    dataset=DEFAULT_DATASET,
    device=None,
) -> Pipeline:
    """Load the chain from explicit paths.

    Defaults point at ``checkpoints/checkpoints_baseline_20260726/`` rather than ``train/``
    on purpose: a running chain overwrites ``train/`` in place, so evaluating
    against it mid-run compares checkpoints on top of different upstream models.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr1, _ = load_sr1(str(sr1_config), str(sr1_ckpt), device)
    zhead, z_mean, z_std, use_sigma, _ = load_zhead(str(zhead_ckpt), device, unfreeze_last_n=0)

    with np.load(str(dataset), allow_pickle=True) as d:
        wave = np.asarray(d["wavelength_high"], dtype=np.float32)

    ds = FixedGridSpectraDataset(str(dataset), normalize_flux=True)
    train_idx = load_split("train", dataset)
    z_train = np.asarray(ds.z[train_idx], dtype=np.float32)
    z_min_n = float((z_train.min() - z_mean) / z_std)
    z_max_n = float((z_train.max() - z_mean) / z_std)

    sr2 = None
    cfg: dict = {}
    if sr2_ckpt is not None:
        sr2, cfg = load_sr2(str(sr2_ckpt), wave, line_rest_um(), device)

    prov = {
        "sr1_ckpt": str(sr1_ckpt), "sr1_config": str(sr1_config),
        "zhead_ckpt": str(zhead_ckpt), "sr2_ckpt": str(sr2_ckpt),
        "dataset": str(dataset),
        "sr2_wandb_name": cfg.get("wandb_name", ""),
        "sr2_epochs_configured": cfg.get("epochs", ""),
    }
    print("pipeline:")
    for k, v in prov.items():
        print(f"  {k:22s} {v}")

    return Pipeline(sr1=sr1, zhead=zhead, sr2=sr2, cfg=cfg, z_mean=z_mean, z_std=z_std,
                    use_sigma=use_sigma, z_min_n=z_min_n, z_max_n=z_max_n, wave=wave,
                    device=device, provenance=prov)


@torch.no_grad()
def predict(p: Pipeline, split: str = "val", *, dataset=DEFAULT_DATASET,
            batch_size: int = 32, allow_test: bool = False) -> dict:
    """Run the chain over ``split`` and return everything the figures need.

    All spectra come back in **physical HR units** (de-normalised), because that
    is what line fluxes and residuals mean something in. The normalisation
    moments are returned too, for anything that needs the model's own space.
    """
    idx = load_split(split, dataset, allow_test=allow_test)
    ds = FixedGridSpectraDataset(str(dataset), normalize_flux=True)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idx), batch_size=batch_size, shuffle=False)

    with np.load(str(dataset), allow_pickle=True) as d:
        err_phys_all = np.asarray(d["flux_high_err"], dtype=np.float64)
        # LR is normalised by its OWN mean/std, which the dataset does not keep,
        # so it cannot be de-normalised with the HR moments -- doing so gave a
        # median of 4.0e-21 against a true 1.9e-21. Read the physical LR straight
        # from the product instead.
        lr_phys_all = np.asarray(d["flux_low"], dtype=np.float64)
        parent = np.asarray(d["parent_id"])[idx]
        valid_hi = np.asarray(d["valid_high"])[idx] if "valid_high" in d else None

    lr = np.asarray(line_rest_um())
    wave_t = torch.tensor(p.wave, device=p.device)
    out: dict[str, list] = {k: [] for k in
                            ("flux_low", "flux_high", "flux_high_err", "sr1", "sr1_sigma",
                             "sr2", "sr2_sigma", "z_true", "z_pred", "z_sigma",
                             "hi_mean", "hi_std", "presence")}

    for batch in loader:
        x_low = torch.nan_to_num(batch[0].to(p.device).unsqueeze(1))
        x_high = torch.nan_to_num(batch[1].to(p.device).unsqueeze(1))
        z_true = batch[3].to(p.device).float()
        hi_mean = batch[4].to(p.device).view(-1, 1, 1)
        hi_std = batch[5].to(p.device).view(-1, 1, 1)

        sr1_mean, sr1_logvar = p.sr1(x_low)
        sr1_log_sigma = 0.5 * sr1_logvar
        sr1_sigma = torch.exp(sr1_log_sigma).clamp_min(1e-6)

        z_in = torch.cat([sr1_mean, sr1_log_sigma], 1) if p.use_sigma else sr1_mean
        mu_raw, logvar_z = p.zhead(z_in)
        # Decode through the shared transform, branching on `bounded_mean`.
        # This used to apply the sigmoid range-squash unconditionally, which is
        # correct only for the Gaussian head. The classification head's mean is
        # already in normalised redshift units, so squashing it again compressed
        # every estimate towards the range centre: the cache came out with a
        # median |dz|/(1+z) of 0.96 instead of 0.0018, and because `zhat` also
        # drives the line mask and SR2's z channel, SR2 was being evaluated at
        # roughly twice each galaxy's true redshift. Every other consumer already
        # branched on `zhead.bounded_mean`; this was the one hand-rolled copy.
        ztf = RedshiftTransform(mean=p.z_mean, std=p.z_std,
                                z_min_n=p.z_min_n, z_max_n=p.z_max_n)
        zhat, z_sig = ztf.predict(mu_raw.squeeze(-1), logvar_z.squeeze(-1),
                                  bounded=p.zhead.bounded_mean)
        zhat = torch.nan_to_num(zhat.reshape(-1), nan=float(p.z_mean))
        z_sig = z_sig.reshape(-1)

        def den(t, _std=hi_std, _mean=hi_mean):
            """De-normalise back to physical HR units, binding this batch's moments."""
            return (t * _std + _mean).squeeze(1).cpu().numpy()

        out["flux_high"].append(den(x_high))
        out["sr1"].append(den(sr1_mean))
        out["sr1_sigma"].append((sr1_sigma * hi_std).squeeze(1).cpu().numpy())
        out["z_true"].append(z_true.cpu().numpy())
        out["z_pred"].append(zhat.cpu().numpy())
        out["z_sigma"].append(z_sig.cpu().numpy())
        out["hi_mean"].append(hi_mean.reshape(-1).cpu().numpy())
        out["hi_std"].append(hi_std.reshape(-1).cpu().numpy())

        if p.sr2 is not None:
            line_mask = torch.nan_to_num(build_line_mask(
                wave_t, zhat, lr, sigma_base_um=float(p.cfg["sigma_base_um"])))
            chans = [x_low, sr1_mean]
            if p.cfg["use_sr1_sigma"]:
                chans.append(sr1_sigma)
            if p.cfg["use_line_mask"]:
                chans.append(line_mask)
            if p.cfg["use_zhat_channel"]:
                chans.append(zhat[:, None, None].expand(-1, 1, sr1_mean.shape[-1]))
            grab = {}
            h = p.sr2.presence_head.register_forward_hook(
                lambda m, i, o, _g=grab: _g.__setitem__("p", torch.sigmoid(o.detach())))
            delta_raw, sr2_logvar = p.sr2(torch.cat(chans, 1), zhat)
            h.remove()
            sr2_mean = sr1_mean + constrain_delta(delta_raw, float(p.cfg["delta_cap"]))
            sr2_logvar = sr2_logvar.clamp(float(p.cfg["logvar_min"]), float(p.cfg["logvar_max"]))
            out["sr2"].append(den(sr2_mean))
            out["sr2_sigma"].append(
                (torch.exp(0.5 * sr2_logvar) * hi_std).squeeze(1).cpu().numpy())
            out["presence"].append(grab["p"].mean(dim=1).cpu().numpy())

    res = {k: np.concatenate(v) for k, v in out.items() if v}
    res["wave"] = p.wave
    res["flux_low"] = lr_phys_all[idx]
    res["flux_high_err"] = err_phys_all[idx]
    res["parent_id"] = parent
    res["row_index"] = idx
    res["split"] = split
    if valid_hi is not None:
        res["valid_high"] = valid_hi
    res["provenance"] = np.array(str(p.provenance))
    return res
