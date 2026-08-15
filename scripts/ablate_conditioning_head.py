#!/usr/bin/env python
"""Which redshift head should condition SR2's line branch?

    python scripts/ablate_conditioning_head.py

SR2 places its line Gaussians at a redshift inferred by a frozen head. Today that
head reads SR1's output; the `lowres` arm reads the prism directly and is
*strictly better* on both metrics (6.12% vs 10.84% catastrophic outliers, 0.00151
vs 0.00177 median). This runs the released SR2 unchanged and swaps only the head
supplying that redshift.

**The result is a lower bound on the benefit**, because SR2 was trained against
the SR1 head: feeding it another head's output is a train/inference mismatch
working against the proposal rather than for it. A retrained SR2 should do at
least this well.

It reports two things, and the second is the one that mattered:

1. Aggregate MSE and median SR/HR integrated line flux under each head.
2. The same, stratified by whether each head identified the line system. On the
   bright lines the paper reports, the two heads almost always agree -- redshift
   outliers live in faint galaxies, which the HR S/N cut removes -- so the
   aggregate barely moves. What the stratification exposes is that a fifth of
   bright [O III] lines are missing from SR2's output *even when the redshift is
   right*, which is an amplitude/presence problem no conditioning change reaches.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "src", REPO / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from flux_conservation import integrated_line_flux  # noqa: E402

from specsr.data.datasets import FixedGridSpectraDataset  # noqa: E402
from specsr.data.splits import get_training_split  # noqa: E402
from specsr.models.lines import LINE_LIST_REST_AA  # noqa: E402
from specsr.models.loaders import load_sr1, load_sr2, load_zhead  # noqa: E402
from specsr.models.sr2 import build_sr2_input, constrain_delta  # noqa: E402
from specsr.training.ztransform import RedshiftTransform  # noqa: E402

SR1_CKPT = "runs/finetune_20260730_003724/sr1/best_superres_model.pth"
SR1_CONFIG = "checkpoints/checkpoints_baseline_20260726/config_logR.yaml"
SR2_CKPT = "runs/sr2_maskfix_20260803_170711/best_sr2.pth"
HEADS = {
    "SR1 head (current)": "runs/zhead_pdf_8020/sr1/best_zhead_sr1.pth",
    "LR head (proposed)": "runs/zarms_8020_20260731_010103/lowres/best_zhead_lowres.pth",
}
LINES = {"[O II] 3727": 0.3727, "Hbeta": 0.4861, "[O III] 5007": 0.5007, "Halpha": 0.6563}


def build(dataset, device):
    sr1, _ = load_sr1(str(REPO / SR1_CONFIG), str(REPO / SR1_CKPT), device)
    line_rest = np.asarray([w for _, w in LINE_LIST_REST_AA], dtype=np.float32) * 1e-4
    with np.load(dataset, allow_pickle=True) as d:
        wave = np.asarray(d["wavelength_high"], dtype=np.float32)
    sr2, cfg = load_sr2(str(REPO / SR2_CKPT), wave, line_rest, device)
    return sr1, sr2, cfg, wave, line_rest


def run_head(ckpt, *, sr1, sr2, cfg, wave, line_rest, loader, ds, train_idx, device):
    """Full LR -> SR1 -> zhat -> SR2 pass with ``ckpt`` supplying zhat."""
    zhead, z_mean, z_std, use_sigma, _ = load_zhead(str(REPO / ckpt), device, unfreeze_last_n=0)
    zhead.eval()
    z_train = np.asarray(ds.z)[train_idx]
    ztf = RedshiftTransform(mean=z_mean, std=z_std,
                            z_min_n=float((z_train.min() - z_mean) / z_std),
                            z_max_n=float((z_train.max() - z_mean) / z_std))
    wave_t = torch.tensor(wave, device=device)
    cap = float(cfg.get("delta_cap", 30.0))

    zh, out, hi, er = [], [], [], []
    with torch.no_grad():
        for b in loader:
            x_low = torch.nan_to_num(b[0].to(device).unsqueeze(1))
            x_high = torch.nan_to_num(b[1].to(device).unsqueeze(1))
            x_err = torch.nan_to_num(b[2].to(device).unsqueeze(1))
            sr1_mean, sr1_logvar = sr1(x_low)
            sr1_log_sigma = 0.5 * sr1_logvar

            # The only line that differs between arms. A head trained on `lowres`
            # wants the prism; one trained on `sr1` wants SR1's mean and sigma.
            # `use_sigma_channel` rides in the checkpoint and distinguishes them.
            z_in = torch.cat([sr1_mean, sr1_log_sigma], dim=1) if use_sigma else x_low

            mu_raw, logvar_n = zhead(z_in)
            z_t, z_s = ztf.predict(mu_raw, logvar_n, bounded=zhead.bounded_mean)
            zhat = torch.nan_to_num(z_t.reshape(-1), nan=float(z_mean))
            z_sigma = torch.nan_to_num(z_s.reshape(-1), nan=0.05)

            x_in = build_sr2_input(
                x_low=x_low, sr1_mean=sr1_mean, sr1_log_sigma=sr1_log_sigma,
                zhat=zhat, wave_hi_um=wave_t, line_rest_um=line_rest,
                use_sr1_sigma=True, use_line_mask=True, use_zhat_channel=True,
                use_zsigma_channel=False, sigma_base_um=0.005,
                z_sigma=z_sigma, sigma_max_um=0.05)
            delta, _ = sr2(x_in, zhat)

            zh.append(zhat.cpu().numpy())
            out.append((sr1_mean + constrain_delta(delta, cap)).squeeze(1).cpu().numpy())
            hi.append(x_high.squeeze(1).cpu().numpy())
            er.append(x_err.squeeze(1).cpu().numpy())

    return (np.concatenate(zh), np.concatenate(out).astype(np.float64),
            np.concatenate(hi).astype(np.float64), np.concatenate(er).astype(np.float64))


def line_ratios(sr, hr, er, wave, dwave, z_true, rest, snr_min):
    """Per-galaxy SR/HR integrated flux at the *true* line position.

    Measured at the catalogue redshift, not the predicted one: if the head was
    wrong, SR2's Gaussian is somewhere else and this is what notices.
    """
    idx, rat = [], []
    for i in range(len(z_true)):
        c = rest * (1 + z_true[i])
        if not (wave[0] * 1.02 < c < wave[-1] * 0.98):
            continue
        f_hr, s_hr = integrated_line_flux(hr[i], er[i], wave, dwave, c)
        if not (np.isfinite(f_hr) and s_hr and f_hr / s_hr >= snr_min):
            continue
        f_sr, _ = integrated_line_flux(sr[i], None, wave, dwave, c)
        if not np.isfinite(f_sr):
            continue
        idx.append(i)
        rat.append(f_sr / f_hr)
    return np.asarray(idx), np.asarray(rat)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=str(REPO / "data" / "paired_DR4_logR.npz"))
    ap.add_argument("--snr-min", type=float, default=20.0,
                    help="HR line S/N floor; below ~20 the HR flux is itself mostly noise")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr1, sr2, cfg, wave, line_rest = build(args.dataset, device)
    train_idx, val_idx, _ = get_training_split(args.dataset, 0.8, 0.2, 42)
    ds = FixedGridSpectraDataset(args.dataset, normalize_flux=True)
    loader = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size, shuffle=False)
    z_true = np.asarray(ds.z)[val_idx]
    dwave = np.gradient(wave.astype(np.float64))

    res = {}
    for label, ckpt in HEADS.items():
        zh, sr, hr, er = run_head(ckpt, sr1=sr1, sr2=sr2, cfg=cfg, wave=wave,
                                  line_rest=line_rest, loader=loader, ds=ds,
                                  train_idx=train_idx, device=device)
        finite = np.isfinite(sr) & np.isfinite(hr)
        rel = np.abs(zh - z_true) / (1 + z_true)
        res[label] = dict(
            zhat=zh, sr=sr, hr=hr, er=er,
            outlier=100 * float(np.mean(rel > 0.15)), med=float(np.median(rel)),
            mse=float(np.mean((sr[finite] - hr[finite]) ** 2)))
        print(f"\n{label}")
        print(f"  zhat outliers {res[label]['outlier']:.2f}%   "
              f"median |dz|/(1+z) {res[label]['med']:.5f}")
        print(f"  SR2 MSE vs HR {res[label]['mse']:.6f}")
        for name, rest in LINES.items():
            _, r = line_ratios(sr, hr, er, wave, dwave, z_true, rest, args.snr_min)
            print(f"    {name:14s} median SR/HR flux "
                  f"{np.median(r) if len(r) else float('nan'):.3f}  (n={len(r)})")

    a, b = res["SR1 head (current)"], res["LR head (proposed)"]
    print("\n" + "=" * 66)
    print(f"MSE vs HR : {a['mse']:.6f} -> {b['mse']:.6f}   "
          f"({100 * (a['mse'] - b['mse']) / a['mse']:+.2f}% better)")

    # The stratification. Aggregates hide this: on bright lines the two heads
    # almost always agree, so the interesting comparison is conditional.
    rest = LINES["[O III] 5007"]
    ia, ra = line_ratios(a["sr"], a["hr"], a["er"], wave, dwave, z_true, rest, args.snr_min)
    ib, rb = line_ratios(b["sr"], b["hr"], b["er"], wave, dwave, z_true, rest, args.snr_min)
    keep = np.intersect1d(ia, ib)
    ra = ra[np.isin(ia, keep)]
    rb = rb[np.isin(ib, keep)]
    oa = (np.abs(a["zhat"] - z_true) / (1 + z_true) > 0.15)[keep]
    ob = (np.abs(b["zhat"] - z_true) / (1 + z_true) > 0.15)[keep]

    print(f"\n[O III] 5007, {len(keep)} galaxies at HR S/N >= {args.snr_min:g}")
    for name, m in (("both heads right", ~oa & ~ob), ("SR1 wrong, LR right", oa & ~ob),
                    ("LR wrong, SR1 right", ~oa & ob), ("both wrong", oa & ob),
                    ("ALL", np.ones(len(keep), bool))):
        if not m.sum():
            print(f"  {name:22s} n=0")
            continue
        print(f"  {name:22s} n={m.sum():3d}   SR/HR median "
              f"{np.median(ra[m]):.3f} -> {np.median(rb[m]):.3f}   "
              f"line missing (<0.1): {100 * np.mean(ra[m] < 0.1):.1f}%"
              f" -> {100 * np.mean(rb[m] < 0.1):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
