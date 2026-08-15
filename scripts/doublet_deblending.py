#!/usr/bin/env python
"""Can super-resolution deblend the [O III] doublet? LR vs SR1 vs SR2 vs HR.

The question this answers is whether SR1 earns its place at all. The redshift
arms say a head reading the *raw prism* recovers redshift better than one
reading SR1 (6.12% vs 10.84% catastrophic), and that direction is forced: SR1 is
a deterministic function of the prism, so it cannot carry information the prism
does not. Any claim for super-resolution therefore has to be about making
structure *accessible*, not about adding information -- and the paper's headline
claim is deblending.

[O III] 4959/5007 is the right test because **its flux ratio is fixed by atomic
physics at 2.98**, so no ground truth is needed to score an arm: any departure
from 2.98 is measurement error. The two lines sit 2898 km/s apart, which on this
grid is 38.7 samples -- 22.8 sigma for an R=1000 line (resolved) but 2.3 sigma
for the R=100 prism (a single blob). Reddening between two lines 48 A apart is
negligible, so the ratio is not astrophysically variable.

Each arm is fit with the *same* classical model a careful observer would use:
a linear continuum plus three Gaussians at the known redshifted positions of
Hbeta, 4959 and 5007, sharing one width (one LSF per spectrum), amplitudes
constrained non-negative. Hbeta is modelled rather than excluded because at
R=100 it leaks into the doublet window, and dropping it would penalise the prism
for a blend the fit can in principle handle.

Deliberately *not* measured as integrated flux in a +/-500 km/s window: that
window is matched to HR resolution, so the prism would fail it by construction
and SR1 would look good for the wrong reason.

Ratios need no de-normalisation -- the per-spectrum scale cancels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares

from specsr.data.datasets import FixedGridSpectraDataset
from specsr.evaluation import load_split
from specsr.models.lines import LINE_LIST_REST_AA
from specsr.models.loaders import load_sr1, load_zhead
from specsr.models.sr2 import (
    SR2Attention,
    build_sr2_input,
    constrain_delta,
    sr2_input_channels,
)
from specsr.training.ztransform import RedshiftTransform

REPO = Path(__file__).resolve().parents[1]
C_KMS = 299792.458

HB, O4959, O5007 = 4861.33, 4958.91, 5006.84
ATOMIC_RATIO = 2.98  # F(5007)/F(4959), fixed by the transition probabilities


def integrated_line_flux(flux, err, wave, dwave, center,
                         core_kms=500.0, sb_lo_kms=800.0, sb_hi_kms=1500.0):
    """Only used to select which galaxies have a well-detected HR doublet."""
    v = (wave - center) / center * C_KMS
    core = np.abs(v) <= core_kms
    side = (np.abs(v) >= sb_lo_kms) & (np.abs(v) <= sb_hi_kms)
    if core.sum() < 3 or side.sum() < 5:
        return np.nan, np.nan
    cont = np.median(flux[side])
    f = float(np.sum((flux[core] - cont) * dwave[core]))
    s = float(np.sqrt(np.sum((err[core] * dwave[core]) ** 2)))
    return f, s


def fit_doublet(wave, flux, centres, *, sigma_lo_kms=40.0, sigma_hi_kms=3000.0,
                p0_ratio=3.33):
    """Linear continuum + three Gaussians at fixed centres, one shared width.

    The fit runs in **velocity** relative to the doublet midpoint, not in
    microns. That keeps the width in units with a physical meaning -- one grid
    sample is 74.96 km/s, an R=1000 line is sigma ~127 km/s and the R=100 prism
    ~1273 km/s -- so a single pair of bounds covers every arm and the fitted
    width is directly readable as the resolution the arm is behaving at. (Bounds
    expressed in samples against a micron abscissa silently force every Gaussian
    flat, which returns zero amplitudes for *all* arms including HR.)

    Returns ``(A_hb, A_4959, A_5007, sigma_kms, ok)``. Amplitudes are bounded
    non-negative: a negative emission line is not physical, and letting 4959 go
    negative lets the fit trade a spurious absorption there against a too-bright
    5007 -- exactly the degeneracy under test.
    """
    c_hb, c1, c2 = centres
    lam0 = 0.5 * (c1 + c2)
    v = (wave - lam0) / lam0 * C_KMS
    v_hb, v1, v2 = ((c - lam0) / lam0 * C_KMS for c in (c_hb, c1, c2))

    def model(p):
        a, b, sig, A_hb, A1, A2 = p
        g = a + b * v
        for A, vc in ((A_hb, v_hb), (A1, v1), (A2, v2)):
            g = g + A * np.exp(-0.5 * ((v - vc) / sig) ** 2)
        return g

    # Checked rather than caught. A blanket `except` here turns a shape mismatch
    # -- passing the unwindowed spectrum, say -- into a NaN that looks exactly
    # like a hard fit, and every arm then reports "no usable fits" with no clue
    # why. A coding error must raise.
    if wave.shape != flux.shape:
        raise ValueError(f"wave {wave.shape} and flux {flux.shape} must match")
    if not np.isfinite(flux).all():
        return (np.nan,) * 4 + (False,)

    med = float(np.median(flux))
    amp0 = max(float(np.max(flux) - med), 1e-6)
    p0 = [med, 0.0, 300.0, amp0 * 0.3, amp0 / max(p0_ratio, 1e-6), amp0]
    lo = [-np.inf, -np.inf, sigma_lo_kms, 0.0, 0.0, 0.0]
    hi = [np.inf, np.inf, sigma_hi_kms, np.inf, np.inf, np.inf]
    r = least_squares(lambda p: model(p) - flux, p0, bounds=(lo, hi),
                      max_nfev=4000, method="trf")
    _, _, sig, A_hb, A1, A2 = r.x
    return A_hb, A1, A2, sig, bool(r.success)


def main():
    ap = argparse.ArgumentParser()
    RUNS = REPO / "runs"
    ap.add_argument("--ckpt", default=str(RUNS / "sr2_maskfix_20260803_170711/best_sr2.pth"))
    ap.add_argument("--sr1-ckpt",
                    default=str(RUNS / "finetune_20260730_003724/sr1/best_superres_model.pth"))
    ap.add_argument("--sr1-config",
                    default=str(REPO / "checkpoints/checkpoints_run5_20260728/config_logR.yaml"))
    ap.add_argument("--zhead-ckpt",
                    default=str(RUNS / "zhead_pdf_8020/sr1/best_zhead_sr1.pth"))
    ap.add_argument("--dataset", default=str(REPO / "data/paired_DR4_logR.npz"))
    ap.add_argument("--snr-min", type=float, default=20.0,
                    help="HR integrated SNR at 5007 required to enter the sample")
    ap.add_argument("--window-kms", type=float, default=8000.0,
                    help="half-width of the fitting window about the doublet midpoint")
    ap.add_argument("--p0-ratio", type=float, default=3.33,
                    help="starting 5007/4959 ratio; vary it to test whether a "
                         "near-degenerate fit is merely returning its own guess")
    ap.add_argument("--out", default=str(REPO / "evaluations" / "doublet_deblending.npz"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    print(f"SR1   : {args.sr1_ckpt}")
    print(f"ZHead : {args.zhead_ckpt}")
    print(f"SR2   : {args.ckpt}  (epoch {ck.get('epoch')})")

    sr1, _ = load_sr1(args.sr1_config, args.sr1_ckpt, device)
    zhead, z_mean, z_std, use_sigma, _ = load_zhead(
        args.zhead_ckpt, device, unfreeze_last_n=0)

    data = np.load(args.dataset, allow_pickle=True)
    wave = np.asarray(data["wavelength_high"], dtype=np.float64)
    dwave = np.gradient(wave)
    wave_t = torch.tensor(wave, dtype=torch.float32, device=device)
    line_rest_um = np.asarray([w for _, w in LINE_LIST_REST_AA], dtype=np.float32) * 1e-4

    train_idx = load_split("train", args.dataset)
    idx = np.asarray(load_split("val", args.dataset))
    target_key = cfg.get("target_key_raw", "flux_high")
    ds = FixedGridSpectraDataset(args.dataset, normalize_flux=True, target_key_raw=target_key)
    err_phys_all = np.asarray(data[target_key + "_err"], dtype=np.float64)

    z_train = np.array(ds.z[train_idx], dtype=np.float32)
    ztf = RedshiftTransform(mean=z_mean, std=z_std,
                            z_min_n=float((z_train.min() - z_mean) / z_std),
                            z_max_n=float((z_train.max() - z_mean) / z_std))
    use_zsigma = bool(cfg.get("zsigma_line_mask", False)) or \
        bool(cfg.get("use_zsigma_channel", False))
    zsigma_max = float(cfg.get("zsigma_mask_max_um", 0.05))

    sr2 = SR2Attention(
        in_channels=sr2_input_channels(
            use_sr1_sigma=bool(cfg["use_sr1_sigma"]), use_line_mask=bool(cfg["use_line_mask"]),
            use_zhat_channel=bool(cfg["use_zhat_channel"]),
            use_zsigma_channel=bool(cfg.get("use_zsigma_channel", False))),
        line_rest_um=line_rest_um, wave_hi_um=wave.astype(np.float32),
        line_dim=int(cfg["line_dim"]), num_attn_heads=int(cfg["num_attn_heads"]),
        num_attn_layers=int(cfg["num_attn_layers"]), window_half=int(cfg["window_half"]),
        cnn_dim=int(cfg["cnn_dim"]), num_cnn_blocks=int(cfg["num_cnn_blocks"]),
        dropout=float(cfg["dropout"]), cnn_scale=float(cfg.get("cnn_scale", 1.0)),
    ).to(device)
    sr2.load_state_dict(ck["sr2_state_dict"])
    sr2.eval()

    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idx), batch_size=32, shuffle=False)
    arms = ("lr", "sr1", "sr2", "hr")
    rows = []
    cursor = 0

    with torch.no_grad():
        for batch in loader:
            x_low = torch.nan_to_num(batch[0].to(device).unsqueeze(1))
            x_high = torch.nan_to_num(batch[1].to(device).unsqueeze(1))
            z_true = batch[3].to(device).float()

            sr1_mean, sr1_logvar = sr1(x_low)
            sr1_log_sigma = 0.5 * sr1_logvar
            z_in = torch.cat([sr1_mean, sr1_log_sigma], dim=1) if use_sigma else sr1_mean
            mu_raw, logvar_n = zhead(z_in)
            zhat_t, zsig_t = ztf.predict(mu_raw, logvar_n, bounded=zhead.bounded_mean)
            zhat = torch.nan_to_num(zhat_t.reshape(-1), nan=float(z_mean))
            z_sigma = torch.nan_to_num(zsig_t.reshape(-1), nan=zsigma_max) if use_zsigma else None

            x_in = build_sr2_input(
                x_low=x_low, sr1_mean=sr1_mean, sr1_log_sigma=sr1_log_sigma, zhat=zhat,
                wave_hi_um=wave_t, line_rest_um=line_rest_um,
                use_sr1_sigma=bool(cfg["use_sr1_sigma"]), use_line_mask=bool(cfg["use_line_mask"]),
                use_zhat_channel=bool(cfg["use_zhat_channel"]),
                use_zsigma_channel=bool(cfg.get("use_zsigma_channel", False)),
                sigma_base_um=float(cfg["sigma_base_um"]),
                z_sigma=z_sigma, sigma_max_um=zsigma_max)
            delta_raw, _ = sr2(x_in, zhat)
            sr2_mean = sr1_mean + constrain_delta(delta_raw, float(cfg["delta_cap"]))

            spec = {
                "lr": x_low.squeeze(1).cpu().numpy(),
                "sr1": sr1_mean.squeeze(1).cpu().numpy(),
                "sr2": sr2_mean.squeeze(1).cpu().numpy(),
                "hr": x_high.squeeze(1).cpu().numpy(),
            }
            err_phys = err_phys_all[idx[cursor:cursor + spec["hr"].shape[0]]]
            cursor += spec["hr"].shape[0]
            zt = z_true.cpu().numpy()
            # The SNR cut below compares against errors that are in PHYSICAL
            # units while the spectra here are per-row normalised, so the HR
            # arm has to be put back on its own scale first. Skipping this makes
            # the "SNR" ~1e21 and the cut selects everything.
            hi_mean, hi_std = batch[4].numpy(), batch[5].numpy()

            for b in range(spec["hr"].shape[0]):
                zb = float(zt[b])
                c_hb, c1, c2 = (w * 1e-4 * (1.0 + zb) for w in (HB, O4959, O5007))
                if c1 < wave[0] or c2 > wave[-1]:
                    continue
                # select on the HR doublet being genuinely detected
                hr_phys = spec["hr"][b] * float(hi_std[b]) + float(hi_mean[b])
                f5, s5 = integrated_line_flux(hr_phys, err_phys[b], wave, dwave, c2)
                if not (np.isfinite(f5) and np.isfinite(s5) and s5 > 0):
                    continue
                if f5 / s5 < args.snr_min:
                    continue

                lam0 = 0.5 * (c1 + c2)
                m = np.abs((wave - lam0) / lam0 * C_KMS) <= args.window_kms
                if m.sum() < 30:
                    continue

                row = [zb, f5 / s5]
                for a in arms:
                    # Both arguments must be the windowed slice; passing the
                    # full 6671-sample spectrum against a windowed abscissa
                    # raises inside least_squares and every fit returns NaN.
                    _, A1, A2, sig, ok = fit_doublet(
                        wave[m], spec[a][b][m], (c_hb, c1, c2),
                        p0_ratio=args.p0_ratio)
                    row += [A1, A2, sig, float(ok)]
                rows.append(row)

    r = np.array(rows, dtype=np.float64)
    print(f"\n{len(r)} galaxies with an HR-detected [O III] 5007 (SNR >= {args.snr_min:g})")
    print(f"fitting window +/-{args.window_kms:g} km/s about the doublet midpoint\n")

    print("=" * 74)
    print("[O III] 5007 / 4959 amplitude ratio -- atomic value is 2.98")
    print("=" * 74)
    print(f"  {'arm':>5} {'n_ok':>6} {'median':>8} {'16-84 spread':>18} "
          f"{'within 20%':>11} {'med |err|':>10} {'sig km/s':>9} {'4959 lost':>10}")

    summary = {}
    n_tot = len(r)
    for j, a in enumerate(arms):
        A1, A2, sig, ok = (r[:, 2 + 4 * j + k] for k in range(4))
        # A 4959 driven to the non-negativity bound is not a bad fit -- it is the
        # fit reporting that it could not separate the doublet at all. Counted
        # rather than silently dropped, because for the prism it is the result.
        lost = (ok > 0) & np.isfinite(A1) & (A1 <= 1e-6 * np.maximum(A2, 1e-12))
        good = (ok > 0) & np.isfinite(A1) & np.isfinite(A2) & ~lost & (A1 > 0)
        ratio = np.where(good, A2 / np.where(good, A1, np.nan), np.nan)
        v = ratio[np.isfinite(ratio)]
        if not v.size:
            print(f"  {a:>5} {0:>6}  -- no usable fits --")
            continue
        err = np.abs(v / ATOMIC_RATIO - 1.0)
        lo, hi = np.percentile(v, [16, 84])
        print(f"  {a:>5} {v.size:>6} {np.median(v):>8.2f} "
              f"{f'[{lo:.2f}, {hi:.2f}]':>18} {float((err < 0.2).mean()):>10.1%} "
              f"{float(np.median(err)):>10.2f} {float(np.median(sig[good])):>9.0f} "
              f"{float(lost.sum()) / n_tot:>9.1%}")
        summary[a] = dict(n=int(v.size), median=float(np.median(v)),
                          within20=float((err < 0.2).mean()),
                          med_err=float(np.median(err)),
                          lost_frac=float(lost.sum()) / n_tot)

    print("\n  'within 20%' is the number that matters: a ratio is only usable "
          "for\n  diagnostics if it is close to right on an individual galaxy, "
          "not merely\n  right on average. 'sig km/s' is a sanity check that "
          "each arm is being\n  fit at the resolution it actually has (~127 for "
          "R=1000, ~1273 for R=100).")

    # --- the crossover: where, if anywhere, does super-resolution pay? ---
    #
    # The aggregate above hides the only thing that decides how the paper should
    # frame this. The prism is photon-limited, so its accuracy climbs with S/N;
    # SR2 is limited by its own reconstruction error, so it does not. If the two
    # curves cross, super-resolution is a *low-S/N aid* with a measurable
    # crossover an observer can act on -- and since most real spectra sit at
    # modest S/N, that regime is the majority of any survey, not a corner case.
    snr = r[:, 1]
    print(f"\n{'=' * 74}")
    print("Within 20% of 2.98, split by HR [O III] 5007 S/N")
    print("=" * 74)
    print(f"  {'S/N bin':>12} {'n':>5} " + " ".join(f"{a:>8}" for a in arms))
    edges = [3, 5, 10, 20, 50, 150, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (snr >= lo) & (snr < hi)
        if sel.sum() < 5:
            continue
        cells = []
        for j in range(len(arms)):
            A1, A2, _, ok = (r[:, 2 + 4 * j + k] for k in range(4))
            lost = (ok > 0) & np.isfinite(A1) & (A1 <= 1e-6 * np.maximum(A2, 1e-12))
            good = sel & (ok > 0) & np.isfinite(A1) & np.isfinite(A2) & ~lost & (A1 > 0)
            v = (A2 / np.where(good, A1, np.nan))[good]
            v = v[np.isfinite(v)]
            cells.append(f"{100 * np.mean(np.abs(v / ATOMIC_RATIO - 1) < 0.2):>7.1f}%"
                         if v.size else f"{'--':>8}")
        lbl = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"
        print(f"  {lbl:>12} {int(sel.sum()):>5} " + " ".join(cells))
    print("\n  A bin where SR2 exceeds LR is a regime where super-resolution "
          "genuinely\n  pays. Read the n column before believing any single row.")

    np.savez(args.out, rows=r, arms=np.array(arms), atomic_ratio=ATOMIC_RATIO)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
