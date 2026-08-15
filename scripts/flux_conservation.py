#!/usr/bin/env python
"""Line-flux conservation and SR2 branch ablation.

Three questions, one forward pass over a held-out split:

1. Does SR2 conserve integrated line flux? Measured as
   continuum-subtracted integrated flux in a velocity window around each
   expected line, SR2 vs HR truth, with SR1 as the baseline to beat.

2. Which SR2 branch is responsible for its MSE being worse than SR1's?
   ``delta = line_delta + cnn_delta`` (sr2.py), so capturing ``cnn_delta``
   with a forward hook gives ``line_delta`` by subtraction -- an exact
   split rather than a re-implementation that could drift from the model.

3. How much of the flux deficit is carried by galaxies whose redshift the
   frozen head got wrong? Every window below is centred on the *true* redshift,
   while SR2 is conditioned on the head's estimate, so ``dv = c*(zhat-z)/(1+z)``
   is the distance between the two. Measured 2026-08-02, the aggregate ratio is
   a mixture of two regimes rather than a property of the model's amplitudes:
   ~0.75 for the 63% of bright lines with ``|dv| < 500 km/s`` and ~0.29 for the
   rest, with the head's median ``|dv|`` at 532 km/s against a +/-500 km/s core.

   Note this is **not** the window missing the line: re-measuring SR2's flux in
   the model's own ``zhat``-centred window does not recover it (0.581 vs 0.618),
   so the model under-emits rather than misplaces. SR1 splits the same way
   (0.579 -> 0.112) despite never seeing a redshift, and HR SNR is flat across
   the bins, so ``dv`` is standing in for objects whose prism spectrum does not
   determine the line structure -- it is a conditioning axis, not a cause.

Defaults to the validation split. The test split is reserved for the final
number on a frozen model: iterating against it would quietly turn it into a
second validation set.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

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
# The chain the paper reports and the Hub serves (v4). Verified 2026-08-13 to
# reproduce cache/flux_conservation_results.npz to 2e-29 in every column.
SR1_RELEASE = REPO / "runs/finetune_20260730_003724/sr1"
ZHEAD_RELEASE = REPO / "runs/zhead_pdf_8020/sr1"
SR2_RELEASE = REPO / "runs/sr2_maskfix_20260803_170711"

C_KMS = 299792.458


def integrated_line_flux(flux, err, wave, dwave, center, *,
                         core_kms=500.0, sb_lo_kms=800.0, sb_hi_kms=1500.0):
    """Continuum-subtracted integrated flux in a velocity window.

    Returns ``(flux, sigma)`` in the units of ``flux * wave``. The continuum is
    the median of two sidebands, so the per-spectrum additive normalisation
    offset cancels and only the multiplicative scale survives -- which is why
    SR/HR ratios are meaningful in normalised space.
    """
    v = (wave - center) / center * C_KMS
    core = np.abs(v) <= core_kms
    side = (np.abs(v) >= sb_lo_kms) & (np.abs(v) <= sb_hi_kms)
    if core.sum() < 3 or side.sum() < 5:
        return np.nan, np.nan

    cont = np.median(flux[side])
    f = float(np.sum((flux[core] - cont) * dwave[core]))
    # Continuum error is subdominant (median over a wider band); propagate the
    # core only, which is the honest floor rather than an optimistic estimate.
    s = float(np.sqrt(np.sum((err[core] * dwave[core]) ** 2))) if err is not None else np.nan
    return f, s


# Lines the model carries as separate entries but this measurement cannot
# separate. The core is +/-500 km/s and [O II] 3726/3729 are 224 km/s apart, so
# each component's window already contains both and neither row is the doublet:
# the two disagree by ~9% on the reference flux and by more on the summed ratio,
# which is the choice of centre showing up, not a measurement. Measured once
# more at the conventional 3727 -- the centre the S/N figure fits its Gaussian
# at, so the two figures measure the same thing -- and written to a separate
# array, so every aggregate over ``rows`` still counts each line exactly once.
BLENDED_LINES = ((3727.0e-4, "[OII]_3727"),)
BLEND_INDEX0 = 1000  # past any real line index, so the two cannot collide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    # All three stages default into a frozen archive, never into a live run
    # directory: a running chain rewrites its checkpoints in place, so
    # evaluating mid-run silently compares SR2s sitting on different SR1s.
    #
    # These default to the *released* chain, not to RUN5. They used to default
    # to RUN5 throughout, which still loads and still prints a full summary --
    # but RUN5's redshift head is the superseded Gaussian one, and against the
    # current SR2 it puts the median galaxy 30,000 km/s from its true redshift
    # against a +/-500 km/s window, so every line falls outside its own window
    # and the flux ratio comes out near 0.04 instead of 0.61. A stale default
    # that fails loudly is a nuisance; one that returns a plausible wrong number
    # is a trap, which is why the paths are pinned here rather than documented.
    ap.add_argument("--ckpt", default=str(SR2_RELEASE / "best_sr2.pth"))
    ap.add_argument("--dataset", default=str(REPO / "data" / "paired_DR4_logR.npz"))
    ap.add_argument("--sr1-ckpt", default=str(SR1_RELEASE / "best_superres_model.pth"))
    # The fine-tuned SR1 run kept no config of its own; it is architecturally
    # the RUN5 config, which the release bundle carries.
    ap.add_argument("--sr1-config", default=str(REPO / "checkpoints/release/config_logR.yaml"))
    ap.add_argument("--zhead-ckpt", default=str(ZHEAD_RELEASE / "best_zhead_sr1.pth"))
    ap.add_argument("--snr-min", type=float, default=5.0,
                    help="minimum HR line SNR to enter the statistics")
    ap.add_argument("--allow-test", action="store_true",
                    help="required for --split test; final numbers only")
    ap.add_argument("--out", default=str(REPO / "evaluations" / "flux_conservation_results.npz"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]

    print(f"SR1   : {args.sr1_ckpt}")
    print(f"ZHead : {args.zhead_ckpt}")
    print(f"SR2   : {args.ckpt}")
    sr1, _ = load_sr1(args.sr1_config, args.sr1_ckpt, device)
    zhead, z_mean, z_std, use_sigma, _ = load_zhead(
        args.zhead_ckpt, device, unfreeze_last_n=0)

    data = np.load(args.dataset, allow_pickle=True)
    wave = np.asarray(data["wavelength_high"], dtype=np.float64)
    dwave = np.gradient(wave)
    wave_t = torch.tensor(wave, dtype=torch.float32, device=device)

    # Same construction the trainer uses, so token order matches the checkpoint.
    line_rest_um = np.asarray([w for _, w in LINE_LIST_REST_AA], dtype=np.float32) * 1e-4

    # get_training_split returns (train, val, split_path) and withholds the test
    # indices deliberately -- unpacking it as three index arrays silently hands
    # back a *path string* where indices belong. load_split resolves all three
    # properly and refuses the test set without an explicit opt-in.
    train_idx = load_split("train", args.dataset)
    idx = load_split(args.split, args.dataset, allow_test=args.allow_test)
    print(f"{args.split} split: {len(idx)} spectra")

    target_key = cfg.get("target_key_raw", "flux_high")
    ds = FixedGridSpectraDataset(args.dataset, normalize_flux=True, target_key_raw=target_key)

    # Errors come straight from the file, in physical units. The dataset class
    # scales them with `e_hi / (s_hi if s_hi > 1e-9 else 1.0)`, but HR std is
    # ~1e-21 here, so that test is false and the error is left unnormalised
    # while the flux is normalised to unit variance. Reading the npz directly
    # keeps the SNR cut meaningful regardless of that bug.
    err_phys_all = np.asarray(data[target_key + "_err"], dtype=np.float64)

    z_train = np.array(ds.z[train_idx], dtype=np.float32)
    # Decode redshift through the same transform the trainer uses, honouring the
    # head's own `bounded_mean`. Open-coding `sigmoid(mu_raw)` here silently
    # mis-decoded the softmax PDF head, whose estimate is already in normalised
    # units -- and a wrong zhat moves every line window off its line, which
    # would read as a flux-conservation failure.
    ztf = RedshiftTransform(
        mean=z_mean, std=z_std,
        z_min_n=float((z_train.min() - z_mean) / z_std),
        z_max_n=float((z_train.max() - z_mean) / z_std),
    )

    # sigma_z is needed if it either widens the line mask or enters as a channel.
    use_zsigma = bool(cfg.get("zsigma_line_mask", False)) or \
        bool(cfg.get("use_zsigma_channel", False))
    zsigma_max = float(cfg.get("zsigma_mask_max_um", 0.05))

    in_ch = sr2_input_channels(
        use_sr1_sigma=bool(cfg["use_sr1_sigma"]),
        use_line_mask=bool(cfg["use_line_mask"]),
        use_zhat_channel=bool(cfg["use_zhat_channel"]),
        use_zsigma_channel=bool(cfg.get("use_zsigma_channel", False)),
    )
    sr2 = SR2Attention(
        in_channels=in_ch, line_rest_um=line_rest_um, wave_hi_um=wave.astype(np.float32),
        line_dim=int(cfg["line_dim"]), num_attn_heads=int(cfg["num_attn_heads"]),
        num_attn_layers=int(cfg["num_attn_layers"]), window_half=int(cfg["window_half"]),
        cnn_dim=int(cfg["cnn_dim"]), num_cnn_blocks=int(cfg["num_cnn_blocks"]),
        dropout=float(cfg["dropout"]), cnn_scale=float(cfg.get("cnn_scale", 1.0)),
    ).to(device)
    sr2.load_state_dict(ck["sr2_state_dict"])
    sr2.eval()

    # Exact branch split: hook the CNN head, subtract to get the line branch.
    grab = {}
    sr2.cnn_out.register_forward_hook(lambda m, i, o: grab.__setitem__("cnn", o.detach()))
    sr2.presence_head.register_forward_hook(
        lambda m, i, o: grab.__setitem__("pres", torch.sigmoid(o.detach())))

    # shuffle=False, so batches walk `idx` in order and this counter maps each
    # row back to its dataset index for the physical error lookup.
    idx = np.asarray(idx)
    cursor = 0
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idx), batch_size=32, shuffle=False)

    mse = {k: 0.0 for k in ("sr1", "full", "lines", "cnn")}
    n_seen = 0
    pres_all = []
    pres_snr = []  # (presence, HR SNR) for every in-range line, cut or not
    rows = []  # per-line measurements
    rows_blend = []  # same columns, for the blends in BLENDED_LINES
    dv_gal = []  # one redshift-error per galaxy, independent of the SNR cut

    with torch.no_grad():
        for batch in loader:
            x_low = torch.nan_to_num(batch[0].to(device).unsqueeze(1))
            x_high = torch.nan_to_num(batch[1].to(device).unsqueeze(1))
            # HR errors are read from the npz in physical units below, not from
            # the dataset class -- see the note at err_phys_all.
            z_true = batch[3].to(device).float()
            hi_mean = batch[4].numpy()
            hi_std = batch[5].numpy()

            sr1_mean, sr1_logvar = sr1(x_low)
            sr1_log_sigma = 0.5 * sr1_logvar

            z_in = torch.cat([sr1_mean, sr1_log_sigma], dim=1) if use_sigma else sr1_mean
            mu_raw, logvar_n = zhead(z_in)
            zhat_t, zsig_t = ztf.predict(mu_raw, logvar_n, bounded=zhead.bounded_mean)
            zhat = torch.nan_to_num(zhat_t.reshape(-1), nan=float(z_mean))
            z_sigma = torch.nan_to_num(zsig_t.reshape(-1), nan=zsigma_max) \
                if use_zsigma else None

            # The line mask is built inside the shared helper rather than
            # open-coded here: the channel order is load-bearing, and a second
            # copy would drift from the model.
            x_in = build_sr2_input(
                x_low=x_low, sr1_mean=sr1_mean, sr1_log_sigma=sr1_log_sigma, zhat=zhat,
                wave_hi_um=wave_t, line_rest_um=line_rest_um,
                use_sr1_sigma=bool(cfg["use_sr1_sigma"]),
                use_line_mask=bool(cfg["use_line_mask"]),
                use_zhat_channel=bool(cfg["use_zhat_channel"]),
                use_zsigma_channel=bool(cfg.get("use_zsigma_channel", False)),
                sigma_base_um=float(cfg["sigma_base_um"]),
                z_sigma=z_sigma, sigma_max_um=zsigma_max,
            )

            delta_raw, _ = sr2(x_in, zhat)
            cnn_delta = grab["cnn"]
            line_delta = delta_raw - cnn_delta
            pres_all.append(grab["pres"].mean(dim=1).cpu().numpy())
            pres_b = grab["pres"].cpu().numpy()   # (B, K), kept per line

            cap = float(cfg["delta_cap"])
            variants = {
                "sr1": sr1_mean,
                "full": sr1_mean + constrain_delta(delta_raw, cap),
                "lines": sr1_mean + constrain_delta(line_delta, cap),
                "cnn": sr1_mean + constrain_delta(cnn_delta, cap),
            }
            for k, v in variants.items():
                mse[k] += float(((v - x_high) ** 2).mean().item()) * x_high.shape[0]
            n_seen += x_high.shape[0]

            # ---- line fluxes, de-normalised into HR physical space ----
            hr = x_high.squeeze(1).cpu().numpy()
            err_phys = err_phys_all[idx[cursor:cursor + hr.shape[0]]]
            cursor += hr.shape[0]
            zt = z_true.cpu().numpy()
            zh = zhat.cpu().numpy()
            phys = {k: v.squeeze(1).cpu().numpy() for k, v in variants.items()}

            for b in range(hr.shape[0]):
                s, m = float(hi_std[b]), float(hi_mean[b])
                hr_b = hr[b] * s + m
                er_b = err_phys[b]
                pv = {k: phys[k][b] * s + m for k in phys}
                # Distance between the window (true z) and the redshift SR2 was
                # conditioned on. Recorded per galaxy as well as per line, since
                # the head's error is a property of the object, not of the line.
                dv = C_KMS * (float(zh[b]) - float(zt[b])) / (1.0 + float(zt[b]))
                dv_gal.append(dv)

                for li, rest in enumerate(line_rest_um):
                    center = float(rest) * (1.0 + float(zt[b]))
                    if center < wave[0] or center > wave[-1]:
                        continue
                    f_hr, s_hr = integrated_line_flux(hr_b, er_b, wave, dwave, center)
                    if not np.isfinite(f_hr) or not np.isfinite(s_hr) or s_hr <= 0:
                        continue
                    # Recorded before the SNR cut, because the presence gate is
                    # judged by how differently it treats real and absent lines
                    # -- and the absent ones are exactly what the cut removes.
                    pres_snr.append((float(pres_b[b, li]), f_hr / s_hr))
                    if f_hr / s_hr < args.snr_min:
                        continue
                    row = [float(zt[b]), float(center), li, f_hr, s_hr]
                    for k in ("sr1", "full", "lines", "cnn"):
                        fk, _ = integrated_line_flux(pv[k], None, wave, dwave, center)
                        row.append(fk)
                    row.append(float(pres_b[b, li]))
                    row.append(dv)
                    rows.append(row)

                # Blends, measured the same way but kept apart. Deliberately a
                # separate loop over the same spectra rather than extra entries
                # in the one above: `rows` then stays exactly what it was, and
                # no aggregate over it can double-count a doublet.
                for j, (rest, _) in enumerate(BLENDED_LINES):
                    center = float(rest) * (1.0 + float(zt[b]))
                    if center < wave[0] or center > wave[-1]:
                        continue
                    f_hr, s_hr = integrated_line_flux(hr_b, er_b, wave, dwave, center)
                    if not np.isfinite(f_hr) or not np.isfinite(s_hr) or s_hr <= 0:
                        continue
                    if f_hr / s_hr < args.snr_min:
                        continue
                    row = [float(zt[b]), float(center), BLEND_INDEX0 + j, f_hr, s_hr]
                    for k in ("sr1", "full", "lines", "cnn"):
                        fk, _ = integrated_line_flux(pv[k], None, wave, dwave, center)
                        row.append(fk)
                    # Presence is a per-model-line gate; a blend of two of them
                    # has none, and inventing one would be a number nothing
                    # measured.
                    row.append(np.nan)
                    row.append(dv)
                    rows_blend.append(row)

    rows = np.array(rows, dtype=np.float64)
    rows_blend = np.array(rows_blend, dtype=np.float64).reshape(-1, rows.shape[1])
    pres_all = np.concatenate(pres_all)
    dv_gal = np.asarray(dv_gal, dtype=np.float64)

    print(f"\n{'='*66}")
    print(f"SR2 branch ablation -- MSE vs HR ({n_seen} spectra, {args.split})")
    print("=" * 66)
    base = mse["sr1"] / n_seen
    for k, label in [("sr1", "SR1 (baseline)"), ("full", "SR2 full"),
                     ("lines", "SR2 line branch only"), ("cnn", "SR2 CNN branch only")]:
        v = mse[k] / n_seen
        print(f"  {label:24s} {v:.6f}   gain vs SR1: {base - v:+.6f}")

    print(f"\n  presence_mean over split: {pres_all.mean():.5f} "
          f"(min {pres_all.min():.5f}, max {pres_all.max():.5f})")

    # The mean alone cannot tell a working gate from a constant one. What
    # matters is whether presence is *higher* on lines the reference actually
    # shows; a ratio near 1 means the gate is a blanket multiplier on every line
    # amplitude, which is how the line branch came to supply ~0% of line flux.
    ps = np.asarray(pres_snr, dtype=np.float64)
    real, absent = ps[ps[:, 1] >= 20.0, 0], ps[ps[:, 1] < 3.0, 0]
    if len(real) and len(absent):
        print(f"  presence on real lines   (HR SNR>=20, n={len(real):6d}): "
              f"median {np.median(real):.5f}")
        print(f"  presence on absent lines (HR SNR< 3,  n={len(absent):6d}): "
              f"median {np.median(absent):.5f}")
        print(f"  discrimination (real/absent): "
              f"{np.median(real) / max(np.median(absent), 1e-9):.2f}x"
              "   [1.0 = the gate carries no information]")

    print(f"\n{'='*66}")
    print(f"Line flux conservation ({len(rows)} lines, HR SNR >= {args.snr_min})")
    print("=" * 66)
    if len(rows) == 0:
        print("  no lines passed the SNR cut")
    else:
        f_hr = rows[:, 3]
        for j, k in zip(range(5, 9), ("sr1", "full", "lines", "cnn"), strict=True):
            r = rows[:, j] / f_hr
            r = r[np.isfinite(r)]
            print(f"  {k:6s} ratio to HR:  median {np.median(r):+.3f}   "
                  f"mean {np.mean(r):+.3f}   scatter(16-84) "
                  f"[{np.percentile(r, 16):+.3f}, {np.percentile(r, 84):+.3f}]")

    # ---- conditioned on how well the head placed the galaxy ----
    #
    # The single aggregate above is a mixture. SR2's line branch is conditioned
    # on `zhat`, but the windows are centred on the true redshift, and the core
    # is only +/-500 km/s wide -- so a galaxy the head missed contributes a
    # near-zero ratio no matter how well calibrated the amplitudes are. Report
    # both regimes; the mixture on its own is not interpretable.
    print(f"\n{'=' * 66}")
    print("Redshift quality of the frozen head, and flux conditioned on it")
    print("=" * 66)
    adv = np.abs(dv_gal)
    q = np.percentile(adv, [25, 50, 75, 90])
    print(f"  |dv| = c|zhat - z|/(1+z) over {len(dv_gal)} galaxies, km/s:")
    print(f"    p25 {q[0]:8.1f}   p50 {q[1]:8.1f}   p75 {q[2]:8.1f}   p90 {q[3]:8.1f}")
    # One sample of the log grid is dln(lambda) = 2.5e-4, i.e. 74.96 km/s.
    print(f"    median offset in grid samples: {q[1] / 74.96:.1f}"
          f"   (measurement core is +/-500 km/s = +/-6.7 samples)")
    print(f"    beyond the core (|dv| > 500):  {float((adv > 500).mean()):.1%}"
          f"   catastrophic (> 2500): {float((adv > 2500).mean()):.1%}")

    if len(rows):
        dv_line = np.abs(rows[:, 10])
        f_hr = rows[:, 3]

        def _med(col, m):
            r = rows[m, col] / f_hr[m]
            r = r[np.isfinite(r)]
            return float(np.median(r)) if r.size else float("nan")

        print(f"\n  SR/HR flux ratio by the galaxy's |dv| (HR SNR >= {args.snr_min:g}):")
        print(f"    {'|dv| km/s':>14} {'n':>6} {'SR1':>8} {'SR2':>8}")
        edges = [0.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, np.inf]
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            m = (dv_line >= lo) & (dv_line < hi)
            if not m.any():
                continue
            lbl = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"
            print(f"    {lbl:>14} {int(m.sum()):>6} {_med(5, m):>8.3f} {_med(6, m):>8.3f}")

        good = dv_line < 500.0
        print("\n  the two regimes, which is what the text should quote:")
        print(f"    |dv| <  500 km/s : {int(good.sum()):4d} lines ({good.mean():.1%})"
              f"   SR1 {_med(5, good):.3f}   SR2 {_med(6, good):.3f}")
        print(f"    |dv| >= 500 km/s : {int((~good).sum()):4d} lines ({(~good).mean():.1%})"
              f"   SR1 {_med(5, ~good):.3f}   SR2 {_med(6, ~good):.3f}")

    np.savez(args.out, rows=rows, rows_blend=rows_blend, presence=pres_all,
             presence_snr=ps, dv_galaxy=dv_gal,
             mse={k: mse[k] / n_seen for k in mse}, split=args.split)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
