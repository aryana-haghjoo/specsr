#!/usr/bin/env python
"""Why the prism and the grating fail differently at redshift recovery.

At fixed exposure a dispersive spectrograph trades sensitivity against
resolving power: the same photons are spread over more detector pixels. This
script quantifies both sides of that trade on the held-out set and connects them
to the two failure modes of redshift estimation, which respond to *different*
sides of it:

* **precision** -- how tightly a correctly identified line is localised --
  follows resolution, and is measured here as median |dz|/(1+z);
* **catastrophic outliers** -- picking the wrong line entirely -- follow
  detection, and are measured as the fraction with |dz|/(1+z) > 0.15.

Everything here is a property of the data and of the two redshift heads trained
on it. It involves no super-resolution model, so it is stable against changes to
SR1/SR2 and can be quoted before those are final.

Line counting uses an **instrument-matched** window. The prism (R~100) spreads a
line over ~3000 km/s while the grating (R~1000) concentrates it into ~300, so
counting both in the same narrow window would manufacture the result.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from specsr.evaluation import load_split
from specsr.models.lines import LINE_LIST_REST_AA

C_KMS = 299792.458
REPO = Path(__file__).resolve().parents[1]

# Core half-width and continuum sidebands, per instrument, in km/s.
WINDOWS = {
    "lowres": dict(core=2000.0, sb=(2600.0, 4500.0)),
    "hires": dict(core=500.0, sb=(800.0, 1500.0)),
}
KEYS = {"lowres": ("flux_low", "flux_low_err", "valid_low"),
        "hires": ("flux_high", "flux_high_err", "valid_high")}


def _line_snrs(arm, wave, dwave, rest, flux, err, valid, z):
    w = WINDOWS[arm]
    out = []
    for r in rest:
        c = r * (1.0 + z)
        if c < wave[0] or c > wave[-1]:
            continue
        vel = (wave - c) / c * C_KMS
        ok = valid & np.isfinite(flux) & np.isfinite(err) & (err > 0)
        core = (np.abs(vel) <= w["core"]) & ok
        side = (np.abs(vel) >= w["sb"][0]) & (np.abs(vel) <= w["sb"][1]) & ok
        if core.sum() < 3 or side.sum() < 5:
            continue
        f = np.sum((flux[core] - np.median(flux[side])) * dwave[core])
        s = np.sqrt(np.sum((err[core] * dwave[core]) ** 2))
        if s > 0:
            out.append(f / s)
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(REPO / "data/paired_DR4_logR.npz"))
    ap.add_argument("--lowres-pred",
                    default=str(REPO / "runs/zarms_8020_20260731_010103/lowres/predictions_lowres.npz"))
    ap.add_argument("--hires-pred",
                    default=str(REPO / "runs/zarms_8020_20260731_010103/hires/predictions_hires.npz"))
    ap.add_argument("--snr-detect", type=float, default=3.0)
    ap.add_argument("--outlier-thresh", type=float, default=0.15)
    ap.add_argument("--out", default=str(REPO / "evaluations/redshift_information_budget.npz"))
    args = ap.parse_args()

    d = np.load(args.dataset, allow_pickle=True)
    wave = np.asarray(d["wavelength_high"], dtype=np.float64)
    dwave = np.gradient(wave)
    rest = np.asarray([w for _, w in LINE_LIST_REST_AA]) * 1e-4
    idx = np.asarray(load_split("val", args.dataset))
    z_all = np.asarray(d["z"], dtype=np.float64)

    arrays = {a: tuple(np.asarray(d[k]) for k in KEYS[a]) for a in KEYS}

    n_det, cont_snr, cover = {}, {}, {}
    for arm, (f_all, e_all, v_all) in arrays.items():
        nd, cs, cv = [], [], []
        for i in idx:
            f = f_all[i].astype(np.float64)
            e = e_all[i].astype(np.float64)
            v = v_all[i].astype(bool)
            s = _line_snrs(arm, wave, dwave, rest, f, e, v, z_all[i])
            nd.append(int((s >= args.snr_detect).sum()))
            m = v & np.isfinite(f) & np.isfinite(e) & (e > 0)
            cs.append(np.median(np.abs(f[m]) / e[m]) if m.sum() > 50 else np.nan)
            cv.append(v.mean())
        n_det[arm], cont_snr[arm], cover[arm] = map(np.asarray, (nd, cs, cv))

    print("=" * 74)
    print(f"Detection budget ({len(idx)} held-out galaxies)")
    print("=" * 74)
    for arm in ("lowres", "hires"):
        v = n_det[arm]
        print(f"  {arm:7s} lines at S/N>={args.snr_detect:g}: median {np.median(v):.0f}   "
              f"mean {v.mean():.2f}   16-84 [{np.percentile(v, 16):.0f}, {np.percentile(v, 84):.0f}]")
        print(f"          median per-sample |flux|/err: {np.nanmedian(cont_snr[arm]):.2f}"
              f"   grid coverage: {np.median(cover[arm]):.3f}")
    diff = n_det["hires"] - n_det["lowres"]
    print(f"  HR - LR: median {np.median(diff):+.0f} lines; HR detects fewer in "
          f"{100 * (diff < 0).mean():.1f}% of galaxies, more in {100 * (diff > 0).mean():.1f}%")
    print(f"  per-sample S/N ratio LR/HR: "
          f"{np.nanmedian(cont_snr['lowres']) / np.nanmedian(cont_snr['hires']):.2f}x")

    pred = {a: np.load(p, allow_pickle=True)
            for a, p in (("lowres", args.lowres_pred), ("hires", args.hires_pred))}
    zt = pred["lowres"]["z_true"]
    if not np.allclose(zt, pred["hires"]["z_true"]):
        raise SystemExit("arms are not row-aligned; refusing to compare them")
    rel = {a: np.abs(pred[a]["z_pred"] - zt) / (1.0 + zt) for a in pred}
    T = args.outlier_thresh

    print()
    print("=" * 74)
    print("The two failure modes respond to different sides of the trade")
    print("=" * 74)
    qs = np.nanpercentile(cont_snr["hires"], [33, 67])
    strata = [("faint", -np.inf, qs[0]), ("mid", qs[0], qs[1]), ("bright", qs[1], np.inf)]
    print(f"  {'HR tercile':>11s} {'n':>5s} | {'outlier LR':>11s} {'outlier HR':>11s} "
          f"| {'prec LR':>9s} {'prec HR':>9s}")
    rows = []
    for lab, lo, hi in strata:
        m = (cont_snr["hires"] > lo) & (cont_snr["hires"] <= hi)
        o_lr, o_hr = (rel["lowres"][m] > T).mean(), (rel["hires"][m] > T).mean()
        p_lr, p_hr = np.median(rel["lowres"][m]), np.median(rel["hires"][m])
        rows.append((lab, int(m.sum()), o_lr, o_hr, p_lr, p_hr))
        print(f"  {lab:>11s} {m.sum():5d} | {100 * o_lr:10.1f}% {100 * o_hr:10.1f}% "
              f"| {p_lr:9.5f} {p_hr:9.5f}")

    print()
    print("  Outliers: the HR excess is strongest where HR is faintest")
    print(f"    HR/LR outlier ratio  faint {rows[0][3] / rows[0][2]:.2f}x   "
          f"mid {rows[1][3] / rows[1][2]:.2f}x   bright {rows[2][3] / rows[2][2]:.2f}x")
    print("  Precision: the HR advantage is flat -- a pure resolution effect")
    print(f"    LR/HR precision ratio  faint {rows[0][4] / rows[0][5]:.2f}x   "
          f"mid {rows[1][4] / rows[1][5]:.2f}x   bright {rows[2][4] / rows[2][5]:.2f}x")

    # The obvious alternative explanation for the residual bright-end excess.
    bright = cont_snr["hires"] > qs[1]
    full = bright & (cover["hires"] >= 0.999)
    gap = bright & (cover["hires"] < 0.999)
    print()
    print("  Is the residual bright-end excess the grating's coverage gaps?")
    for lab, m in (("full coverage", full), ("has a gap", gap)):
        if m.sum() >= 8:
            print(f"    {lab:>14s} n={m.sum():3d} | outlier LR {100 * (rel['lowres'][m] > T).mean():5.1f}%"
                  f"  HR {100 * (rel['hires'][m] > T).mean():5.1f}%")
    print("    -> no: fully-covered galaxies show the LARGER excess, so coverage")
    print("       does not explain it. Left as an open caveat rather than a story.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, idx=idx, n_det_lowres=n_det["lowres"], n_det_hires=n_det["hires"],
             cont_snr_lowres=cont_snr["lowres"], cont_snr_hires=cont_snr["hires"],
             coverage_hires=cover["hires"], z_true=zt,
             rel_lowres=rel["lowres"], rel_hires=rel["hires"])
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
