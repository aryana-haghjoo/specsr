#!/usr/bin/env python3
"""Are the SR2-HR residuals structured at the emission lines, and by how much?

Settles by measurement what the residual map only suggests by eye: for every
catalogued line in range, compare |SR2-HR| inside a velocity window on the line
against a continuum window offset from it in the same spectrum, in the same
normalised units the residual map is drawn in.

Produces the on-line/off-line residual ratios quoted in the paper.  Run from the
repository root:

    python scripts/residual_line_structure.py
"""
import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
C = 299792.458

# Two regions are of particular interest: the [O III] doublet, and the stretch
# between Hbeta and [O II] where Hdelta/Hgamma sit. Both are covered here.
LINES = {
    "[O II] 3727":  0.3727,
    "Hdelta 4102":  0.41017,
    "Hgamma 4340":  0.43405,
    "Hbeta 4861":   0.48613,
    "[O III] 4959": 0.49590,
    "[O III] 5007": 0.50072,
    "Halpha 6563":  0.65628,
}

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--cache", default=str(REPO / "cache" / "predictions_val.npz"))
args = ap.parse_args()

d = np.load(args.cache, allow_pickle=True)
wave = d["wave"]
hr, sr, z = d["flux_high"], d["sr2"], d["z_true"]
hi_m, hi_s = d["hi_mean"][:, None], d["hi_std"][:, None]
hr_n, sr_n = (hr - hi_m) / hi_s, (sr - hi_m) / hi_s
resid = sr_n - hr_n
n = len(z)
print(f"{n} held-out spectra, wave {wave[0]:.2f}-{wave[-1]:.2f} um\n")

HALF = 500.0      # km/s, on-line window
OFF_LO, OFF_HI = 2000.0, 4000.0   # km/s, continuum reference either side


def windows(lam_obs):
    """(on-line mask, continuum mask) in velocity about lam_obs."""
    v = (wave - lam_obs) / lam_obs * C
    on = np.abs(v) <= HALF
    off = (np.abs(v) >= OFF_LO) & (np.abs(v) <= OFF_HI)
    return on, off


print(f"{'line':14s} {'N':>5s} {'|res| on':>9s} {'|res| off':>10s} "
      f"{'ratio':>6s} {'signed':>8s}")
print("-" * 60)
rows = {}
for name, rest in LINES.items():
    on_v, off_v, sgn_v = [], [], []
    for i in range(n):
        lam = rest * (1.0 + z[i])
        if not (wave[0] < lam < wave[-1]):
            continue
        on, off = windows(lam)
        if on.sum() < 5 or off.sum() < 20:
            continue
        # Require the line to be real in the reference, else "residual at the
        # line" is measuring noise at an arbitrary wavelength.
        cont = np.median(hr_n[i][off])
        peak = np.max(hr_n[i][on]) - cont
        if not np.isfinite(peak) or peak < 3 * np.std(hr_n[i][off]):
            continue
        on_v.append(np.median(np.abs(resid[i][on])))
        off_v.append(np.median(np.abs(resid[i][off])))
        sgn_v.append(np.median(resid[i][on]))
    if len(on_v) < 20:
        print(f"{name:14s} {len(on_v):5d}   (too few detections)")
        continue
    a, b, s = np.median(on_v), np.median(off_v), np.median(sgn_v)
    rows[name] = (len(on_v), a, b, a / b, s)
    print(f"{name:14s} {len(on_v):5d} {a:9.3f} {b:10.3f} {a/b:6.2f} {s:+8.3f}")

print("\nratio > 1 means the residual is larger on the line than beside it.")
print("signed < 0 means SR2 sits below HR at the line (under-estimated flux).")

# The vertical boundaries visible in the map: grating stitch points.
print("\n--- residual scatter by wavelength band (all rows) ---")
for lo, hi in [(1.0, 1.5), (1.5, 2.0), (2.0, 2.4), (2.4, 3.0),
               (3.0, 3.7), (3.7, 4.5), (4.5, 5.3)]:
    m = (wave >= lo) & (wave < hi)
    print(f"  {lo:.1f}-{hi:.1f} um : sigma(SR2-HR) = {np.nanstd(resid[:, m]):.3f}"
          f"   sigma(HR) = {np.nanstd(hr_n[:, m]):.3f}")
