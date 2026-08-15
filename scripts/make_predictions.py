#!/usr/bin/env python
"""Run the chain once and cache everything the paper figures need.

The figures should not each re-run the models. One cache per (split, SR2
checkpoint) means every figure in the paper is generated from provably the same
predictions, and regenerating after a retrain is one command plus replotting.

    # after a new SR2 lands, on validation
    python scripts/make_predictions.py --sr2-ckpt train/sr2_best/best_sr2.pth

    # final numbers, once, on a frozen model
    python scripts/make_predictions.py --sr2-ckpt <frozen> --split test --allow-test

Output: ``cache/predictions_<split>.npz``, holding LR, SR1, SR2 and
HR spectra in physical units, their uncertainties, redshifts, the wavelength
grid, per-row provenance, and the checkpoint paths it came from.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from specsr.evaluation import BASELINE, DEFAULT_DATASET, load_pipeline, predict

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr2-ckpt", default=None,
                    help="omit to cache the LR/SR1/ZHead chain only")
    ap.add_argument("--sr1-ckpt", default=str(BASELINE / "best_superres_model.pth"))
    ap.add_argument("--sr1-config", default=str(BASELINE / "config_logR.yaml"))
    ap.add_argument("--zhead-ckpt", default=str(BASELINE / "best_zhead.pth"))
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--allow-test", action="store_true",
                    help="required for --split test; final numbers only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    p = load_pipeline(sr2_ckpt=args.sr2_ckpt, sr1_ckpt=args.sr1_ckpt,
                      sr1_config=args.sr1_config, zhead_ckpt=args.zhead_ckpt,
                      dataset=args.dataset)
    res = predict(p, args.split, dataset=args.dataset, allow_test=args.allow_test)

    default_out = REPO / "cache" / f"predictions_{args.split}.npz"
    out = Path(args.out) if args.out else default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **res)

    n = len(res["z_true"])
    print(f"\ncached {n} spectra -> {out}")
    print(f"  arrays: {', '.join(k for k in sorted(res) if k != 'provenance')}")
    if "sr2" in res:
        d1 = float(np.mean((res['sr1'] - res['flux_high']) ** 2))
        d2 = float(np.mean((res['sr2'] - res['flux_high']) ** 2))
        print(f"  physical-space MSE vs HR: SR1 {d1:.4e}   SR2 {d2:.4e}")
        print(f"  presence_mean: {res['presence'].mean():.5f}")


if __name__ == "__main__":
    main()
