"""Build the paper figures from a prediction cache.

Orchestration only: which figures exist, which cache keys they need, where they
are written. Every figure itself is a function in :mod:`specsr.plotting`, so it
can be imported and tested without rendering.

Both ``scripts/make_figures.py`` and ``specsr evaluate`` call
:func:`build_figures`, so there is one definition of what "the paper figures"
means rather than a script and a CLI that can drift apart.
"""
from __future__ import annotations

import argparse
import pathlib
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # no display on the training box

from .. import linefit, metrics, plotting  # noqa: E402
from ..paths import output_dir  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


def build_parser(ap: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Figure-selection arguments, shared by the script and the CLI."""
    ap = ap or argparse.ArgumentParser()
    ap.add_argument("--dataset",
                    default=str(REPO / "data" / "paired_DR4_logR.npz"))
    ap.add_argument("--cache", default=str(REPO / "cache" / "predictions_val.npz"))
    # Default to the package's own output directory rather than a path that only
    # exists in the development checkout. Point SPECSR_OUTPUT_DIR at the paper's
    # plots directory, or pass --out, to regenerate manuscript figures in place.
    ap.add_argument("--out", default=str(output_dir("figures")))
    ap.add_argument("--spectrum-z", type=float, nargs="*", default=None,
                    help="pick panels nearest these redshifts instead of "
                         "auto-selecting the clearest doublet examples")
    ap.add_argument("--n-spectra", type=int, default=2,
                    help="how many example spectra when auto-selecting")
    ap.add_argument("--flux-rows",
                    default=str(REPO / "cache" / "flux_conservation_results.npz"),
                    help="output of scripts/flux_conservation.py, for the line-flux figure")
    ap.add_argument("--coverage-target", default=None,
                    help="target id for the disperser-coverage figure")
    ap.add_argument("--coverage-field", default="goods-s")
    # Named rather than auto-selected: every parent now carries exactly n_aug
    # realizations, so "the galaxy with the most augmentations" is a tie across
    # the whole training split and resolves to whichever row happens to be
    # first. This is the same galaxy as the submitted version of the figure
    # (obj_id 990 of the retired pre_process_DR4 build, labelled z=1.9988
    # there), identified by its line comb -- Halpha, [S III] 9069, 9531,
    # He I 10830 and Pa beta -- since the old row ids did not survive the
    # rebuild. Keeping it means the figure changes only in the grid it is
    # drawn on, which is the point the revision needs it to make.
    ap.add_argument("--aug-target", default="00028626",
                    help="target id for the augmentation figure")
    # The redshift figure's three arms are separate training runs, so their
    # predictions are passed in rather than read from the main cache.
    ap.add_argument("--z-lowres", default=None, dest="z_lowres",
                    help="predictions .npz from `train zhead --source lowres`")
    ap.add_argument("--z-hires", default=None, dest="z_hires",
                    help="predictions .npz from `train zhead --source hires`")
    ap.add_argument("--z-sr2", default=None, dest="z_sr2",
                    help="predictions .npz from `train zhead --source sr2`")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of figure names to build")
    return ap


def build_figures(args) -> list[str]:
    """Build the requested figures. Returns the names actually written."""

    # Every figure this produces goes into the manuscript, so they all get the
    # paper's font. Set once here rather than per plotting function: this is the
    # single entry point both `scripts/make_figures.py` and `specsr evaluate`
    # go through, so a figure cannot be built for the paper without it.
    plotting.plt.rcParams.update(plotting.PAPER_RC)

    cache = Path(args.cache)
    if not cache.exists():
        raise SystemExit(
            f"no prediction cache at {cache}\n"
            "Build one first:  python scripts/make_predictions.py --sr2-ckpt <ckpt>"
        )
    d = np.load(cache, allow_pickle=True)
    out = Path(args.out)
    print(f"cache: {cache}  ({len(d['z_true'])} spectra, split={d['split']})")

    wave, hr, lr, sr = d["wave"], d["flux_high"], d["flux_low"], d["sr2"]
    built = []

    def want(name):
        return args.only is None or name in args.only

    if want("spectrum"):
        # The paper's axis is normalised flux. HR and SR share the HR moments
        # (SR is predicted in HR-normalised space); LR is standardised by its
        # own statistics, exactly as the dataset does it.
        def _norm(a):
            m, sd = np.nanmean(a), np.nanstd(a)
            return (a - m) / (sd if sd > 0 else 1.0)

        if args.spectrum_z:
            chosen = [int(np.argmin(np.abs(d["z_true"] - zt))) for zt in args.spectrum_z]
        else:
            # Pick objects where the [O III] doublet is blended in LR and
            # separated in SR -- the point the figure exists to make -- rather
            # than hard-coding indices that stop meaning anything after a
            # retrain or a resplit.
            ranked = metrics.rank_doublet_examples(
                wave, lr, sr, hr, d["z_true"], z_pred=d["z_pred"])
            if len(ranked) == 0:
                raise SystemExit("no spectra with the [O III] doublet in range")
            chosen = [int(k) for k in ranked[:args.n_spectra]]

        panels, picked = [], []
        for i in chosen:
            hi_m, hi_s = float(d["hi_mean"][i]), float(d["hi_std"][i])
            panels.append({
                "hr": (hr[i] - hi_m) / hi_s,
                "sr": (sr[i] - hi_m) / hi_s,
                "lr": _norm(lr[i]),
                "z_true": float(d["z_true"][i]),
                "z_pred": float(d["z_pred"][i]),
                "z_sigma": float(d["z_sigma"][i]),
            })
            dl = metrics.doublet_dip_depth(lr[i], wave, d["z_true"][i])
            ds = metrics.doublet_dip_depth(sr[i], wave, d["z_true"][i])
            picked.append(f"row {int(d['row_index'][i])} z={d['z_true'][i]:.3f} "
                          f"dip LR={dl[0]:.2f}->SR={ds[0]:.2f}")
        fig = plotting.plot_spectra_with_inset(
            panels, wave, output_path=out / "generated_spectra.png")
        plotting.plt.close(fig)
        built.append("generated_spectra.png  (" + "; ".join(picked) + ")")

    # Normalised flux, matching the paper's colourbar. These products compare
    # spectra against each other, so a per-spectrum scale would let the
    # brightest few objects set the range for everyone. Figures 5 and 6 are read
    # side by side and must be built from the same arrays -- the PSD used to be
    # taken on physical flux, where a median across objects spanning orders of
    # magnitude in brightness is set by the brightest of them.
    def _normalised_products():
        hi_m = d["hi_mean"][:, None]
        hi_s = d["hi_std"][:, None]
        lr_n = (lr - np.nanmean(lr, axis=1, keepdims=True)) / \
            np.nanstd(lr, axis=1, keepdims=True)
        return lr_n, (hr - hi_m) / hi_s, (sr - hi_m) / hi_s

    if want("residual_maps"):
        lr_n, hr_n, sr_n = _normalised_products()

        # Noise references, so the scatter panel can be read against the data's
        # own noise rather than in the abstract.
        sig_lr = metrics.estimate_median_noise_lambda(lr_n)
        sig_hr = metrics.estimate_median_noise_lambda(hr_n)
        p = plotting.plot_residual_maps(
            wave, lr_n, hr_n, sr_n, z_true=d["z_true"],
            sigma_lr_med=sig_lr, sigma_hr_med=sig_hr,
            residual_units="normalized flux",
            output_path=out / "residual_maps.png",
        )
        plotting.plt.close(p)
        built.append("residual_maps.png")

        # The same figure without the emission-line overlay. The dashed tracks
        # sit exactly on the ridges they label, so they hide the pixels a reader
        # checking those ridges needs to see, so an unobstructed copy is built
        # alongside rather than by hand -- the two can then never disagree about
        # which predictions they came from.
        p = plotting.plot_residual_maps(
            wave, lr_n, hr_n, sr_n, z_true=d["z_true"],
            sigma_lr_med=sig_lr, sigma_hr_med=sig_hr,
            residual_units="normalized flux",
            show_tracks=False,
            output_path=out / "residual_maps_no_tracks.png",
        )
        plotting.plt.close(p)
        built.append("residual_maps_no_tracks.png")

    if want("psd"):
        # Two panels, as the paper reads it: how much fine-scale structure each
        # product carries (left), and what is left over between them (right).
        lr_n, hr_n, sr_n = _normalised_products()
        fig, hf = plotting.plot_residual_psd(
            wave, [lr_n - hr_n, lr_n - sr_n, sr_n - hr_n],
            labels=["LR − HR", "LR − SR", "SR − HR"],
            signal_maps=[lr_n, sr_n, hr_n],
            signal_labels=["LR", "SR", "HR"],
            output_path=out / "psd.png",
        )
        plotting.plt.close(fig)
        built.append("psd.png  (HF frac: " +
                     ", ".join(f"{lab} {v:.3f}" for lab, v in hf) + ")")

    if want("line_flux"):
        # Integrated line flux against the HR reference. Needs the measurements
        # from scripts/flux_conservation.py, not the raw prediction cache.
        fr = Path(args.flux_rows)
        if not fr.exists():
            print(f"\nskipping line_flux: no measurements at {fr}\n"
                  "  run evaluations/flux_conservation.py --out <path> first")
        else:
            fd = np.load(fr, allow_pickle=True)
            rows = fd["rows"]
            # Blended measurements live in their own array so that aggregates
            # over `rows` count each line once; the figure selects lines by
            # index, so it can draw from both.
            if "rows_blend" in fd and len(fd["rows_blend"]):
                rows = np.vstack([rows, fd["rows_blend"]])
            # The same four lines as the S/N figure, in the same columns, so
            # §4.3 can pair detectability with fidelity line by line rather than
            # across two different sets. [O II] is index 1000, the doublet
            # measured in a single window (flux_conservation.BLENDED_LINES),
            # which is what the S/N figure's single Gaussian at 3727 fits too;
            # the model's own 3726 and 3729 entries are 224 km/s apart inside a
            # +/-500 km/s window and neither is the doublet.
            names = {1000: r"[O II] $\lambda3727$", 54: r"H$\beta$",
                     56: r"[O III] $\lambda5007$", 72: r"H$\alpha$"}
            fig, st = plotting.plot_line_flux_comparison(
                rows, names, output_path=out / "line_flux_comparison.png")
            plotting.plt.close(fig)
            built.append("line_flux_comparison.png  (" + ", ".join(
                f"{k} {v['total_ratio']:.2f}" for k, v in st.items()) + ")")

    if want("snr"):
        # Gaussian fits per line per spectrum -- the slow step, so it is opt-in.
        lines = {"OII3727": 0.3727, "Hbeta": 0.4861,
                 "OIII5007": 0.5007, "Halpha": 0.6563}
        # Normalised flux, as the original did. S/N is a ratio so it is
        # scale-free in principle, but normalising keeps the fit's initial
        # guesses and bounds in the range they were tuned for.
        hm, hs = d["hi_mean"][:, None], d["hi_std"][:, None]
        lrn = (lr - np.nanmean(lr, axis=1, keepdims=True)) / \
            np.nanstd(lr, axis=1, keepdims=True)
        snr = linefit.measure_line_snr(
            wave, d["z_true"],
            {"LR": lrn, "SR": (sr - hm) / hs, "HR": (hr - hm) / hs},
            list(lines.values()), list(lines))
        # mincnt=1: the original was tuned for 4,986 augmented rows, but the
        # leak-free validation split is 286 real galaxies, so a threshold of 2
        # hides most bins and the panels come out nearly empty.
        fig = plotting.plot_snr_comparison(
            snr, mincnt=1, output_path=out / "sn_comparison.png")
        plotting.plt.close(fig)
        np.savez_compressed(out / "sn_comparison_values.npz", **snr)
        built.append("sn_comparison.png (+ values npz)")

    if want("redshift"):
        # Needs all three arms of the redshift comparison. They are separate
        # training runs (`specsr train zhead --source ...`), so this figure
        # cannot be built from the main prediction cache alone.
        arms = {}
        for name, key in (("Low-res baseline", "lowres"),
                          ("SR2 z-head", "sr2"),
                          ("High-res oracle", "hires")):
            path = Path(getattr(args, f"z_{key}", None) or "")
            if path.is_file():
                with np.load(path, allow_pickle=True) as zd:
                    arms[name] = (np.asarray(zd["z_true"]).ravel(),
                                  np.asarray(zd["z_pred"]).ravel())
        if len(arms) == 3:
            fig, zstats = plotting.plot_redshift_comparison(
                list(arms.values()), titles=tuple(arms),
                output_path=out / "redshift_comparison.png")
            plotting.plt.close(fig)
            built.append("redshift_comparison.png")
            for title, m in zip(arms, zstats, strict=True):
                print(f"  {title:20s} med|dz|/(1+z)={m['med_abs_dz_over_1pz']:.4f} "
                      f"outlier={m['outlier_frac'] * 100:.2f}%")
        elif args.only and "redshift" in args.only:
            missing = [k for k in ("lowres", "sr2", "hires")
                       if not Path(getattr(args, f"z_{k}", None) or "").is_file()]
            raise SystemExit(
                "redshift_comparison needs all three arms; missing predictions for "
                f"{', '.join(missing)}. Train them with "
                "`specsr train zhead --source <arm>` and pass --z-<arm> <predictions.npz>."
            )

    if want("augmentation"):
        # Reads the product directly: augmentations live only in the training
        # split, so they are not in a validation prediction cache.
        with np.load(args.dataset, allow_pickle=True) as raw:
            parent = np.asarray(raw["parent_id"])
            is_orig = np.asarray(raw["is_original"], bool)
            zz = np.asarray(raw["z"], float)
            targets = np.asarray(raw["target_id"]).astype(str)
            augmented = set(np.unique(parent[~is_orig]).tolist())
            hit = np.where((targets == str(args.aug_target)) & is_orig)[0]
            if hit.size and int(parent[hit[0]]) in augmented:
                best = int(parent[hit[0]])
            else:
                # Held-out galaxies carry no augmentations, so a target that is
                # missing or on the wrong side of the split cannot be drawn.
                best = sorted(augmented)[0]
                print(f"  augmentation: target {args.aug_target} is not an "
                      f"augmented training galaxy; using parent {best}")
            sel = np.where(parent == best)[0]
            sel = np.concatenate([sel[is_orig[sel]], sel[~is_orig[sel]]])
            def _rownorm(a, valid):
                # Samples the shift pulled in from beyond the grid are stored as
                # a literal 0. Drawing them would put a flat line at the edge of
                # every strongly shifted realization, which reads as data rather
                # than as the absence of it -- and would drag the normalization
                # with it. Blank them before, not after, taking the moments.
                a = np.where(np.asarray(valid, bool), np.asarray(a, float), np.nan)
                m = np.nanmean(a, axis=1, keepdims=True)
                sd = np.nanstd(a, axis=1, keepdims=True)
                return (a - m) / np.where(sd > 0, sd, 1.0)

            fig = plotting.plot_augmentation_family(
                np.asarray(raw["wavelength_low"]), np.asarray(raw["wavelength_high"]),
                _rownorm(raw["flux_low"][sel], raw["valid_low"][sel]),
                _rownorm(raw["flux_high"][sel], raw["valid_high"][sel]),
                zz[sel], output_path=out / "augmentation.png")
        plotting.plt.close(fig)
        built.append(f"augmentation.png  (parent {best}, {len(sel) - 1} augmentations)")

    if want("coverage"):
        # Native grating/prism spectra, so this reads the raw tree rather than
        # the built product, which has already stitched and resampled them.
        import os

        from specsr.data.ingest import discover_spectra, group_by_target, read_spectrum

        root = os.environ.get("SPECSR_JADES_ROOT")
        if not root:
            print("\nskipping coverage: set SPECSR_JADES_ROOT to the raw tree")
        else:
            groups = group_by_target(
                discover_spectra(pathlib.Path(root) / "DR4"), require_gratings=3)
            # Only useful if the galaxy is also in the built product: the line
            # markers need its redshift, and a target with no match would put
            # every line at z=0.
            with np.load(args.dataset, allow_pickle=True) as raw:
                tids = np.asarray(raw["target_id"]).astype(str)
                zs = np.asarray(raw["z"], float)
                zmap = {}
                for t, zv in zip(tids, zs, strict=True):
                    zmap.setdefault(t, zv)
            key = (args.coverage_field, args.coverage_target) \
                if args.coverage_target else None
            if key is None or key not in groups:
                usable = [k for k in groups
                          if str(k[1]) in zmap and 1.0 < zmap[str(k[1])] < 4.0]
                if not usable:
                    usable = [k for k in groups if str(k[1]) in zmap]
                if not usable:
                    raise SystemExit("no target present in both the raw tree and the product")
                key = usable[0]
            disp_map = {"clear-prism": "prism", "f070lp-g140m": "g140m",
                        "f170lp-g235m": "g235m", "f290lp-g395m": "g395m"}
            spec, zc = {}, None
            for raw_key, short in disp_map.items():
                if raw_key not in groups[key]:
                    continue
                rec = read_spectrum(groups[key][raw_key][0].path)
                f = np.asarray(rec["flux"], float)
                sd = np.nanstd(f)
                spec[short] = (np.asarray(rec["wavelength"], float),
                               (f - np.nanmean(f)) / (sd if sd > 0 else 1.0))
            zc = float(zmap.get(str(key[1]), 0.0))
            fig = plotting.plot_disperser_coverage(
                spec, zc, lines=plotting.COVERAGE_LINES,
                output_path=out / "matched_spectra_comparison.png")
            plotting.plt.close(fig)
            built.append(f"matched_spectra_comparison.png  ({key[0]} {key[1]}, z={zc:.3f})")

    print("\nwrote:")
    for b in built:
        print(f"  {out}/{b}")


    return built
