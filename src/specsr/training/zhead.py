"""Stage 2: train the redshift head, over a swappable input representation.

One loop, four inputs. ``--source`` selects what the head sees:

==========  =====================================
``lowres``  the observed prism spectrum
``hires``   the observed grating spectrum
``sr1``     the SR1 reconstruction (+ its sigma)
``sr2``     the SR1 -> SR2 refinement (+ its sigma)
==========  =====================================

This replaces four near-duplicate scripts (``train/redshift_head{,_hires,
_lowres,_sr2}/train_z_head.py``, 2,534 lines between them) that had drifted apart
by hundreds of lines. That mattered scientifically, not just tidily: the paper's
redshift comparison claims the four arms differ only in their *input*, and with
four separately-edited scripts that was an assumption nobody could check. Here it
is structural -- one architecture, one loss, one optimiser, one split, one seed,
and the only thing that varies is :mod:`specsr.training.zhead_sources`.

The one irreducible difference is the input channel count: ``lowres``/``hires``
feed 1 channel, ``sr1``/``sr2`` feed 2 (value and predictive sigma). That changes
the first convolution and nothing else, and the parameter difference is reported
at startup so it can be quoted rather than hand-waved.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..data.datasets import FixedGridSpectraDataset
from ..data.splits import get_training_split
from ..metrics import redshift_metrics
from ..models.loaders import load_sr1, load_sr2, load_zhead
from ..models.zhead import ZHead1D, heteroscedastic_nll, redshift_pdf_loss
from ..runtime import (
    apply_env_overrides,
    describe_overrides,
    limit_batches,
    resolve_dataset,
    wandb_mode,
)
from .common import (
    build_param_groups,
    clip_grad_or_skip,
    log_checkpoint_artifact,
    resolve_out_dir,
    set_seed,
    write_run_manifest,
)
from .zhead_sources import SOURCE_NAMES, build_source
from .ztransform import RedshiftTransform

__all__ = ["train_zhead", "SOURCE_NAMES"]

DEFAULTS: dict[str, Any] = {
    "batch_size": 64,
    "dropout": 0.2353384158962288,
    "epochs": 200,
    "hidden_dim": 128,
    "lr": 0.00035215595940893353,
    "num_blocks": 6,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    # Two-way split: 80% of galaxies train, 20% evaluate. Evaluation is
    # restricted to original spectra -- the product augments every galaxy,
    # so an unfiltered 20% would be 95% synthetic rows and would report
    # performance on perturbations over 21x correlated samples.
    "train_frac": 0.8,
    "val_frac": 0.2,
    "seed": 42,
    "z_var_floor": 1e-6,
    "use_sigma_channel": True,
    # v2 architecture (see specsr.models.zhead): position-aware trunk. The v1
    # head was translation-invariant and therefore could not read line
    # *positions* -- the quantity redshift actually lives in.
    "coord_channel": True,
    "dilation_growth": 2,
    "pooling": "attn",
    # Classification head: the redshift posterior is genuinely multimodal (a
    # misidentified line puts a second peak elsewhere), and a single Gaussian
    # cannot represent that -- it splits the difference and lands in the valley
    # between two modes, which is exactly a catastrophic outlier.
    "head": "softmax",
    "n_z_bins": 1024,
    "soft_argmax_half": 8,
    "pdf_target_sigma_bins": 1.5,
    # Train the mean with a fixed variance first: a free variance from epoch 1
    # lets the head inflate sigma instead of learning mu.
    "mean_warmup_epochs": 10,
    # Selection and early stopping run on the metric the paper quotes, not on
    # val NLL -- NLL is dominated by variance calibration and, on 286 val
    # spectra, is noisy enough to select lucky epochs.
    "select_metric": "med_abs_dz_over_1pz",
    "patience": 30,
    "min_epochs": 40,
    # Every redshift run logs the predicted-vs-true panel to W&B, not only
    # scalars -- see the call site for why a number is not enough here.
    "plot_every": 5,
    "wandb_project": "spectral-superresolution",
}


def train_zhead(
    source: str = "sr1",
    config: dict | None = None,
    *,
    dataset: str | Path | None = None,
    out_dir: str | Path | None = None,
    sr1_ckpt: str | Path | None = None,
    sr1_config: str | Path | None = None,
    sr2_ckpt: str | Path | None = None,
    zhead_bootstrap_ckpt: str | Path | None = None,
) -> Path:
    """Train the redshift head on ``source`` and return its output directory.

    ``sr1``/``sr2`` sources need frozen upstream checkpoints. The ``sr2`` source
    additionally needs a *bootstrap* redshift head, because SR2 is conditioned on
    a redshift estimate: that head is frozen and is emphatically not the head
    being trained here, which only ever sees the finished SR2 spectrum.
    """
    import wandb

    if source not in SOURCE_NAMES:
        raise ValueError(f"unknown source {source!r}; choose from {SOURCE_NAMES}")

    cfg = dict(DEFAULTS)
    cfg.update(config or {})
    cfg = apply_env_overrides(cfg)
    print(f"[specsr] {describe_overrides()}", flush=True)

    set_seed(int(cfg["seed"]))
    out = resolve_out_dir(out_dir, f"zhead_{source}")
    dataset_path = resolve_dataset(str(dataset or cfg.get("dataset_npz", "")))
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Identical split and seed across all four arms -- this is what makes the
    # comparison a comparison.
    train_idx, val_idx, split_path = get_training_split(
        dataset_path, float(cfg["train_frac"]), float(cfg["val_frac"]), int(cfg["seed"])
    )
    ds = FixedGridSpectraDataset(dataset_path, normalize_flux=True)
    bs = int(cfg["batch_size"])
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=bs, shuffle=True)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=bs, shuffle=False)

    # Fit the redshift transform on the TRAINING split only. Using all redshifts
    # would leak the held-out range into the normalisation.
    ztf = RedshiftTransform.from_training_redshifts(np.asarray(ds.z)[train_idx])
    print(f"z transform: mean={ztf.mean:.4f} std={ztf.std:.4f} "
          f"bounds_n=[{ztf.z_min_n:.3f},{ztf.z_max_n:.3f}]", flush=True)

    src_kwargs: dict[str, Any] = {"use_sigma": bool(cfg["use_sigma_channel"])}
    if source in ("sr1", "sr2"):
        if sr1_ckpt is None or sr1_config is None:
            raise ValueError(f"source={source!r} requires --sr1-ckpt and --sr1-config")
        sr1, _ = load_sr1(sr1_config, sr1_ckpt, device)
        src_kwargs["sr1"] = sr1
    if source == "sr2":
        if sr2_ckpt is None or zhead_bootstrap_ckpt is None:
            raise ValueError("source='sr2' requires --sr2-ckpt and --zhead-bootstrap-ckpt")
        with np.load(dataset_path, allow_pickle=True) as d:
            wave = np.asarray(d["wavelength_high"], dtype=np.float32)
        from ..models.lines import LINE_LIST_REST_AA
        line_rest = np.asarray([w for _, w in LINE_LIST_REST_AA], dtype=np.float32) * 1e-4
        sr2, sr2_cfg = load_sr2(sr2_ckpt, wave, line_rest, device)
        boot, *_ = load_zhead(zhead_bootstrap_ckpt, device)
        src_kwargs.update(
            sr2=sr2, zhead_bootstrap=boot, ztransform=ztf,
            delta_cap=float(sr2_cfg.get("delta_cap", 0.0)),
            use_sr1_sigma=bool(sr2_cfg.get("use_sr1_sigma", True)),
            use_line_mask=bool(sr2_cfg.get("use_line_mask", True)),
            use_zhat_channel=bool(sr2_cfg.get("use_zhat_channel", True)),
            use_zsigma_channel=bool(sr2_cfg.get("use_zsigma_channel", False)),
            zsigma_line_mask=bool(sr2_cfg.get("zsigma_line_mask", False)),
            zsigma_mask_max_um=float(sr2_cfg.get("zsigma_mask_max_um", 0.05)),
            sigma_base_um=float(sr2_cfg.get("sigma_base_um", 0.005)),
        )

    src = build_source(source, **src_kwargs)

    zhead = ZHead1D(
        in_channels=src.n_channels,
        hidden_dim=int(cfg["hidden_dim"]),
        num_blocks=int(cfg["num_blocks"]),
        dropout=float(cfg["dropout"]),
        coord_channel=bool(cfg["coord_channel"]),
        dilation_growth=int(cfg["dilation_growth"]),
        pooling=str(cfg["pooling"]),
        head=str(cfg["head"]),
        n_z_bins=int(cfg["n_z_bins"]),
        z_grid_min_n=ztf.z_min_n,
        z_grid_max_n=ztf.z_max_n,
        soft_argmax_half=int(cfg["soft_argmax_half"]),
    ).to(device)
    is_pdf_head = str(cfg["head"]) == "softmax"
    n_params = sum(p.numel() for p in zhead.parameters())
    print(f"source={source}  in_channels={src.n_channels}  pooling={cfg['pooling']}  "
          f"head={cfg['head']}  "
          f"zhead params={n_params:,}", flush=True)

    opt = torch.optim.AdamW(
        build_param_groups(zhead, lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(cfg["epochs"]), eta_min=float(cfg["lr"]) * 0.01
    )

    wandb.init(
        mode=wandb_mode(),
        project=cfg["wandb_project"],
        name=cfg.get("wandb_name") or f"zhead_{source}_{int(time.time())}",
        tags=["zhead", "redshift", source],
        config={**cfg, "source": source, "in_channels": src.n_channels,
                "zhead_params": n_params, "dataset_npz": dataset_path, "out_dir": str(out)},
    )

    best_sel = float("inf")
    best_z = None
    best_nll = float("inf")
    no_improve = 0
    epochs_total = int(cfg["epochs"])
    # Keep the warmup strictly inside the run so a shortened smoke run still
    # exercises the NLL phase.
    # The PDF head has no separate variance parameter, so there is nothing for
    # a mean-only warmup to protect against.
    warmup = 0 if is_pdf_head else min(int(cfg["mean_warmup_epochs"]), epochs_total // 2)
    select_metric = str(cfg["select_metric"])
    # Canonical name, matching what every consumer looks for. This was
    # `best_zhead_{source}.pth` until 2026-08-14, which meant a directory
    # written by the trainer could not be loaded by `specsr infer
    # --checkpoints`, `SpecSRPipeline.from_directory` or
    # `specsr.evaluation` -- all of which look for `best_zhead.pth`, and the
    # gap was bridged only by `run_all_stages.sh` renaming it into the bundle.
    #
    # The suffix existed to keep the four comparison arms apart, but each arm
    # is trained into its own `--out-dir`, so the directory already carries
    # that distinction. The arm is recorded inside the checkpoint (`source`)
    # and in `run_manifest.json` besides. On the Hub, where all four arms share
    # one folder, they do still need distinct names -- see
    # `specsr.checkpoints._REGISTRY`, which is unaffected by this.
    best_path = out / "best_zhead.pth"
    pred_path = out / f"predictions_{source}.npz"

    for epoch in range(epochs_total):
        in_warmup = epoch < warmup
        zhead.train()
        tr_loss = tr_n = n_skipped = 0
        for batch in limit_batches(train_loader, kind="train"):
            x_in = src(_as_dict(batch), device)
            z = batch[3].to(device).float()
            if is_pdf_head:
                # No mean-warmup phase: there is no separate variance parameter
                # to run away, the spread is a property of the PDF itself.
                loss = redshift_pdf_loss(
                    zhead.logits(x_in), ztf.normalize(z), zhead.z_grid_n,
                    target_sigma_bins=float(cfg["pdf_target_sigma_bins"]),
                )
            else:
                mu_raw, log_var = zhead(x_in)
                log_var = torch.clamp(log_var, min=-12.0, max=12.0)
                if in_warmup:
                    loss = ((ztf.decode_mean(mu_raw) - ztf.normalize(z)) ** 2).mean()
                else:
                    loss = heteroscedastic_nll(
                        ztf.decode_mean(mu_raw), log_var, ztf.normalize(z),
                        var_floor=float(cfg["z_var_floor"]),
                    )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(cfg["grad_clip"]) > 0:
                _, ok = clip_grad_or_skip(zhead.parameters(), float(cfg["grad_clip"]))
                if not ok:
                    n_skipped += 1
                    opt.zero_grad(set_to_none=True)
                    continue
            opt.step()
            tr_loss += float(loss.item())
            tr_n += 1

        zhead.eval()
        va_loss = va_n = 0
        preds, truths, sigmas = [], [], []
        with torch.no_grad():
            for batch in limit_batches(val_loader, kind="val"):
                x_in = src(_as_dict(batch), device)
                z = batch[3].to(device).float()
                if is_pdf_head:
                    logits = zhead.logits(x_in)
                    va_loss += float(redshift_pdf_loss(
                        logits, ztf.normalize(z), zhead.z_grid_n,
                        target_sigma_bins=float(cfg["pdf_target_sigma_bins"]),
                    ).item())
                    mu_raw, log_var = zhead.moments(logits)
                else:
                    mu_raw, log_var = zhead(x_in)
                    log_var = torch.clamp(log_var, min=-12.0, max=12.0)
                    va_loss += float(heteroscedastic_nll(
                        ztf.decode_mean(mu_raw), log_var, ztf.normalize(z),
                        var_floor=float(cfg["z_var_floor"]),
                    ).item())
                va_n += 1
                z_hat, z_sig = ztf.predict(mu_raw, log_var, float(cfg["z_var_floor"]),
                                           bounded=zhead.bounded_mean)
                preds.append(z_hat.reshape(-1).cpu().numpy())
                truths.append(z.reshape(-1).cpu().numpy())
                sigmas.append(z_sig.reshape(-1).cpu().numpy())

        sched.step()
        val_loss = va_loss / max(1, va_n)
        # Same function the sweeps rank on and the paper figure quotes.
        m = redshift_metrics(np.concatenate(truths), np.concatenate(preds))
        # Log the predicted-vs-true panel, not just scalars. A redshift run's
        # scalars cannot show *how* it is failing, and the distinction that
        # matters most here -- ordinary scatter about the diagonal versus a
        # second locus off it, from a line misidentified as a different
        # transition -- is visible only in the scatter. Every N epochs, because
        # rendering costs more than the epoch does on a 286-spectrum split.
        log_extra: dict[str, Any] = {}
        if (epoch % int(cfg["plot_every"]) == 0) or (epoch + 1 == epochs_total):
            from ..wandb_plots import redshift_panel
            log_extra.update(redshift_panel(
                np.concatenate(truths), np.concatenate(preds),
                title=f"{source} z-head — epoch {epoch + 1}",
            ))

        wandb.log({**log_extra, "epoch": epoch, "train_loss_nll": tr_loss / max(1, tr_n),
                   "val_loss_nll": val_loss, "in_warmup": int(in_warmup),
                   "skipped_nonfinite_steps": n_skipped,
                   "learning_rate": float(opt.param_groups[0]["lr"]),
                   **{f"val_{k}": v for k, v in m.items()}})
        print(f"epoch {epoch + 1:3d}  val_nll={val_loss:.5f}  "
              f"sigma_nmad={m['sigma_nmad']:.5f}  "
              f"med|dz|/(1+z)={m['med_abs_dz_over_1pz']:.5f}  "
              f"outliers={m['outlier_frac']:.4f}"
              + ("  [warmup]" if in_warmup else ""), flush=True)

        sel = float(m[select_metric])
        # Don't checkpoint inside warmup: the variance head is untrained there,
        # so its NLL and sigma are meaningless even when the mean looks good.
        if not in_warmup and sel < best_sel:
            best_sel = sel
            best_nll = val_loss
            no_improve = 0
            # Predictions of the selected epoch, for the end-of-run figure. The
            # head early-stops, so the best epoch is often not one that landed
            # on `plot_every` and would otherwise never be plotted.
            best_z = (np.concatenate(truths), np.concatenate(preds))
            torch.save({
                "zhead_state_dict": zhead.state_dict(),
                "config": dict(cfg),
                "source": source,
                "use_sigma_channel": src.n_channels == 2,
                **ztf.as_dict(),
                "val_metrics": m,
            }, best_path)
            # Predictions from the *same* epoch as the checkpoint. Writing them
            # here rather than in a later inference pass means the figure cannot
            # end up showing a different epoch from the saved model -- which is
            # exactly the class of mismatch that made two SR2 evaluations
            # disagree earlier in this project.
            np.savez(
                pred_path,
                z_true=np.concatenate(truths),
                z_pred=np.concatenate(preds),
                z_sigma=np.concatenate(sigmas),
                source=source,
                epoch=epoch,
                checkpoint=str(best_path),
                dataset=dataset_path,
                split="val",
                **{k: v for k, v in m.items()},
            )
            print(f"  saved best -> {best_path}", flush=True)
            print(f"  predictions -> {pred_path}", flush=True)
        elif not in_warmup:
            no_improve += 1
            if no_improve >= int(cfg["patience"]) and epoch + 1 >= int(cfg["min_epochs"]):
                print(f"early stop at epoch {epoch + 1}: no {select_metric} improvement "
                      f"in {no_improve} epochs", flush=True)
                break

    if best_z is not None:
        from ..wandb_plots import FINAL_REDSHIFT_KEY, redshift_panel
        wandb.log(redshift_panel(
            best_z[0], best_z[1], key=FINAL_REDSHIFT_KEY,
            title=f"{source} z-head final — {select_metric}={best_sel:.6f}"))

    # `predictions_{source}.npz` stays local and is deliberately NOT uploaded:
    # it is regenerable from the checkpoint plus the split, both of which are.
    # W&B carries the weights and the provenance, not derived products.
    manifest = write_run_manifest(
        out, cfg, stage=f"zhead-{source}", dataset=dataset_path, split_path=split_path,
        upstream={"sr1_ckpt": sr1_ckpt, "sr1_config": sr1_config, "sr2_ckpt": sr2_ckpt,
                  "zhead_bootstrap_ckpt": zhead_bootstrap_ckpt},
    )
    log_checkpoint_artifact(
        [best_path, *manifest], kind=f"zhead-{source}",
        metadata={**cfg, "source": source, f"best_{select_metric}": best_sel},
    )
    print(f"ZHead[{source}] done. best {select_metric}={best_sel:.6f} "
          f"(val_nll at that epoch={best_nll:.5f})", flush=True)
    print(f"  checkpoint  {best_path}", flush=True)
    print(f"  predictions {pred_path}", flush=True)
    return out


def _as_dict(batch) -> dict:
    """Adapt the dataset's tuple to the mapping the sources expect."""
    return {"flux_low": batch[0], "flux_high": batch[1], "flux_high_err": batch[2], "z": batch[3]}
