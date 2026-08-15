"""Stage 3: train the physics-informed residual refiner.

SR2 predicts a correction to SR1's output, built from two branches:
``delta = line_delta + cnn_scale * cnn_delta``. The line branch places Gaussians
at the observed wavelengths of catalogued transitions, given a redshift from the
frozen head; the CNN branch handles continuum and everything between lines.

Ported from ``train/sr2_best/train_sr2_attention.py``. SR1 and the redshift head
are frozen (the head optionally unfreezes its last few layers). The channel stack
is assembled by :func:`specsr.models.sr2.build_sr2_input` rather than open-coded,
because its order is load-bearing and it used to be duplicated in four places.

Three guards exist because of runs that failed silently
------------------------------------------------------
``lam_sparse_warmup`` holds the sparsity penalty at zero for the first N epochs.
At initialisation the line amplitudes are ~0, so the likelihood exerts no upward
pull on the presence gate while sparsity pushes it down at a constant rate.
Presence reaches zero, which zeroes the amplitude gradient in turn, and both
heads are stuck: an absorbing state, not a slow patch. Amplitudes have to become
useful first.

``presence_floor`` aborts the run if the gate collapses anyway. A collapsed SR2
still trains for hours and still logs a respectable redshift-outlier rate --
because that metric barely depends on the line branch -- so without this the
failure is invisible until someone plots a spectrum.

``presence_sep_floor`` exists because the two guards above were both satisfied
by a model whose line branch was contributing nothing. Presence sat at 0.0038,
comfortably clear of the floor, but it was *the same value on real and absent
lines* -- so the gate had stopped selecting anything and had become a constant
0.002 multiplier on every amplitude, and the line branch supplied 0.17% of the
flux the reference required. A magnitude floor cannot detect that; only the
ratio between presence on real lines and on absent ones can. This is also why
``lam_sparse`` now defaults to zero and presence is supervised instead.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ..data.datasets import FixedGridSpectraDataset
from ..data.splits import get_training_split
from ..models.lines import LINE_LIST_REST_AA
from ..models.loaders import load_sr1, load_zhead
from ..models.sr2 import (
    SR2Attention,
    build_line_mask,
    build_sr2_input,
    constrain_delta,
    sr2_input_channels,
)
from ..runtime import (
    apply_env_overrides,
    describe_overrides,
    limit_batches,
    resolve_dataset,
    wandb_mode,
)
from .common import (
    clip_grad_or_skip,
    log_checkpoint_artifact,
    resolve_out_dir,
    set_seed,
    state_fingerprint,
    write_run_manifest,
)
from .losses import line_presence_target, line_window_flux, sr2_loss
from .ztransform import RedshiftTransform

__all__ = ["train_sr2"]

DEFAULTS: dict[str, Any] = {
    "batch_size": 32,
    "epochs": 200,
    "lr": 1e-4,
    "weight_decay": 2e-5,
    "grad_clip": 0.5,
    # Two-way split: 80% of galaxies train, 20% evaluate. Evaluation is
    # restricted to original spectra -- the product augments every galaxy,
    # so an unfiltered 20% would be 95% synthetic rows and would report
    # performance on perturbations over 21x correlated samples.
    "train_frac": 0.8,
    "val_frac": 0.2,
    "seed": 42,
    "use_sr1_sigma": True,
    "use_line_mask": True,
    "use_zhat_channel": True,
    # Feed the redshift *uncertainty* to SR2. `zsigma_line_mask` widens the line
    # mask by it and is free to enable -- the channel count is unchanged, so an
    # existing checkpoint can still be warm-started. `use_zsigma_channel` adds
    # sigma_z as a sixth input channel, which changes the input width and so
    # requires training SR2 from scratch.
    "zsigma_line_mask": True,
    "use_zsigma_channel": False,
    "zsigma_mask_max_um": 0.05,
    "sigma_base_um": 0.005,
    "lam_hp_in": 3.0,
    "lam_hp_out": 0.03,
    "hp_k": 163,
    "lam_z": 5.0,
    "lam_z_warmup": 5,
    # False on purpose; see specsr.training.losses.sr2_loss. Recorded in the run
    # config so the choice is visible in W&B rather than buried in code, because
    # it is a deliberate departure from a purely statistical objective.
    "hp_in_noise_weighted": False,
    # The blanket sparsity penalty is off: it drove the presence gate to a
    # constant and took the line branch with it. See specsr.training.losses.
    # Presence is now supervised against the reference spectrum instead.
    "lam_sparse": 0.0,
    "lam_sparse_warmup": 20,
    "lam_presence": 2.0,
    "presence_pos_weight": 0.0,  # 0 = balance to parity per batch
    # Integrated line flux, the headline flux-conservation metric. Warmed up
    # because it is meaningless until presence has learned to open at all: with
    # the gate shut every real line reads as a 100% flux deficit and the term
    # is a constant offset with no usable direction.
    "lam_flux": 2.0,
    "lam_flux_warmup": 5,
    "flux_core_half": 7,    # +/-500 km/s at 74.96 km/s per sample
    "flux_sb_lo": 11,       # 800 km/s
    # 2250 km/s. Wider than the 1500 the reported metric uses, because the
    # continuum is now estimated from line-free samples only and a blended line
    # needs somewhere clean to find. Widening costs little: the continuum is
    # smooth on this scale.
    "flux_sb_hi": 30,
    # Samples within this many of another catalogued line are not continuum.
    "flux_clean_half": 7,
    "flux_eps": 1.0,
    # Kernel sizes for the HR-derived mask that labels which lines are really
    # there. They set both the presence targets and which lines the flux term
    # sees, and they are in *samples*, so they are grid-dependent -- until
    # 2026-08-02 sr2_loss let them default to the retired linear grid's values
    # and the mask missed 55% of the lines with HR SNR >= 20. Chosen by
    # measurement against integrated HR line SNR, not by rescaling arithmetic;
    # see specsr.training.losses.sr2_loss for the table.
    "presence_mask_smooth_k": 259,
    "presence_mask_thresh_mad": 7.5,
    "presence_mask_dilate": 11,
    # 7 -> 5 is the whole fix: keep_only_wide needs this many *consecutive*
    # supra-threshold samples, and an R=1000 line on this R=4000 grid clears
    # 7.5 sigma over only ~5-6.
    "presence_mask_min_width": 5,
    # 1 weights each line by its HR flux, so the term optimises the bright lines
    # the paper's number is computed on rather than the far more numerous faint
    # ones. 0 recovers the unweighted mean over all mask positives.
    "flux_weight_power": 1.0,
    "presence_floor": 1e-3,
    # A *discrimination* floor, checked because `presence_floor` provably
    # cannot catch the failure that actually happened: presence sat at 0.0038,
    # four times above the floor, while being identical on real and absent
    # lines. Ratio of mean presence on real lines to mean presence on absent
    # ones; 1.0 is a gate that carries no information.
    "presence_sep_floor": 1.5,
    "line_dim": 128,
    "num_attn_heads": 4,
    "num_attn_layers": 4,
    "window_half": 80,
    "cnn_dim": 96,
    "num_cnn_blocks": 6,
    "dropout": 0.02,
    "cnn_scale": 1.0,  # 0.0 ablates the CNN continuum branch
    "delta_cap": 10.0,
    "logvar_min": float(np.log(0.05)),
    "logvar_max": float(np.log(10.0)),
    "var_floor": 1e-8,
    # 0 = the redshift head is fully frozen, and SR2 must leave it that way.
    #
    # This was 2, which reopened `attn_score` and `z_logits` (133,249 of 710,145
    # weights) at 0.1x the SR2 learning rate. Measured on the 572 held-out
    # galaxies with the trainer's own metric, the refinement objective degrades
    # the head immediately and monotonically -- catastrophic-outlier rate 10.84%
    # frozen, 14.69% after one epoch, 14.16% at epoch 7, 17.83% by epoch 30 --
    # consistently across three independent runs. The adapted head was then
    # discarded anyway, because the save block writes only `sr2_state_dict`.
    #
    # It also biased SR2's own checkpoint selection: `goal` contains
    # `z_outlier`, so the head's decay dominated the metric and dragged the pick
    # towards the first few epochs. Holding it constant moves the top-5 epochs
    # of run `1wp146ua` from [7, 6, 4, 9, 1] to [9, 15, 6, 7, 20].
    #
    # `_assert_zhead_untouched` enforces this at the end of training rather than
    # trusting the flag.
    "zhead_unfreeze_last_n": 0,
    "zhead_lr_mult": 0.1,
    # Every run logs validation pictures, not only scalars.
    "plot_every": 5,
    # goal = z_outlier - w_mse * mse_gain + w_flux * |1 - flux_ratio|
    "ckpt_mse_weight": 0.5,
    "ckpt_flux_weight": 0.2,
    "patience": 40,
    "min_epochs": 80,
    "wandb_project": "spectral-superresolution",
}


def train_sr2(
    config: dict | None = None,
    *,
    dataset: str | Path | None = None,
    out_dir: str | Path | None = None,
    sr1_ckpt: str | Path,
    sr1_config: str | Path,
    zhead_ckpt: str | Path,
    init_ckpt: str | Path | None = None,
) -> Path:
    """Train SR2 on top of a frozen SR1 and redshift head.

    ``init_ckpt`` warm-starts SR2's weights from an earlier ``best_sr2.pth``
    for fine-tuning -- typically against a *newer* upstream redshift head, which
    is exactly the situation where retraining from scratch wastes a night. The
    architecture flags in ``config`` must match the checkpoint's; loading is
    strict so a mismatch fails at launch.
    """
    import wandb

    cfg = dict(DEFAULTS)
    cfg.update(config or {})
    cfg = apply_env_overrides(cfg)
    print(f"[specsr] {describe_overrides()}", flush=True)

    set_seed(int(cfg["seed"]))
    out = resolve_out_dir(out_dir, "sr2")
    dataset_path = resolve_dataset(str(dataset or cfg.get("dataset_npz", "")))
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sr1, sr1_cfg = load_sr1(sr1_config, sr1_ckpt, device)
    zhead, z_mean, z_std, use_sigma, zhead_cfg = load_zhead(
        zhead_ckpt, device, unfreeze_last_n=int(cfg["zhead_unfreeze_last_n"])
    )
    # SR2 refines a reconstruction; it must not retune the instrument that
    # measures it. Fingerprinted here and checked at the end of training, so a
    # future change that reopens the head fails loudly instead of quietly
    # degrading the redshift arm that §4.4 rests on.
    zhead_fingerprint = state_fingerprint(zhead)

    train_idx, val_idx, split_path = get_training_split(
        dataset_path, float(cfg["train_frac"]), float(cfg["val_frac"]), int(cfg["seed"])
    )
    ds = FixedGridSpectraDataset(dataset_path, normalize_flux=True)
    bs = int(cfg["batch_size"])
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=bs, shuffle=True)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=bs, shuffle=False)

    z_train = np.asarray(ds.z)[train_idx]
    ztf = RedshiftTransform(
        mean=z_mean, std=z_std,
        z_min_n=float((z_train.min() - z_mean) / z_std),
        z_max_n=float((z_train.max() - z_mean) / z_std),
    )

    with np.load(dataset_path, allow_pickle=True) as d:
        wave_hi_um = np.asarray(d["wavelength_high"], dtype=np.float32)
    line_rest_um = np.asarray([w for _, w in LINE_LIST_REST_AA], dtype=np.float32) * 1e-4

    sr2 = SR2Attention(
        in_channels=sr2_input_channels(
            use_sr1_sigma=bool(cfg["use_sr1_sigma"]),
            use_line_mask=bool(cfg["use_line_mask"]),
            use_zhat_channel=bool(cfg["use_zhat_channel"]),
            use_zsigma_channel=bool(cfg["use_zsigma_channel"]),
        ),
        line_rest_um=line_rest_um, wave_hi_um=wave_hi_um,
        line_dim=int(cfg["line_dim"]), num_attn_heads=int(cfg["num_attn_heads"]),
        num_attn_layers=int(cfg["num_attn_layers"]), window_half=int(cfg["window_half"]),
        cnn_dim=int(cfg["cnn_dim"]), num_cnn_blocks=int(cfg["num_cnn_blocks"]),
        dropout=float(cfg["dropout"]), cnn_scale=float(cfg["cnn_scale"]),
    ).to(device)
    print(f"SR2Attention: {sum(p.numel() for p in sr2.parameters()):,} params", flush=True)

    if init_ckpt is not None:
        init = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        sr2.load_state_dict(init["sr2_state_dict"] if isinstance(init, dict)
                            and "sr2_state_dict" in init else init)
        print(f"[specsr] warm-started SR2 from {init_ckpt}", flush=True)

    zhead_params = [p for p in zhead.parameters() if p.requires_grad]
    groups = [{"params": list(sr2.parameters()), "lr": float(cfg["lr"])}]
    if zhead_params:
        groups.append({"params": zhead_params,
                       "lr": float(cfg["lr"]) * float(cfg["zhead_lr_mult"])})
    opt = torch.optim.AdamW(groups, weight_decay=float(cfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(cfg["epochs"]), eta_min=1e-6)

    run_cfg = {**cfg, "dataset_npz": dataset_path, "sr1_ckpt": str(sr1_ckpt),
               "sr1_config": str(sr1_config), "zhead_ckpt": str(zhead_ckpt),
               "out_dir": str(out), "init_ckpt": str(init_ckpt) if init_ckpt else None}
    wandb.init(
        mode=wandb_mode(), project=cfg["wandb_project"],
        name=cfg.get("wandb_name") or f"sr2_{int(time.time())}",
        tags=["sr2", "attention", "lines"], config=run_cfg,
    )

    # sigma_z is needed if it either widens the mask or enters as a channel.
    use_zsigma = bool(cfg["zsigma_line_mask"]) or bool(cfg["use_zsigma_channel"])
    wave_t = torch.tensor(wave_hi_um, device=device)
    best_goal = float("inf")
    best_example = None
    no_improve = 0
    best_path = out / "best_sr2.pth"
    sparse_warm = int(cfg["lam_sparse_warmup"])

    def forward(batch, *, train: bool):
        """Shared LR -> SR1 -> zhat -> SR2 path, so train and val cannot diverge."""
        x_low = torch.nan_to_num(batch[0].to(device).unsqueeze(1))
        x_high = torch.nan_to_num(batch[1].to(device).unsqueeze(1))
        x_err = torch.nan_to_num(batch[2].to(device).unsqueeze(1))

        with torch.no_grad():
            sr1_mean, sr1_logvar = sr1(x_low)
            sr1_log_sigma = 0.5 * sr1_logvar
            was_training = zhead.training
            zhead.eval()
            z_in = torch.cat([sr1_mean, sr1_log_sigma], dim=1) if use_sigma else sr1_mean
            mu_raw, logvar_n = zhead(z_in)
            zhat_t, zsig_t = ztf.predict(mu_raw, logvar_n, bounded=zhead.bounded_mean)
            zhat = torch.nan_to_num(zhat_t.reshape(-1), nan=float(z_mean))
            # A NaN uncertainty must not read as "certain": fall back to the
            # widest mask the cap allows rather than to zero.
            z_sigma = torch.nan_to_num(
                zsig_t.reshape(-1), nan=float(cfg["zsigma_mask_max_um"])
            ) if use_zsigma else None
            if was_training:
                zhead.train()
            line_mask = torch.nan_to_num(build_line_mask(
                wave_t, zhat, line_rest_um, sigma_base_um=float(cfg["sigma_base_um"]),
                z_sigma=z_sigma, sigma_max_um=float(cfg["zsigma_mask_max_um"])))

        x_in = build_sr2_input(
            x_low=x_low, sr1_mean=sr1_mean, sr1_log_sigma=sr1_log_sigma, zhat=zhat,
            wave_hi_um=wave_t, line_rest_um=line_rest_um,
            use_sr1_sigma=bool(cfg["use_sr1_sigma"]),
            use_line_mask=bool(cfg["use_line_mask"]),
            use_zhat_channel=bool(cfg["use_zhat_channel"]),
            use_zsigma_channel=bool(cfg["use_zsigma_channel"]),
            sigma_base_um=float(cfg["sigma_base_um"]),
            z_sigma=z_sigma, sigma_max_um=float(cfg["zsigma_mask_max_um"]),
        )
        if not torch.isfinite(x_in).all():
            return None

        out_ = sr2(x_in, zhat)
        delta_raw, sr2_logvar = out_[0], out_[1]
        presence = out_[2] if len(out_) > 2 else None
        presence_logit = out_[3] if len(out_) > 3 else None
        # Positions come from the model rather than being recomputed here: they
        # define where the line branch actually placed its Gaussians, and the
        # presence labels and flux windows have to line up with that exactly.
        line_pos, line_in_range = sr2.line_positions(zhat)
        if not (torch.isfinite(delta_raw).all() and torch.isfinite(sr2_logvar).all()):
            return None

        sr2_mean = sr1_mean + constrain_delta(delta_raw, float(cfg["delta_cap"]))
        sr2_logvar = sr2_logvar.clamp(float(cfg["logvar_min"]), float(cfg["logvar_max"]))
        if not torch.isfinite(sr2_mean).all():
            return None
        return dict(sr1_mean=sr1_mean, sr2_mean=sr2_mean, sr2_logvar=sr2_logvar,
                    x_high=x_high, x_err=x_err, line_mask=line_mask,
                    presence=presence, presence_logit=presence_logit,
                    line_pos=line_pos, line_in_range=line_in_range, zhat=zhat)

    for epoch in range(1, int(cfg["epochs"]) + 1):
        lam_z_eff = float(cfg["lam_z"]) * min(1.0, epoch / max(1, int(cfg["lam_z_warmup"])))
        lam_sparse_eff = 0.0 if epoch <= sparse_warm else float(cfg["lam_sparse"]) * min(
            1.0, (epoch - sparse_warm) / max(1, sparse_warm))
        lam_flux_eff = float(cfg["lam_flux"]) * min(
            1.0, epoch / max(1, int(cfg["lam_flux_warmup"])))

        sr2.train()
        if zhead_params:
            zhead.train()
        tr_loss = tr_pres = tr_sep = tr_flux_ratio = 0.0
        tr_sep_n = tr_ratio_n = 0
        tr_n = n_skipped = 0
        for batch in limit_batches(train_loader, kind="train"):
            f = forward(batch, train=True)
            if f is None:
                continue
            loss, comps = sr2_loss(
                sr2_mean=f["sr2_mean"], sr2_logvar=f["sr2_logvar"],
                x_high=f["x_high"], x_high_err=f["x_err"], line_mask=f["line_mask"],
                presence=f["presence"], zhead=zhead, z_true=batch[3].to(device).float(),
                ztransform=ztf, use_sigma=use_sigma, lam_z=lam_z_eff,
                lam_hp_in=float(cfg["lam_hp_in"]), lam_hp_out=float(cfg["lam_hp_out"]),
                hp_k=int(cfg["hp_k"]), lam_sparse=lam_sparse_eff,
                var_floor=float(cfg["var_floor"]),
                hp_in_noise_weighted=bool(cfg["hp_in_noise_weighted"]),
                presence_logit=f["presence_logit"],
                line_positions=f["line_pos"], line_in_range=f["line_in_range"],
                lam_presence=float(cfg["lam_presence"]),
                lam_flux=lam_flux_eff,
                presence_pos_weight=float(cfg["presence_pos_weight"]),
                flux_core_half=int(cfg["flux_core_half"]),
                flux_sb_lo=int(cfg["flux_sb_lo"]),
                flux_sb_hi=int(cfg["flux_sb_hi"]),
                flux_eps=float(cfg["flux_eps"]),
                flux_weight_power=float(cfg["flux_weight_power"]),
                flux_clean_half=int(cfg["flux_clean_half"]),
                presence_mask_smooth_k=int(cfg["presence_mask_smooth_k"]),
                presence_mask_thresh_mad=float(cfg["presence_mask_thresh_mad"]),
                presence_mask_dilate=int(cfg["presence_mask_dilate"]),
                presence_mask_min_width=int(cfg["presence_mask_min_width"]),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(cfg["grad_clip"]) > 0:
                _, ok = clip_grad_or_skip(sr2.parameters(), float(cfg["grad_clip"]))
                if not ok:
                    # A non-finite gradient carries no direction, and applying it
                    # would NaN every weight -- see clip_grad_or_skip.
                    n_skipped += 1
                    opt.zero_grad(set_to_none=True)
                    continue
            opt.step()
            tr_loss += float(loss.item())
            tr_pres += float(f["presence"].mean().item()) if f["presence"] is not None else 0.0
            # Batches with no real line in them yield no separation and no flux
            # ratio; averaging over the batches that *do* keeps a quiet batch
            # from dragging the epoch statistic towards zero.
            if np.isfinite(comps["presence_sep"]):
                tr_sep += comps["presence_sep"]
                tr_sep_n += 1
            if np.isfinite(comps["flux_ratio_bright"]):
                tr_flux_ratio += comps["flux_ratio_bright"]
                tr_ratio_n += 1
            tr_n += 1

        tr_n = max(1, tr_n)
        tr_loss /= tr_n
        tr_pres /= tr_n
        tr_sep = tr_sep / tr_sep_n if tr_sep_n else float("nan")
        tr_flux_ratio = tr_flux_ratio / tr_ratio_n if tr_ratio_n else float("nan")

        sr2.eval()
        zhead.eval()
        va_mse_sr1 = va_mse_sr2 = 0.0
        va_n = 0
        zp, zt = [], []
        va_ratio, va_fhr = [], []
        example = None
        with torch.no_grad():
            for batch in limit_batches(val_loader, kind="val"):
                f = forward(batch, train=False)
                if f is None:
                    continue
                if example is None:
                    example = (batch[0][0], f["sr2_mean"][0, 0], f["x_high"][0, 0],
                               batch[3][0], f["zhat"][0])
                va_mse_sr1 += float(F.mse_loss(f["sr1_mean"], f["x_high"]).item())
                va_mse_sr2 += float(F.mse_loss(f["sr2_mean"], f["x_high"]).item())

                # Line flux on the held-out set, so the checkpoint criterion can
                # see the quantity the paper reports. Accumulated per line
                # rather than per batch: the statistic is a median over lines,
                # and a mean of per-batch medians is not that.
                # Same mask settings as the training term. Left to default they
                # select a *different* line population from the one being
                # trained on, so `val/flux_ratio_bright` -- and through it the
                # checkpoint criterion -- would score the model on lines the
                # objective never asked it to reproduce.
                tgt_v, w_v = line_presence_target(
                    f["x_high"], f["line_pos"], in_range=f["line_in_range"],
                    core_half=int(cfg["flux_core_half"]),
                    smooth_k=int(cfg["presence_mask_smooth_k"]),
                    thresh_mad=float(cfg["presence_mask_thresh_mad"]),
                    dilate=int(cfg["presence_mask_dilate"]),
                    min_width=int(cfg["presence_mask_min_width"]))
                sel = (tgt_v * w_v) > 0
                if sel.any():
                    fkw = dict(
                        core_half=int(cfg["flux_core_half"]),
                        sb_lo=int(cfg["flux_sb_lo"]), sb_hi=int(cfg["flux_sb_hi"]),
                        all_positions=f["line_pos"],
                        clean_half=int(cfg["flux_clean_half"]))
                    fh = line_window_flux(f["x_high"], f["line_pos"], **fkw)
                    fs = line_window_flux(f["sr2_mean"], f["line_pos"], **fkw)
                    keep = sel & (fh.abs() > float(cfg["flux_eps"]))
                    if keep.any():
                        va_ratio.append((fs[keep] / fh[keep]).cpu())
                        va_fhr.append(fh[keep].abs().cpu())
                z_in2 = torch.cat([f["sr2_mean"], 0.5 * f["sr2_logvar"]], dim=1) \
                    if use_sigma else f["sr2_mean"]
                mu_z, lv_z = zhead(z_in2)
                z_hat, _ = ztf.predict(mu_z, lv_z, bounded=zhead.bounded_mean)
                zp.append(z_hat.reshape(-1).cpu())
                zt.append(batch[3].float().cpu())
                va_n += 1

        if not zp:
            # Every validation batch was rejected by the finite-check in
            # `forward`, which in practice means the weights are already NaN.
            # Say so, rather than letting torch.cat raise "expected a non-empty
            # list of Tensors" and leaving the actual cause to be guessed.
            nf = [n for n, p in sr2.named_parameters() if not torch.isfinite(p).all()]
            raise RuntimeError(
                f"SR2 produced no usable validation output at epoch {epoch}: every batch "
                f"was rejected as non-finite. Non-finite weight tensors: {len(nf)} of "
                f"{sum(1 for _ in sr2.parameters())}"
                + (f" (e.g. {nf[:3]})" if nf else "")
                + f". Training skipped {n_skipped} non-finite gradient step(s) this epoch."
            )

        va_n = max(1, va_n)
        mse_gain = va_mse_sr1 / va_n - va_mse_sr2 / va_n
        zp_t, zt_t = torch.cat(zp), torch.cat(zt)
        ok = torch.isfinite(zp_t) & torch.isfinite(zt_t)
        rel = (zp_t[ok] - zt_t[ok]).abs() / (1 + zt_t[ok].abs())
        z_outlier = float((rel > 0.15).float().mean()) if ok.any() else float("nan")
        sched.step()

        # Median SR/HR line flux over the brighter half of the held-out lines --
        # the population the paper's number is computed on.
        va_flux_ratio = float("nan")
        if va_ratio:
            r_all, f_all = torch.cat(va_ratio), torch.cat(va_fhr)
            good = torch.isfinite(r_all)
            if good.any():
                bright = good & (f_all >= f_all[good].median())
                va_flux_ratio = float(r_all[bright].median().item()) \
                    if bright.any() else float(r_all[good].median().item())

        # `goal` used to be blind to line flux, so it could not select a
        # flux-improved checkpoint -- and because MSE mildly prefers a smooth
        # prediction to a slightly mis-scaled sharp line, a real flux gain could
        # even read as a regression. The penalty is on |1 - ratio| so that
        # over-shooting is punished as much as starving.
        goal = z_outlier - float(cfg["ckpt_mse_weight"]) * mse_gain \
            if np.isfinite(z_outlier) else float("inf")
        if np.isfinite(goal) and np.isfinite(va_flux_ratio):
            goal = goal + float(cfg["ckpt_flux_weight"]) * abs(1.0 - va_flux_ratio)

        # SR2's failure mode is manufacturing lines in the wrong place, which is
        # invisible in a loss curve and obvious in a spectrum.
        log_extra: dict[str, Any] = {}
        if example is not None and (
            (epoch % int(cfg["plot_every"]) == 0) or (epoch == int(cfg["epochs"]))
        ):
            from ..wandb_plots import spectrum_panel
            lo, sr_, hi, zt, zp = example
            log_extra.update(spectrum_panel(
                wave_hi_um, lo, sr_, hi, z_true=zt, z_pred=zp,
                title=f"SR2 — epoch {epoch}"))

        wandb.log({**log_extra,
                   "epoch": epoch, "train/loss": tr_loss, "train/presence_mean": tr_pres,
                   "train/presence_sep": tr_sep, "train/flux_ratio_bright": tr_flux_ratio,
                   "val/flux_ratio_bright": va_flux_ratio,
                   "val/mse_sr2_hr": va_mse_sr2 / va_n, "val/mse_gain": mse_gain,
                   "val/z_outlier_rate": z_outlier, "val/goal_score": goal,
                   "lam_sparse_eff": lam_sparse_eff, "lam_flux_eff": lam_flux_eff,
                   "train/skipped_nonfinite_steps": n_skipped}, step=epoch)
        print(f"E{epoch:3d} loss={tr_loss:.4f} pres={tr_pres:.4g} sep={tr_sep:.2f} "
              f"fratio={tr_flux_ratio:.3f} vfratio={va_flux_ratio:.3f} "
              f"z_out={z_outlier:.1%} mse_gain={mse_gain:.3e} goal={goal:.4e}"
              + (f" skipped={n_skipped}" if n_skipped else ""), flush=True)

        collapsed = tr_pres < float(cfg["presence_floor"])
        if collapsed and epoch > sparse_warm:
            raise RuntimeError(
                f"Presence gate collapsed at epoch {epoch}: presence_mean={tr_pres:.3e} "
                f"< {float(cfg['presence_floor']):.1e}. The line branch is emitting "
                "nothing; aborting instead of training a model that cannot add line flux."
            )

        # The gate can also fail *without* collapsing in magnitude, by settling
        # at the same value for real and absent lines. That is what happened
        # before presence was supervised, and `presence_floor` cannot see it.
        sep_floor = float(cfg["presence_sep_floor"])
        if (sep_floor > 0 and float(cfg["lam_presence"]) > 0 and epoch > sparse_warm
                and np.isfinite(tr_sep) and tr_sep < sep_floor):
            raise RuntimeError(
                f"Presence gate carries no information at epoch {epoch}: mean presence on "
                f"real lines / on absent lines = {tr_sep:.2f} < {sep_floor:.2f}, while "
                f"presence_mean={tr_pres:.3e} looks healthy. The gate is acting as a "
                "constant multiplier on every line amplitude rather than selecting lines."
            )

        # Never checkpoint a collapsed model: `goal` is dominated by z_outlier,
        # which stays respectable with a dead line branch, so on its own it will
        # happily select one.
        if goal < best_goal - 1e-4 and not collapsed:
            best_goal = goal
            no_improve = 0
            # Keep this epoch's example for the end-of-run figure: SR2
            # early-stops, so the selected epoch need not be one that was
            # plotted, and manufacturing lines in the wrong place is invisible
            # in a loss curve and obvious in a spectrum.
            best_example = example
            torch.save({
                "sr2_state_dict": sr2.state_dict(), "config": dict(run_cfg),
                "sr1_config": sr1_cfg, "zhead_config": zhead_cfg,
                # The head SR2 was conditioned on travels with it. It is frozen,
                # so this is a copy rather than an adaptation -- but without it
                # the checkpoint cannot be evaluated without also locating the
                # exact upstream head, and pairing SR2 with the wrong one
                # silently changes every line position it emits.
                "zhead_state_dict": zhead.state_dict(),
                "z_mean": z_mean, "z_std": z_std, "use_sigma_channel": use_sigma,
                "epoch": epoch, "goal": best_goal,
            }, best_path)
            print(f"  ** saved {best_path} | goal={best_goal:.4e}", flush=True)
        else:
            no_improve += 1

        if epoch >= int(cfg["min_epochs"]) and no_improve >= int(cfg["patience"]):
            print(f"  early stop @ {epoch}", flush=True)
            break

    if best_example is not None:
        from ..wandb_plots import FINAL_SPECTRUM_KEY, spectrum_panel
        lo, sr_, hi, zt, zp = best_example
        wandb.log(spectrum_panel(
            wave_hi_um, lo, sr_, hi, z_true=zt, z_pred=zp, key=FINAL_SPECTRUM_KEY,
            title=f"SR2 final — best goal={best_goal:.4e}"))

    if int(cfg["zhead_unfreeze_last_n"]) == 0 and state_fingerprint(zhead) != zhead_fingerprint:
        raise RuntimeError(
            "SR2 training modified the redshift head despite zhead_unfreeze_last_n=0. "
            "The head is the estimator §4.4 reports on, and letting the refinement "
            "objective tune it was measured to take the held-out catastrophic-outlier "
            "rate from 10.84% to 17.83%. Do not work around this by relaxing the "
            "check: find what is writing to those tensors."
        )

    manifest = write_run_manifest(
        out, run_cfg, stage="sr2", dataset=dataset_path, split_path=split_path,
        upstream={"sr1_ckpt": sr1_ckpt, "sr1_config": sr1_config, "zhead_ckpt": zhead_ckpt},
    )
    log_checkpoint_artifact(
        [best_path, *manifest], kind="sr2",
        metadata={**run_cfg, "best_goal": best_goal},
    )
    print(f"SR2 done. best goal={best_goal:.6f}  ->  {best_path}", flush=True)
    return out
