"""Stage 1: train the super-resolution backbone.

LR spectrum in, HR mean and predictive log-variance out. The objective is
:func:`specsr.training.losses.sr1_deblend_loss` -- a heteroscedastic likelihood
plus gated derivative terms that reward sharpness specifically where the
reference has line structure.

Ported from ``train/sr1_best/train_sr1.py``. The loop is unchanged; what changed
is that the model, dataset, split and loss now come from the package instead of
being redefined here, and the configuration is read from YAML rather than
hardcoded in the function body. The two losses were checked bit-identical on the
validation split before the swap.
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
from ..models.loaders import get_activation
from ..models.sr1 import SuperRes1D
from ..runtime import (
    apply_env_overrides,
    describe_overrides,
    limit_batches,
    resolve_dataset,
    wandb_mode,
)
from .common import (
    EMA,
    build_param_groups,
    clip_grad_or_skip,
    log_checkpoint_artifact,
    resolve_out_dir,
    set_seed,
    write_run_manifest,
)
from .losses import sr1_deblend_loss

__all__ = ["train_sr1"]

DEFAULTS: dict[str, Any] = {
    "activation": "gelu",
    "batch_size": 32,
    "dropout": 0.023538492919758583,
    "epochs": 200,
    "grad_clip": 0.5,
    "hidden_dim": 120,
    "lr": 8.23706977169561e-05,
    "num_res_blocks": 16,
    "weight_decay": 4.953557427559904e-05,
    # Two-way split: 80% of galaxies train, 20% evaluate. Evaluation is
    # restricted to original spectra -- the product augments every galaxy,
    # so an unfiltered 20% would be 95% synthetic rows and would report
    # performance on perturbations over 21x correlated samples.
    "train_frac": 0.8,
    "val_frac": 0.2,
    "seed": 42,
    "target_key_raw": "flux_high",
    # Clamping the predictive variance stops the likelihood running away early,
    # when the model can lower the loss faster by claiming huge uncertainty than
    # by fitting anything.
    "use_var_clamp": True,
    "var_clamp_min": 0.1,
    "var_clamp_max": 30.0,
    "logvar_reg": 1.5590946894260903e-06,
    "mask_smooth_k": 161,
    "mask_thresh_mad": 2.9458145924384755,
    "mask_dilate": 11,
    "mask_min_width": 7,
    "lam_d1": 0.11,
    "lam_d2": 0.01,
    "gate_min_frac": 0.01282614837763313,
    "gate_temp": 0.2962018702777486,
    "score_w_line": 0.50334848260373,
    "score_w_recon": 0.2,
    # Sharpness terms are warmed *down*: they buy resolution early, then start
    # manufacturing structure once the reconstruction is already close.
    "sharp_wd_start_epoch": 25,
    "sharp_wd_rate": 0.008,
    "sharp_wd_floor": 0.15,
    "ema_alpha": 0.9,
    "log_hist_every": 5,
    # Every run logs validation pictures, not only scalars.
    "plot_every": 5,
    "n_val_examples_to_log": 1,
    "wandb_project": "spectral-superresolution",
}


def _clamp_logvar(log_var: torch.Tensor, cfg: dict) -> torch.Tensor:
    if not bool(cfg["use_var_clamp"]):
        return log_var
    return torch.clamp(
        log_var,
        min=float(np.log(float(cfg["var_clamp_min"]))),
        max=float(np.log(float(cfg["var_clamp_max"]))),
    )


def _loss_kwargs(cfg: dict, sharp_scale: float) -> dict:
    return dict(
        logvar_reg=float(cfg["logvar_reg"]),
        mask_smooth_k=int(cfg["mask_smooth_k"]),
        mask_thresh_mad=float(cfg["mask_thresh_mad"]),
        mask_dilate=int(cfg["mask_dilate"]),
        mask_min_width=int(cfg["mask_min_width"]),
        lam_d1=float(cfg["lam_d1"]) * sharp_scale,
        lam_d2=float(cfg["lam_d2"]) * sharp_scale,
        gate_min_frac=float(cfg["gate_min_frac"]),
        gate_temp=float(cfg["gate_temp"]),
        score_w_recon=float(cfg["score_w_recon"]),
        score_w_line=float(cfg["score_w_line"]),
    )


def train_sr1(
    config: dict | None = None,
    *,
    dataset: str | Path | None = None,
    out_dir: str | Path | None = None,
    init_ckpt: str | Path | None = None,
) -> Path:
    """Train SR1 and return the directory holding its checkpoints.

    ``best_superres_model.pth`` is selected on an EMA-smoothed validation loss;
    the raw value is noisy enough that picking on it selects lucky epochs. There
    is deliberately no early stopping -- the schedule runs to ``epochs``.

    ``init_ckpt`` warm-starts the model from an earlier run's state dict for
    fine-tuning; the config must describe the same architecture, and loading is
    strict so a mismatch fails here rather than training a franken-model.
    """
    import wandb

    cfg = dict(DEFAULTS)
    cfg.update(config or {})
    cfg = apply_env_overrides(cfg)
    print(f"[specsr] {describe_overrides()}", flush=True)

    set_seed(int(cfg["seed"]))
    out = resolve_out_dir(out_dir, "sr1")
    dataset_path = resolve_dataset(str(dataset or cfg.get("dataset_npz", "")))
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # `val_idx` drives checkpoint selection and the LR schedule. The test split
    # is withheld by `get_training_split` and never touched here.
    train_idx, val_idx, split_path = get_training_split(
        dataset_path, float(cfg["train_frac"]), float(cfg["val_frac"]), int(cfg["seed"])
    )

    full = FixedGridSpectraDataset(
        dataset_path, normalize_flux=True, target_key_raw=cfg["target_key_raw"]
    )
    with np.load(dataset_path, allow_pickle=True) as _d:
        wave_um = np.asarray(_d["wavelength_high"], dtype=np.float32)
    bs = int(cfg["batch_size"])
    train_loader = DataLoader(Subset(full, train_idx), batch_size=bs, shuffle=True)
    val_loader = DataLoader(Subset(full, val_idx), batch_size=bs, shuffle=False)

    model = SuperRes1D(
        in_channels=1,
        hidden_dim=int(cfg["hidden_dim"]),
        num_res_blocks=int(cfg["num_res_blocks"]),
        dropout=float(cfg["dropout"]),
        activation_fn=get_activation(cfg["activation"]),
    ).to(device)

    if init_ckpt is not None:
        model.load_state_dict(torch.load(init_ckpt, map_location="cpu"))
        print(f"[specsr] warm-started SR1 from {init_ckpt}", flush=True)

    optimizer = torch.optim.AdamW(
        build_param_groups(model, lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    wandb.init(
        mode=wandb_mode(),
        project=cfg["wandb_project"],
        name=cfg.get("wandb_name") or f"sr1_{int(time.time())}",
        tags=["sr1", "superres", "1D", "uncertainty", "deblend"],
        config={**cfg, "dataset_npz": dataset_path, "out_dir": str(out),
                "init_ckpt": str(init_ckpt) if init_ckpt else None},
    )
    wandb.watch(model, log="all", log_freq=10)

    ema = EMA(float(cfg["ema_alpha"]))
    best = float("inf")
    best_example = None
    best_path = out / "best_superres_model.pth"

    for epoch in range(int(cfg["epochs"])):
        # Warm the sharpness terms down after `sharp_wd_start_epoch`.
        if epoch < int(cfg["sharp_wd_start_epoch"]):
            sharp_scale = 1.0
        else:
            decayed = 1.0 - float(cfg["sharp_wd_rate"]) * (epoch - int(cfg["sharp_wd_start_epoch"]))
            sharp_scale = max(float(cfg["sharp_wd_floor"]), decayed)
        lkw = _loss_kwargs(cfg, sharp_scale)

        model.train()
        tr_loss = tr_n = n_skipped = 0
        for batch in limit_batches(train_loader, kind="train"):
            x_low, x_high, x_high_err = (t.unsqueeze(1).to(device) for t in batch[:3])
            mean, log_var = model(x_low)
            log_var = _clamp_logvar(log_var, cfg)
            loss, _ = sr1_deblend_loss(
                mean=mean, log_var=log_var, x_high=x_high, x_high_err=x_high_err, **lkw
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(cfg["grad_clip"]) > 0:
                _, ok = clip_grad_or_skip(model.parameters(), float(cfg["grad_clip"]))
                if not ok:
                    n_skipped += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
            optimizer.step()
            tr_loss += float(loss.item())
            tr_n += 1

        model.eval()
        va_loss = va_mse = va_n = 0
        example = None
        with torch.no_grad():
            for batch in limit_batches(val_loader, kind="val"):
                x_low, x_high, x_high_err = (t.unsqueeze(1).to(device) for t in batch[:3])
                mean, log_var = model(x_low)
                log_var = _clamp_logvar(log_var, cfg)
                loss, _ = sr1_deblend_loss(
                    mean=mean, log_var=log_var, x_high=x_high, x_high_err=x_high_err, **lkw
                )
                va_loss += float(loss.item())
                va_mse += float(((mean - x_high) ** 2).mean().item())
                va_n += 1
                if example is None:
                    example = (x_low[0, 0], mean[0, 0], x_high[0, 0], batch[3][0])

        val_loss = va_loss / max(1, va_n)
        smooth = ema.update(val_loss)
        scheduler.step(smooth)

        # A picture every epoch, not only scalars: a reconstruction that starts
        # inventing lines shows up here long before any loss curve says so.
        log_extra: dict[str, Any] = {}
        if example is not None and (
            (epoch % int(cfg["plot_every"]) == 0) or (epoch + 1 == int(cfg["epochs"]))
        ):
            from ..wandb_plots import spectrum_panel
            lo, sr_, hi, zt = example
            log_extra.update(spectrum_panel(
                wave_um, lo, sr_, hi, z_true=zt, title=f"SR1 — epoch {epoch + 1}"))

        wandb.log({
            **log_extra,
            "epoch": epoch,
            "avg_train_loss": tr_loss / max(1, tr_n),
            "avg_val_loss_raw": val_loss,
            "avg_val_loss_smooth": smooth,
            "val_mse": va_mse / max(1, va_n),
            "sharp_scale": sharp_scale,
            "skipped_nonfinite_steps": n_skipped,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })

        if smooth < best:
            best = smooth
            # Keep this epoch's example so the end-of-run figure shows the model
            # that was actually selected. The per-epoch panels only cover epochs
            # divisible by `plot_every`, which the best epoch need not be.
            best_example = example
            torch.save(model.state_dict(), best_path)
            print(f"saved best SR1 @ epoch {epoch + 1}  val_smooth={best:.6f}", flush=True)

    # One evaluation figure per run, from the selected checkpoint, logged under
    # its own key so it is not buried in the per-epoch series.
    if best_example is not None:
        from ..wandb_plots import FINAL_SPECTRUM_KEY, spectrum_panel
        lo, sr_, hi, zt = best_example
        wandb.log(spectrum_panel(
            wave_um, lo, sr_, hi, z_true=zt, key=FINAL_SPECTRUM_KEY,
            title=f"SR1 final — best val_smooth={best:.6f}"))

    torch.save(model.state_dict(), out / "final_model.pth")
    manifest = write_run_manifest(
        out, cfg, stage="sr1", dataset=dataset_path, split_path=split_path,
        upstream={"init_ckpt": init_ckpt} if init_ckpt else None,
    )
    log_checkpoint_artifact(
        [best_path, out / "final_model.pth", *manifest], kind="sr1",
        metadata={**cfg, "best_val_loss_smooth": best},
    )
    print(f"SR1 done. best={best:.6f}  ->  {best_path}", flush=True)
    return out
