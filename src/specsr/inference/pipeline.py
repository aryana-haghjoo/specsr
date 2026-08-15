"""The user-facing inference pipeline: LR spectrum in, SR spectrum and redshift out.

This is the object the README promises. It wraps the three stages so a caller
never has to know the channel order of the SR2 input stack, that SR1 emits
``log_var`` while the head consumes ``log_sigma``, or that the redshift decoding
has to use the transform the head was trained with. Getting any of those subtly
wrong produces a plausible-looking spectrum rather than an error.

Everything returned is in the **physical units of the input spectrum**, not the
per-spectrum normalised units the models work in. The normalisation is applied on
the way in and undone on the way out, using the input's own moments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..models.lines import LINE_LIST_REST_AA
from ..models.loaders import load_sr1, load_sr2, load_zhead
from ..models.sr2 import build_sr2_input, constrain_delta
from ..training.ztransform import RedshiftTransform

__all__ = ["SpecSRPipeline", "SpecSRResult", "resolve_zhead_in", "run_infer"]


def resolve_zhead_in(directory: Path) -> Path | None:
    """Find the redshift head in a checkpoint directory, under either name.

    Two naming conventions coexist, and both are load-bearing.
    ``best_zhead.pth`` is what the Hub layout and the assembled bundles use.
    ``best_zhead_<source>.pth`` is what ``specsr train zhead`` writes, because
    the four comparison arms of the redshift experiment are distinguished by
    that suffix and three of them are *published* under it
    (``zhead/best_zhead_lowres.pth`` and friends in :mod:`specsr.checkpoints`).

    Renaming either would break the other, so lookup accepts both --
    :mod:`specsr.checkpoints` already did this via ``_LOCAL_ALIASES`` and this
    is the same policy for directory-based loading.

    Raises rather than guessing when a directory holds several arms: choosing
    one by sort order would silently pair SR2 with the wrong redshift head, and
    every line position it emits depends on that choice.
    """
    canonical = directory / "best_zhead.pth"
    if canonical.exists():
        return canonical

    candidates = sorted(directory.glob("best_zhead_*.pth"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"{directory} holds {len(candidates)} redshift heads "
            f"({', '.join(p.name for p in candidates)}). Pass zhead_ckpt explicitly: "
            "picking one by sort order would silently select a comparison arm."
        )
    return None


def _redshift_transform(zhead_ckpt, z_mean, z_std, dataset) -> RedshiftTransform:
    """Rebuild the transform the head was trained under.

    The head's raw output is squashed onto ``[z_min_n, z_max_n]``, so these
    bounds are not cosmetic -- they set the range of representable redshifts.
    Heads trained by :func:`specsr.training.zhead.train_zhead` store them.
    Older checkpoints do not, and for those the bounds have to be recomputed
    from the *training* split of the dataset the head was trained on.

    There is deliberately no default. Guessing plausible bounds does not fail,
    it just returns wrong redshifts: with ``[-3, 3]`` against the true
    ``[-1.78, 6.25]``, objects at z~3.9 came back at z~0.7 -- an answer that
    looks like a real measurement and is off by a factor of five.
    """
    ck = torch.load(str(zhead_ckpt), map_location="cpu", weights_only=False)
    if "z_min_n" in ck and "z_max_n" in ck:
        return RedshiftTransform(
            mean=z_mean, std=z_std,
            z_min_n=float(ck["z_min_n"]), z_max_n=float(ck["z_max_n"]),
        )

    if dataset is None:
        raise ValueError(
            f"{Path(zhead_ckpt).name} predates stored redshift bounds, so they must be "
            "recomputed from the training split — pass dataset=... pointing at the "
            "dataset this head was trained on. Without them the decoded redshifts are "
            "silently wrong rather than absent."
        )

    from ..data.datasets import FixedGridSpectraDataset
    from ..evaluation import load_split

    ds = FixedGridSpectraDataset(str(dataset), normalize_flux=True)
    z_train = np.asarray(ds.z)[load_split("train", dataset)]
    return RedshiftTransform(
        mean=z_mean, std=z_std,
        z_min_n=float((z_train.min() - z_mean) / z_std),
        z_max_n=float((z_train.max() - z_mean) / z_std),
    )


@dataclass
class SpecSRResult:
    """Output of one pipeline call, in the input's physical flux units."""

    sr1: np.ndarray
    """SR1 reconstruction, shape ``(B, L_hi)``."""
    sr1_sigma: np.ndarray
    """SR1 predictive 1-sigma."""
    sr2: np.ndarray | None
    """SR2 refinement, or ``None`` if no SR2 checkpoint was loaded."""
    sr2_sigma: np.ndarray | None
    z: np.ndarray
    """Predicted redshift, shape ``(B,)``."""
    z_sigma: np.ndarray
    wavelength: np.ndarray
    """HR wavelength grid in microns, shape ``(L_hi,)``."""


class SpecSRPipeline:
    """Frozen SR1 -> redshift head -> SR2, ready for inference.

    Construct with :meth:`from_checkpoints`. Calling the pipeline on a batch of
    low-resolution spectra returns a :class:`SpecSRResult`.
    """

    def __init__(self, sr1, zhead, sr2, cfg, ztransform, wave, device):
        self.sr1, self.zhead, self.sr2 = sr1, zhead, sr2
        self.cfg = cfg
        self.ztransform = ztransform
        self.wave = np.asarray(wave, dtype=np.float32)
        self.device = device
        self.line_rest_um = np.asarray(
            [w for _, w in LINE_LIST_REST_AA], dtype=np.float32) * 1e-4

    @property
    def wavelength(self) -> np.ndarray:
        """The model's output grid in microns, shape ``(L_hi,)``.

        Alias of :attr:`wave`, matching :attr:`SpecSRResult.wavelength` so the
        same name means the same thing on both objects.
        """
        return self.wave

    @classmethod
    def from_checkpoints(
        cls,
        directory: str | Path | None = None,
        *,
        sr1_ckpt: str | Path | None = None,
        sr1_config: str | Path | None = None,
        zhead_ckpt: str | Path | None = None,
        sr2_ckpt: str | Path | None = None,
        wavelength: np.ndarray | None = None,
        dataset: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> SpecSRPipeline:
        """Load a chain, either from a directory of checkpoints or explicit paths.

        A ``directory`` is expected to hold ``best_superres_model.pth``,
        ``config_logR.yaml``, a redshift head and optionally ``best_sr2.pth``
        -- the layout the archived run directories use.

        The head is accepted under either name: ``best_zhead.pth`` (the Hub and
        bundle layout) or ``best_zhead_<source>.pth`` (what ``specsr train
        zhead`` actually writes). Only the first was recognised until
        2026-08-14, so a directory produced directly by the trainer could not be
        loaded here at all -- which is the normal case when evaluating a run you
        have just finished.

        The HR wavelength grid must come from somewhere: pass ``wavelength``
        directly, or ``dataset`` to read it from a built ``.npz``.
        """
        d = Path(directory) if directory else None
        sr1_ckpt = sr1_ckpt or (d / "best_superres_model.pth" if d else None)
        sr1_config = sr1_config or (d / "config_logR.yaml" if d else None)
        zhead_ckpt = zhead_ckpt or (resolve_zhead_in(d) if d else None)
        if sr2_ckpt is None and d is not None and (d / "best_sr2.pth").exists():
            sr2_ckpt = d / "best_sr2.pth"
        if not (sr1_ckpt and sr1_config and zhead_ckpt):
            raise ValueError(
                "need sr1_ckpt, sr1_config and zhead_ckpt (or a directory containing them)"
            )

        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        sr1, _ = load_sr1(sr1_config, sr1_ckpt, device)
        zhead, z_mean, z_std, use_sigma, _ = load_zhead(zhead_ckpt, device)

        if wavelength is None:
            if dataset is not None:
                with np.load(str(dataset), allow_pickle=True) as npz:
                    wavelength = np.asarray(npz["wavelength_high"], dtype=np.float32)
            else:
                # Every published dataset is built on DEFAULT_GRID, and the models
                # are convolutional over it, so requiring a 1 GB .npz just to
                # recover 6,671 known numbers was pure friction. Verified equal to
                # data/paired_DR4_logR.npz's `wavelength_high` to float precision.
                from ..data.grid import DEFAULT_GRID

                wavelength = DEFAULT_GRID.centers().astype(np.float32)

        line_rest = np.asarray([w for _, w in LINE_LIST_REST_AA], dtype=np.float32) * 1e-4
        sr2, cfg = (None, {})
        if sr2_ckpt is not None:
            sr2, cfg = load_sr2(sr2_ckpt, wavelength, line_rest, device)

        ztf = _redshift_transform(zhead_ckpt, z_mean, z_std, dataset)
        cfg = {**cfg, "use_sigma_channel": use_sigma}
        return cls(sr1, zhead, sr2, cfg, ztf, wavelength, device)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str | None = None,
        revision: str | None = None,
        *,
        dataset: str | Path | None = None,
        wavelength: np.ndarray | None = None,
        device: str | torch.device | None = None,
    ) -> SpecSRPipeline:
        """Load the published chain, fetching weights from the Hugging Face Hub.

        Set ``SPECSR_CHECKPOINT_DIR`` to load from a local directory instead --
        no network and no Hub account. See :mod:`specsr.checkpoints`.

        The default revision deliberately excludes ``v1-submission``: those
        weights were trained on a leaky split and reproduce numbers that are not
        real.
        """
        from ..checkpoints import get_checkpoint

        return cls.from_checkpoints(
            sr1_ckpt=get_checkpoint("sr1", repo_id, revision),
            # Fetched by name, not looked for beside the weights: a Hub snapshot
            # only contains the files that were actually requested.
            sr1_config=get_checkpoint("sr1_config", repo_id, revision),
            zhead_ckpt=get_checkpoint("zhead", repo_id, revision),
            sr2_ckpt=get_checkpoint("sr2", repo_id, revision),
            dataset=dataset,
            wavelength=wavelength,
            device=device,
        )

    @torch.no_grad()
    def __call__(
        self,
        flux_low: np.ndarray,
        wavelength: np.ndarray | None = None,
        *,
        batch_size: int = 32,
    ) -> SpecSRResult:
        """Run the chain on ``flux_low``, shaped ``(L,)`` or ``(B, L)``.

        The input is standardised per spectrum before the models see it, and the
        output is mapped back using **the input's** mean and standard deviation.

        .. warning::

           That is not the same as the true high-resolution spectrum's scale, and
           the difference is large -- a median factor of ~4 on JADES validation
           data, ranging past 40 on individual objects. The models are trained on
           each spectrum standardised independently, so the mapping they learn is
           *shape to shape*; the absolute flux scale of the high-resolution
           spectrum is simply not recoverable from a low-resolution input alone.

           Practically: the returned arrays are self-consistent and fine to plot,
           integrate and compare **within one spectrum**. Comparing them against a
           real grating spectrum, or across objects, requires standardising both
           sides first -- which is why every figure in the paper is labelled
           :math:`F_\\lambda` *(normalized)*. See
           :meth:`specsr.data.datasets.PairedSpectra.denormalize`, which needs the
           high-resolution moments this call does not have.

        Parameters
        ----------
        flux_low
            Observed low-resolution flux, ``(L,)`` or ``(B, L)``.
        wavelength
            The wavelength array ``flux_low`` is sampled on, in microns. Pass it
            whenever your spectrum is on its own grid — the input is then
            resampled onto the model's grid with
            :func:`~specsr.data.grid.resample_flux_conserving`. Omit it only if
            ``flux_low`` is already on :attr:`wavelength`.

            Resampling **integrates** rather than interpolating, because flux is
            an area under the curve, so integrated line flux is preserved by
            construction rather than by luck.

            The size of that effect is worth being accurate about: onto this
            model's R=4000 grid, integrating and interpolating agree to well
            under a percent, because the grid is fine compared with the lines.
            The two diverge as the target grid coarsens (~3% apart by R~700),
            and by then *both* are losing flux fast -- the dominant term is grid
            coarseness, not the method. This is why the pipeline resamples onto
            a grid that is fine enough for the question rather than leaving the
            choice to the caller: it is the grid, more than the interpolant,
            that destroys emission lines.
        """
        flux_low = np.asarray(flux_low, dtype=np.float64)
        if wavelength is not None:
            wavelength = np.asarray(wavelength, dtype=np.float64)
            if wavelength.shape[0] != flux_low.shape[-1]:
                raise ValueError(
                    f"wavelength has length {wavelength.shape[0]} but flux_low's last "
                    f"axis is {flux_low.shape[-1]}; they describe the same spectrum "
                    "and must match"
                )
            if not (
                wavelength.shape[0] == self.wave.shape[0]
                and np.allclose(wavelength, self.wave, rtol=1e-6)
            ):
                from ..data.grid import resample_flux_conserving

                single = flux_low.ndim == 1
                rows = np.atleast_2d(flux_low)
                out = np.empty((rows.shape[0], self.wave.shape[0]), dtype=np.float64)
                for i, row in enumerate(rows):
                    out[i] = resample_flux_conserving(wavelength, row, self.wave)
                flux_low = out[0] if single else out

        x = np.atleast_2d(np.asarray(flux_low, dtype=np.float32))
        mu = np.nanmean(x, axis=1, keepdims=True)
        sd = np.nanstd(x, axis=1, keepdims=True)
        sd = np.where(sd > 1e-30, sd, 1.0)
        xn = np.nan_to_num((x - mu) / sd)

        acc: dict[str, list] = {k: [] for k in
                                ("sr1", "sr1_sigma", "sr2", "sr2_sigma", "z", "z_sigma")}
        for start in range(0, xn.shape[0], batch_size):
            chunk = xn[start:start + batch_size]
            s = torch.tensor(sd[start:start + batch_size], device=self.device).view(-1, 1, 1)
            m = torch.tensor(mu[start:start + batch_size], device=self.device).view(-1, 1, 1)
            x_low = torch.tensor(chunk, device=self.device).unsqueeze(1)

            sr1_mean, sr1_logvar = self.sr1(x_low)
            sr1_log_sigma = 0.5 * sr1_logvar
            sr1_sigma = torch.exp(sr1_log_sigma).clamp_min(1e-6)

            z_in = torch.cat([sr1_mean, sr1_log_sigma], 1) \
                if self.cfg.get("use_sigma_channel", True) else sr1_mean
            mu_raw, logvar_z = self.zhead(z_in)
            zhat, z_sigma = self.ztransform.predict(
                mu_raw, logvar_z, bounded=self.zhead.bounded_mean)
            zhat = zhat.reshape(-1)

            acc["sr1"].append(((sr1_mean * s + m).squeeze(1)).cpu().numpy())
            acc["sr1_sigma"].append(((sr1_sigma * s).squeeze(1)).cpu().numpy())
            acc["z"].append(zhat.cpu().numpy())
            acc["z_sigma"].append(z_sigma.reshape(-1).cpu().numpy())

            if self.sr2 is not None:
                x_in = build_sr2_input(
                    x_low=x_low, sr1_mean=sr1_mean, sr1_log_sigma=sr1_log_sigma,
                    zhat=zhat,
                    wave_hi_um=torch.tensor(self.wave, device=self.device),
                    line_rest_um=self.line_rest_um,
                    use_sr1_sigma=bool(self.cfg.get("use_sr1_sigma", True)),
                    use_line_mask=bool(self.cfg.get("use_line_mask", True)),
                    use_zhat_channel=bool(self.cfg.get("use_zhat_channel", True)),
                    use_zsigma_channel=bool(self.cfg.get("use_zsigma_channel", False)),
                    sigma_base_um=float(self.cfg.get("sigma_base_um", 0.005)),
                    # Flags come from the SR2 checkpoint's own config, so
                    # inference reproduces the conditioning it was trained with.
                    z_sigma=(z_sigma.reshape(-1)
                             if (self.cfg.get("zsigma_line_mask", False)
                                 or self.cfg.get("use_zsigma_channel", False))
                             else None),
                    sigma_max_um=float(self.cfg.get("zsigma_mask_max_um", 0.05)),
                )
                delta_raw, sr2_logvar = self.sr2(x_in, zhat)[:2]
                sr2_mean = sr1_mean + constrain_delta(
                    delta_raw, float(self.cfg.get("delta_cap", 0.0)))
                acc["sr2"].append(((sr2_mean * s + m).squeeze(1)).cpu().numpy())
                acc["sr2_sigma"].append(
                    ((torch.exp(0.5 * sr2_logvar) * s).squeeze(1)).cpu().numpy())

        cat = {k: (np.concatenate(v) if v else None) for k, v in acc.items()}
        return SpecSRResult(
            sr1=cat["sr1"], sr1_sigma=cat["sr1_sigma"],
            sr2=cat["sr2"], sr2_sigma=cat["sr2_sigma"],
            z=cat["z"], z_sigma=cat["z_sigma"], wavelength=self.wave,
        )


def run_infer(args) -> int:
    """CLI entry point for ``specsr infer``."""
    pipe = SpecSRPipeline.from_checkpoints(
        getattr(args, "checkpoints", None), dataset=args.dataset)

    with np.load(str(args.dataset), allow_pickle=True) as d:
        flux_low = np.asarray(d["flux_low"], dtype=np.float32)

    if getattr(args, "idx", None):
        rows = np.asarray(args.idx, dtype=int)
    elif getattr(args, "range", None):
        rows = np.arange(int(args.range[0]), int(args.range[1]))
    else:
        rows = np.arange(flux_low.shape[0])

    res = pipe(flux_low[rows])
    print(f"ran {len(rows)} spectra   z range [{res.z.min():.3f}, {res.z.max():.3f}]")

    if getattr(args, "save", None):
        # A bare filename goes to the package's output directory; an explicit
        # path (absolute, or containing a separator) is honoured as given.
        save_path = Path(args.save)
        if not save_path.is_absolute() and save_path.parent == Path("."):
            from ..paths import output_dir

            save_path = output_dir("predictions") / save_path.name
        args.save = str(save_path)
        np.savez(
            args.save, rows=rows, wavelength=res.wavelength, sr1=res.sr1,
            sr1_sigma=res.sr1_sigma, z=res.z, z_sigma=res.z_sigma,
            **({"sr2": res.sr2, "sr2_sigma": res.sr2_sigma} if res.sr2 is not None else {}),
        )
        print(f"wrote {args.save}")
    return 0
