"""Redshift normalisation and bounded decoding.

The redshift head does not regress ``z`` directly. Training normalises ``z`` to
zero mean and unit variance using statistics from the *training split only*, and
the head's raw output is squashed through a sigmoid onto the observed redshift
range before the loss is applied.

The sigmoid bound matters: an unbounded head can emit redshifts far outside the
range the data covers, and the heteroscedastic likelihood will happily trade a
wild prediction against a large predicted variance. Bounding the mean keeps the
uncertainty interpretable. It also keeps gradients alive at the extremes, which a
hard clamp would not.

Collecting this in one object matters because the same transform must be applied
identically at training, inference and evaluation time. When it lived inline in
four copies of the training script, a divergence would have shown up as a subtle
redshift bias rather than as an error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["RedshiftTransform"]


@dataclass(frozen=True)
class RedshiftTransform:
    """Affine normalisation plus a sigmoid bound on the decoded mean.

    Attributes
    ----------
    mean, std
        Normalisation statistics, computed on the **training split only** so no
        information about held-out redshifts leaks through the transform.
    z_min_n, z_max_n
        Normalised bounds of the observed redshift range, used to squash the
        head's raw output.
    """

    mean: float
    std: float
    z_min_n: float
    z_max_n: float

    @classmethod
    def from_training_redshifts(cls, z_train, pad: float = 0.0) -> RedshiftTransform:
        """Fit the transform to the training-split redshifts.

        ``pad`` optionally widens the bounds by a fraction of the range, giving
        the sigmoid a little headroom at the extremes.
        """
        z = np.asarray(z_train, dtype=np.float64)
        mean = float(np.mean(z))
        std = float(np.std(z))
        if std < 1e-8:
            std = 1.0
        lo, hi = float(np.min(z)), float(np.max(z))
        if pad:
            span = hi - lo
            lo -= pad * span
            hi += pad * span
        return cls(
            mean=mean,
            std=std,
            z_min_n=(lo - mean) / std,
            z_max_n=(hi - mean) / std,
        )

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        """Physical redshift -> normalised units."""
        return (z - self.mean) / self.std

    def denormalize(self, z_n: torch.Tensor) -> torch.Tensor:
        """Normalised units -> physical redshift."""
        return z_n * self.std + self.mean

    def decode_mean(self, mu_raw: torch.Tensor, bounded: bool = True) -> torch.Tensor:
        """Squash a raw head output onto the observed range, in normalised units.

        ``bounded=False`` passes the value through untouched, for heads whose
        estimate is already in normalised redshift units and already inside the
        range -- the classification head reads its estimate off a grid, so
        squashing it a second time would compress it towards the range centre.
        Pass ``zhead.bounded_mean`` rather than deciding per call site.
        """
        if not bounded:
            return mu_raw
        return self.z_min_n + (self.z_max_n - self.z_min_n) * torch.sigmoid(mu_raw)

    def decode_sigma(self, log_var: torch.Tensor, var_floor: float = 1e-6) -> torch.Tensor:
        """Predicted log-variance (normalised) -> physical 1-sigma uncertainty."""
        return torch.sqrt(torch.exp(log_var).clamp_min(var_floor)) * self.std

    def predict(
        self, mu_raw: torch.Tensor, log_var: torch.Tensor, var_floor: float = 1e-6,
        bounded: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw head outputs -> physical ``(z, sigma_z)``.

        ``log_var`` is clamped before exponentiation; without it an early,
        overconfident-in-the-wrong-direction update can send ``exp(log_var)`` to
        infinity and take the loss with it.
        """
        log_var = torch.clamp(log_var, min=-12.0, max=12.0)
        mu_n = self.decode_mean(mu_raw, bounded=bounded)
        return self.denormalize(mu_n), self.decode_sigma(log_var, var_floor)

    def as_dict(self) -> dict:
        return {
            "z_mean": self.mean,
            "z_std": self.std,
            "z_min_n": self.z_min_n,
            "z_max_n": self.z_max_n,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RedshiftTransform:
        return cls(
            mean=float(d["z_mean"]),
            std=float(d["z_std"]),
            z_min_n=float(d["z_min_n"]),
            z_max_n=float(d["z_max_n"]),
        )
