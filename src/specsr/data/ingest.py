"""Discovery and reading of raw JADES NIRSpec ``x1d`` products.

The raw tree is laid out by field, disperser and tier::

    DR4/<field>/spectra/<disperser>/<tier>/hlsp_jades_jwst_nirspec_<tier>-<id>_<disperser>_v1.0_x1d.fits

Two facts about this layout drive the design.

**A target can appear under more than one tier.** In DR4, 33 of 5,157 targets
are observed in two tiers (for example ``goods-n-mediumhst`` and
``goods-n-mediumjwst``). Keying a galaxy on ``(field, tier, target_id)`` would
turn each of those into two independent "galaxies", which then land on opposite
sides of a train/test split — reintroducing exactly the duplicate-object leak
that the DR3 sample suffered from. The identity of a galaxy here is therefore
``(field, target_id)``, and multiple tiers are alternative observations of the
same object.

**Not every target has every disperser.** DR4 has 5,106 prism, 4,728 G140M,
4,539 G235M and 4,728 G395M exposures across GOODS-N and GOODS-S. Pairing must
be explicit about what it requires rather than assuming a complete grid.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "SpectrumFile",
    "DISPERSERS",
    "GRATINGS",
    "PRISM",
    "discover_spectra",
    "read_spectrum",
    "group_by_target",
]

#: Directory names of the dispersers we use.
PRISM = "clear-prism"
GRATINGS = ("f070lp-g140m", "f170lp-g235m", "f290lp-g395m")
DISPERSERS = (PRISM, *GRATINGS)

# hlsp_jades_jwst_nirspec_<tier>-<id>_<disperser>_v<ver>_x1d.fits
_NAME = re.compile(
    r"nirspec_(?P<tier>[a-z0-9-]+?)-(?P<target_id>\d+)_(?P<disperser>[a-z0-9-]+)_v[\d.]+_x1d\.fits$"
)


@dataclass(frozen=True)
class SpectrumFile:
    """One ``x1d`` product on disk, identified without opening it."""

    path: Path
    field: str
    tier: str
    target_id: str
    disperser: str

    @property
    def key(self) -> tuple[str, str]:
        """Galaxy identity: ``(field, target_id)``.

        Deliberately excludes the tier — see the module docstring.
        """
        return (self.field, self.target_id)

    @property
    def is_prism(self) -> bool:
        return self.disperser == PRISM


def discover_spectra(
    release_dir: Path | str,
    fields: tuple[str, ...] = ("goods-n", "goods-s"),
    dispersers: tuple[str, ...] = DISPERSERS,
) -> list[SpectrumFile]:
    """Enumerate ``x1d`` products under a release directory.

    Filenames are parsed rather than headers opened, so this is fast enough to
    run over the whole tree (~20k files) before deciding what to read.
    """
    release_dir = Path(release_dir)
    out: list[SpectrumFile] = []
    for field in fields:
        for disperser in dispersers:
            base = release_dir / field / "spectra" / disperser
            if not base.is_dir():
                continue
            for path in base.glob("*/*x1d*.fits"):
                m = _NAME.search(path.name)
                if not m:
                    continue
                out.append(
                    SpectrumFile(
                        path=path,
                        field=field,
                        tier=m["tier"],
                        target_id=m["target_id"],
                        disperser=m["disperser"],
                    )
                )
    return out


def read_spectrum(
    path: Path | str,
    extension: str = "EXTRACT3PIX1D",
) -> dict:
    """Read one ``x1d`` product.

    Parameters
    ----------
    path
        FITS file to read.
    extension
        Which extraction to use. ``EXTRACT3PIX1D`` is the 3-pixel extraction;
        ``EXTRACT5PIX1D`` is the wider aperture. The choice must be the same for
        the low- and high-resolution members of a pair, or their flux scales
        differ systematically and the model learns the aperture difference as if
        it were a resolution effect.

    Returns
    -------
    dict with ``wavelength`` (µm), ``flux``, ``flux_err`` (erg s⁻¹ cm⁻² Å⁻¹),
    plus ``ra``, ``dec``, ``target``, ``grating``, ``filter``, ``tier``.

    Non-finite samples are preserved rather than filled: downstream resampling
    ignores them, and silently interpolating over a detector gap would invent
    flux that was never observed.
    """
    from astropy.io import fits

    path = Path(path)
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        if extension not in hdul:
            available = [h.name for h in hdul if h.name]
            raise KeyError(f"{path.name} has no {extension!r}; available: {available}")
        data = hdul[extension].data
        wave = np.asarray(data["WAVELENGTH"], dtype=float)
        flux = np.asarray(data["FLUX"], dtype=float)
        err = np.asarray(data["FLUX_ERR"], dtype=float)

        meta = {
            "ra": float(hdr["RA"]),
            "dec": float(hdr["DEC"]),
            "target": str(hdr["TARGET"]).strip(),
            "grating": str(hdr.get("GRATING", "")).strip(),
            "filter": str(hdr.get("FILTER", "")).strip(),
            "tier": str(hdr.get("HLSPTIER", "")).strip(),
            "bunit": str(hdr.get("BUNIT", "")).strip(),
        }

    order = np.argsort(wave)
    return {
        "wavelength": wave[order],
        "flux": flux[order],
        "flux_err": err[order],
        **meta,
    }


def group_by_target(
    files: list[SpectrumFile],
    require_prism: bool = True,
    require_gratings: int = 1,
) -> dict[tuple[str, str], dict[str, list[SpectrumFile]]]:
    """Group files into candidate pairs, keyed by ``(field, target_id)``.

    Parameters
    ----------
    require_prism
        Drop targets with no prism exposure; the prism is the model's input.
    require_gratings
        Minimum number of distinct medium gratings required. The reference is
        stitched from up to three, and a target contributing only one covers a
        small fraction of the 1–5 µm range.

    Returns
    -------
    ``{(field, target_id): {disperser: [SpectrumFile, ...]}}``. A disperser maps
    to a *list* because the same target may have been observed in more than one
    tier; keeping all of them lets the pairing step choose (or combine) rather
    than having the choice made silently here.
    """
    grouped: dict[tuple[str, str], dict[str, list[SpectrumFile]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for f in files:
        grouped[f.key][f.disperser].append(f)

    out = {}
    for key, by_disp in grouped.items():
        if require_prism and PRISM not in by_disp:
            continue
        n_gratings = sum(1 for g in GRATINGS if g in by_disp)
        if n_gratings < require_gratings:
            continue
        out[key] = {k: sorted(v, key=lambda s: s.path.name) for k, v in by_disp.items()}
    return out
