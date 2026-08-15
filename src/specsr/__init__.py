"""specsr — physics-informed deep learning for super-resolving galaxy spectra.

Enhances low-resolution galaxy spectra by ~10x in resolving power (R~100 to
R~1000), recovering narrow emission-line features — including blended doublets
such as [O III] 4959,5007 and Hbeta — that are unresolvable at prism resolution.

Trained on paired JWST/NIRSpec prism + medium-grating observations from JADES.
"""

from __future__ import annotations

# Read from the installed metadata rather than restated here. A second copy of
# the version drifts from pyproject.toml the first time one of them is bumped
# alone -- which had already happened -- and the copy that is wrong is the one
# users see.
try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _version

    __version__ = _version("specsr")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0.dev0"

__all__ = ["__version__", "get_checkpoint"]


def __getattr__(name: str):
    # Lazy re-export keeps `import specsr` cheap: torch and huggingface_hub are
    # only imported when a model or checkpoint is actually requested.
    if name == "get_checkpoint":
        from .checkpoints import get_checkpoint

        return get_checkpoint
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
