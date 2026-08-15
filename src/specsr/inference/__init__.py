"""Run a trained chain on spectra.

The public entry point most callers want is :class:`SpecSRPipeline`::

    from specsr.inference import SpecSRPipeline

    pipe = SpecSRPipeline.from_pretrained()          # weights from the Hub
    result = pipe(flux_low, wavelength_low)          # your own wavelength grid

``wavelength_low`` is optional, but pass it unless your spectrum is already on
``pipe.wavelength``: it makes the pipeline resample for you, integrating rather
than interpolating. See :meth:`SpecSRPipeline.__call__` for why that distinction
costs ~20% of every line peak if you get it wrong.

This name was advertised in the README and the docs for some time before it
existed, and the example here described a two-argument call for some time after
that, while ``__call__`` still took one. Both are now true.
"""

from __future__ import annotations

from .pipeline import SpecSRPipeline, run_infer

__all__ = ["SpecSRPipeline", "run_infer"]
