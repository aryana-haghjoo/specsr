# API reference

Leaf modules are listed explicitly rather than recursively: several are
re-exported from their package `__init__`, and a recursive sweep documents those
objects twice, which Sphinx reports as a duplicate description.

```{eval-rst}
.. autosummary::
   :toctree: generated

   specsr.inference.pipeline
   specsr.paths
   specsr.checkpoints
   specsr.config
   specsr.runtime
   specsr.metrics
   specsr.linefit
   specsr.data.grid
   specsr.data.augment
   specsr.data.build
   specsr.data.ingest
   specsr.data.stitch
   specsr.data.datasets
   specsr.data.splits
   specsr.models.blocks
   specsr.models.sr1
   specsr.models.zhead
   specsr.models.sr2
   specsr.models.lines
   specsr.training.losses
   specsr.training.ztransform
   specsr.training.zhead_sources
```
