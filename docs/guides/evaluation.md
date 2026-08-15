# Evaluation

All evaluations run on the held-out split produced by
{func}`specsr.data.splits.get_training_split`, which is shared across every
training stage and analysis script.

The split is **80/20 over parent galaxies**, and the evaluation partition holds
**original spectra only** — 572 real galaxies. The built product augments every
galaxy at 21 rows each, so an unfiltered partition would be ~95% synthetic rows
and would compute statistics over 21x correlated samples, understating
uncertainties. See [`ARCHITECTURE.md`](https://github.com/aryana-haghjoo/specsr/blob/main/ARCHITECTURE.md)
for why there is no separate sealed test partition in this configuration, and how
to restore one.

```bash
specsr evaluate line-flux --outdir figures/
specsr evaluate line-snr  --outdir figures/
specsr evaluate residuals --outdir figures/
specsr evaluate redshift  --outdir figures/
specsr evaluate sample    --outdir figures/
```

## Analyses

**`line-flux`** — Fits the same Gaussian profile model to the super-resolved and
high-resolution reference spectra for [O II] λ3727, Hβ, [O III] λ5007 and Hα,
and compares the resulting integrated fluxes one-to-one. This is the test of
whether the model recovers line *fluxes*, which is what downstream diagnostics
(star-formation rates, ionisation parameters, metallicities) actually consume.

**`line-snr`** — Compares emission-line signal-to-noise between the
low-resolution input and the super-resolved output, and reports the fraction of
spectra improved per line, stratified by input S/N.

**`residuals`** — Two-dimensional residual maps over the evaluation set sorted by
redshift, for LR−HR, LR−SR and SR−HR.

**`redshift`** — Redshift recovery from LR, SR and HR inputs. Because a learned
estimator could in principle favour its own training distribution, this analysis
also supports method-independent estimators so the comparison does not rest on a
single approach.

**`sample`** — Characterises the spectroscopic sample against the parent
photometric catalogue (redshift–magnitude, stellar mass versus star-formation
rate), making the selection function and its biases explicit.

```{note}
Line fluxes and signal-to-noise are distinct claims. A higher S/N measured
without reference to ground truth does not by itself establish that a line is
correctly modelled; the `line-flux` analysis is what does that.
```
