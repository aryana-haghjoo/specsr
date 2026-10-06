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

**`line-flux`** — Fits Gaussian profiles to the super-resolved output, to the
low-resolution input it was given, and to the high-resolution reference, for
[O II] λ3727, Hβ, [O III] λ5007 and Hα, and compares the integrated fluxes
(`sqrt(2π) · A · σ`, in erg s⁻¹ cm⁻²) one-to-one against the reference. This is
the test of whether the model recovers line *fluxes*, which is what downstream
diagnostics (star-formation rates, ionisation parameters, metallicities)
actually consume.

Each spectrum is measured through its own line profile, because the three differ
by a factor of ten in resolution: a fixed velocity aperture narrow enough to
isolate a line in the reference holds about a fifth of the same line at prism
resolution, and would report unresolved flux as missing flux. [O II] and Hα use
a single Gaussian (`specsr.linefit.measure_line_fluxes`). Hβ, [O III] λ4959 and
[O III] λ5007 lie within 9000 km/s of one another, and the prism (median line
FWHM 4800 km/s there) leaves λ4959 blended with λ5007 in 84% of the held-out
galaxies and Hβ blended with the doublet in 45%, mostly below z ≈ 3. The three
are therefore fit jointly as Gaussians of a shared width
(`specsr.linefit.fit_hbeta_oiii`), so the flux of λ4959 is not assigned to a
neighbour; a single Gaussian reads [O III] λ5007 about 30% high and can widen
over the whole complex when centred on Hβ. A line
enters the comparison when the reference fit detects it in emission at
S/N ≥ 5.

On the held-out set the low-resolution input already recovers the total flux of
strong lines (median ratio to the reference 0.91–1.09 for Hβ, [O III] and Hα,
with 2–11% of lines off by more than a factor of two). The super-resolved
fluxes are lower and more scattered (median 0.72–0.87, 24–30% off by more than
a factor of two). Use the low-resolution spectrum, not the reconstruction, when
what you need is the total flux of a line.

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
