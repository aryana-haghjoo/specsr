# Tutorials

Three executable notebooks live in
[`tutorials_for_user/`](https://github.com/aryana-haghjoo/specsr/tree/main/tutorials_for_user).
They run with no survey data and no configuration: real JADES spectra are
bundled alongside them, and weights download from the Hub on first use.

```bash
pip install "specsr[hub]"
jupyter lab tutorials_for_user/01_quickstart.ipynb
```

## What each one covers

**1 — Quickstart.** Load the pretrained chain, run it on one real galaxy, and
read the result. Covers the three stages (SR1 → ZHead → SR2), what each field of
{class}`~specsr.inference.SpecSRResult` means, and how to plot a prediction
against a true grating spectrum without misleading yourself about scale.

**2 — Your own spectrum.** The case you will actually have: a spectrum on its
own wavelength grid. Covers passing `wavelength=`, why the model's grid is
logarithmic, and a measurement of what grid coarseness does to integrated line
flux — the dominant effect, larger than the choice of interpolant.

**3 — Batches and trust.** Batch inference, then the harder question: how far to
believe the output. Uncertainty calibration, the shape of redshift failures, and
line-flux recovery as a function of line brightness. Read this one before
putting a number from this model into a paper.

## Bundled data

| file | contents |
|---|---|
| `sample_one_spectrum.npz` | a single galaxy (z = 3.00) |
| `sample_24_spectra.npz` | 24 galaxies spanning z = 0.31–9.43 |

Both are drawn from the **validation** split of the paired DR4 dataset — real
spectra the model never trained on, sampled across the redshift range rather
than cherry-picked. Each carries `wavelength`, `flux_low` (the prism input),
`flux_low_err`, `flux_high` (the grating truth, for comparison only), `z`,
`target_id` and `field`.

The 24-galaxy sample reaches z ~ 9.4 deliberately, past where the training data
is dense. That is where most of the model's failures are.

## Two caveats the notebooks make concrete

```{admonition} Absolute flux scale is not predicted
:class: warning

Only spectral shape is. Work in normalized units; see the note on the
{doc}`front page <../index>`.
```

```{admonition} These weights are not the product of an exhaustive search
:class: note

Hyperparameters carry over from an early sweep, and the released checkpoints are
selected on a composite criterion that does not track redshift accuracy directly.
The numbers the notebooks print are the model's current behaviour, and are
expected to improve.
```
