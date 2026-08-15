# Tutorials

Three executable notebooks. They run with no survey data and no configuration —
real JADES spectra are bundled here, and weights download from the Hub on first
use.

```bash
pip install "specsr[hub]"
jupyter lab 01_quickstart.ipynb
```

| notebook | covers |
|---|---|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | run the chain on one spectrum, read the output, and plot it fairly |
| [`02_your_own_spectrum.ipynb`](02_your_own_spectrum.ipynb) | arbitrary wavelength grids, resampling, and what grid coarseness does to emission lines |
| [`03_batches_and_trust.ipynb`](03_batches_and_trust.ipynb) | batches, uncertainty calibration, flux recovery vs line brightness, and the failure modes |

Read them in order. Notebook 1 is the five-minute version; 2 is the one you
actually need for your own data; 3 is the one to read before putting a number in
a paper.

## Bundled data

| file | contents |
|---|---|
| `sample_one_spectrum.npz` | a single galaxy (z = 3.00), for notebook 1 |
| `sample_24_spectra.npz` | 24 galaxies spanning z = 0.31–9.43, for notebook 3 |

Both are drawn from the **held-out** split of `data/paired_DR4_logR.npz` — real
spectra the model never trained on, sampled across the redshift range rather
than cherry-picked. Each carries `wavelength`, `flux_low` (prism input),
`flux_low_err`, `flux_high` (grating truth, for comparison only), `z`,
`target_id` and `field`.

```{note}
**Why there is no leak, and why that is worth stating.** Training augments each
galaxy into 21 near-identical rows differing only by a small redshift shift and
noise. Splitting such a product *by row* puts near-duplicates of a held-out
galaxy into training, and every held-out number then comes out optimistic — this
happened here, and it is why the dataset is rebuilt and the split is drawn over
**parent galaxies** instead. Augmentation is applied to the training partition
only, so a held-out galaxy contributes exactly one row: its real observed
spectrum, with no synthetic copy of it anywhere in training.

Every galaxy in both sample files was checked against the split used to train
the released weights: 1 of 1 and 24 of 24 are on the held-out side, with zero
training overlap. So the numbers these notebooks print are real held-out
performance, not a model recognising something it has seen.
```

The 24-galaxy sample deliberately reaches z ~ 9.4, past where the training data
is dense. That is where most of the model's failures are, and hiding them would
make the tutorials useless.

## Two things the notebooks will tell you, stated here too

**The model does not predict absolute flux scale.** Training standardises each
spectrum independently, so the mapping learned is shape-to-shape, and output
comes back on the *input's* scale. Comparing against a real grating spectrum
requires standardising both sides — which is why every figure in the paper is
labelled $F_\lambda$ *(normalized)*.

**These weights are un-tuned.** Hyperparameters come from an early sweep, and
the released checkpoints are known to be selected on a criterion that does not
track redshift accuracy. The numbers the notebooks print are the model's honest
current behaviour and are expected to improve.

## Reproducing the outputs

The notebooks are committed with their outputs so they read correctly on GitHub.
To re-execute them all:

```bash
export SPECSR_CHECKPOINT_DIR=../checkpoints/release   # or omit, to use the Hub
for nb in 0*.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

Numbers will shift slightly between GPU and CPU, and will shift substantially
whenever the published weights are retrained.
